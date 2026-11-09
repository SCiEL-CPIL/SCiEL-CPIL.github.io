import numpy as np
import time
from scipy.optimize import minimize

# Pre-compute Hamiltonian terms for efficiency
# H = -J * sum Z_i Z_{i+1} - h * sum X_i
# We will represent the Hamiltonian as a list of local terms or just use the full matrix for n<=6
# Since n is small (4, 6), full matrix is fine, but MPS simulation is key for speed.

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    # Z_i Z_{i+1} is diagonal. Eigenvalue is +1 if bits are same, -1 if different.
    # So -coupling * (1 if same, -1 if diff)
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H[b, b] -= coupling
            else:
                H[b, b] += coupling
            
    # X terms: -field * sum X_i
    # X_i flips bit i.
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _mps_simulate_energy_and_grad(params, n, layers, H, E0):
    """
    Simulate the circuit using Matrix Product States (MPS) and compute energy and analytic gradient.
    
    The circuit is:
    Layer 0: Ry(theta_0) on all qubits
    CZ on adjacent pairs (0-1, 1-2, ..., n-2 to n-1)
    Layer 1: Ry(theta_1) on all qubits
    CZ on adjacent pairs
    Layer 2: Ry(theta_2) on all qubits
    
    We represent the state as a tensor network.
    Since the circuit is shallow and local, the bond dimension grows but remains manageable for n=6.
    However, implementing a full MPS backpropagation is complex. 
    Alternative: Since n is small (max 6), we can use the full state vector but optimize the gradient computation 
    using analytic backpropagation through the circuit gates, which is much faster than finite differences.
    
    Let's stick to full state vector but implement TRUE analytic gradients via backpropagation.
    This is simpler and robust for n=6.
    """
    dim = 1 << n
    # State is a vector of size dim
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # We need to store intermediate states or recompute?
    # For backprop, we store the state after each gate.
    # Number of gates: layers * n (Ry) + (layers-1) * (n-1) (CZ)
    # For n=6, layers=3: 18 Ry + 10 CZ = 28 gates.
    # Storing 28 vectors of size 64 is trivial.
    
    # Forward pass
    # We will store the state after each Ry gate, as CZ is diagonal and easy to handle in backward pass.
    # Actually, let's just store the state after every single-qubit gate (Ry).
    # The CZ gates are applied between layers.
    
    # Structure:
    # L0: Ry(0..n-1) -> CZ(0..n-2) -> L1: Ry(0..n-1) -> CZ(0..n-2) -> L2: Ry(0..n-1)
    
    # We'll store states after each Ry application.
    # state_0 = |0>^n
    # Apply Ry(0), store state_1
    # Apply Ry(1), store state_2
    # ...
    # Apply Ry(n-1), store state_n
    # Apply CZ(0,1), state changes
    # Apply CZ(1,2), state changes
    # ...
    # Apply CZ(n-2, n-1), state changes
    # Apply Ry(0) of next layer, store state...
    
    # This is getting messy to index. Let's use a list of states.
    states = [state.copy()]
    
    for l in range(layers):
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Apply Ry to qubit q
            # Ry = [[c, -s], [s, c]]
            # We need to apply this to the tensor component q.
            state_tensor = state.reshape([2] * n)
            # Move axis q to front for easier manipulation
            state_tensor = np.moveaxis(state_tensor, q, 0)
            
            new_tensor = np.empty_like(state_tensor)
            new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
            new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
            
            state_tensor = np.moveaxis(new_tensor, 0, q)
            state = state_tensor.reshape(dim)
            states.append(state.copy())
            
        # Apply CZ gates
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                # CZ applies -1 if both qubits are 1
                both_one = (indices & mask1) != 0
                both_one &= (indices & mask2) != 0
                state[both_one] *= -1
                states.append(state.copy())
    
    psi = states[-1]
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    
    # Backward pass for gradients
    # We want dE/dtheta[l, q]
    # E = <psi|H|psi>
    # dE/dpsi* = Hpsi
    # Let v = Hpsi
    # We propagate v backwards through the circuit.
    # The gradient for a parameter theta in a gate U(theta) is:
    # dE/dtheta = 2 Re[ <psi_after| (dU/dtheta U^dagger) |psi_before> * <psi_before| ... > ]
    # More simply, using the adjoint method:
    # Let |psi_k> be the state after gate k.
    # Let |v_k> be the backward vector.
    # |v_{end}> = H |psi_end>
    # For a unitary gate U_k: |psi_k> = U_k |psi_{k-1}>
    # |v_{k-1}> = U_k^dagger |v_k>
    # Gradient for parameter theta in U_k:
    # g = 2 Re[ <psi_{k-1}| (dU_k/dtheta) |v_k> ]  -- Wait, this is not quite right.
    # Correct formula:
    # dE = <dpsi|H|psi> + <psi|H|dpsi> = <dpsi|v> + <v|dpsi> = 2 Re[ <dpsi|v> ]
    # |dpsi> = |dpsi_k> where |dpsi_k> is the state derivative at step k.
    # For a parameter in gate k, |dpsi_k> = (dU_k/dtheta) |psi_{k-1}>
    # And |v> at step k is |v_k>.
    # So g_k = 2 Re[ <psi_{k-1}| (dU_k/dtheta)^dagger |v_k> ]
    
    # We need to propagate |v> backwards.
    v = Hpsi.copy()
    grad = np.zeros((layers, n))
    
    # We iterate through the gates in reverse order.
    # We need to know which gate was applied at each step to undo it and compute the gradient.
    
    # Let's reconstruct the sequence of gates.
    # The 'states' list has length 1 + layers*n + (layers-1)*(n-1)
    # Index 0: Initial state
    # Indices 1..n: After Ry layer 0
    # Indices n+1 .. n+(n-1): After CZ layer 0
    # Indices ...: After Ry layer 1
    # etc.
    
    # It is safer to just re-run the forward pass logic in reverse, keeping track of indices.
    
    # Let's define a helper to apply Ry and CZ in reverse.
    
    # We will iterate l from layers-1 down to 0.
    # For each l, we process the CZ gates (if any) in reverse, then the Ry gates in reverse.
    
    # Current v corresponds to the state after the last operation of layer l (or after CZ if l < layers-1).
    # Wait, the 'states' list ends with the state after the last Ry of the last layer.
    # So v starts as H @ psi_final.
    
    # We need to step back through the operations.
    
    # Let's create a list of operations in forward order to make reverse iteration easy.
    # op_list: list of tuples ('ry', l, q) or ('cz', q)
    # This is redundant but clear.
    
    ops = []
    for l in range(layers):
        for q in range(n):
            ops.append(('ry', l, q))
        if l < layers - 1:
            for q in range(n - 1):
                ops.append(('cz', q))
    
    # Now iterate ops in reverse.
    # We need the state BEFORE each operation to compute the gradient for that operation's parameter.
    # We have the 'states' list.
    # Let's map each op to the index in 'states' AFTER it.
    # op_index in ops corresponds to states[op_index + 1].
    
    # Let's just re-simulate the forward pass but store the states in a list corresponding to ops.
    # Actually, the 'states' list we built earlier already has this.
    # states[0] is before op 0.
    # states[1] is after op 0.
    # states[k] is after op k-1.
    # states[len(ops)] is the final state.
    
    # So for op i (0-indexed in ops), the state after is states[i+1], state before is states[i].
    # The backward vector v currently corresponds to the state after op i.
    # To compute gradient for op i:
    #   if op is Ry(l, q) with theta:
    #     dU/dtheta = 0.5 * [[-s, -c], [c, -s]]
    #     g = 2 Re[ <psi_before| (dU/dtheta)^dagger |v_after> ]
    #   if op is CZ:
    #     No parameter, just propagate v.
    #     v_before = U^dagger v_after = U v_after (since CZ is Hermitian and unitary)
    
    # We need to be careful with the direction of v.
    # v_after is the vector at the output of the gate.
    # v_before is the vector at the input of the gate.
    # v_before = U^dagger v_after.
    
    # Let's iterate i from len(ops)-1 down to 0.
    # v is currently the vector for the state after op i.
    # Initially, v = H @ psi_final, which is the vector after the last op.
    
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        psi_before = states[i]
        psi_after = states[i+1]
        
        if op[0] == 'ry':
            l, q = op[1], op[2]
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # dU/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # We need <psi_before| (dU/dtheta)^dagger |v>
            # Let's apply (dU/dtheta)^dagger to v on the qubit q.
            # v is a vector. We reshape it to a tensor.
            v_tensor = v.reshape([2] * n)
            v_tensor = np.moveaxis(v_tensor, q, 0)
            
            # (dU/dtheta)^dagger = 0.5 * [[-s, c], [-c, -s]]
            # Let A = (dU/dtheta)^dagger
            # new_v[0] = A[0,0] v[0] + A[0,1] v[1]
            # new_v[1] = A[1,0] v[0] + A[1,1] v[1]
            
            A_00 = -0.5 * s
            A_01 = 0.5 * c
            A_10 = -0.5 * c
            A_11 = -0.5 * s
            
            new_v = np.empty_like(v_tensor)
            new_v[0] = A_00 * v_tensor[0] + A_01 * v_tensor[1]
            new_v[1] = A_10 * v_tensor[0] + A_11 * v_tensor[1]
            
            v_tensor = np.moveaxis(new_v, 0, q)
            v_after_adj = v_tensor.reshape(dim)
            
            # Gradient = 2 Re[ <psi_before| v_after_adj> ]
            g = 2.0 * np.vdot(psi_before, v_after_adj).real
            grad[l, q] = g
            
            # Propagate v backwards through the gate
            # v_before = U^dagger v_after
            # U = [[c, -s], [s, c]]
            # U^dagger = [[c, s], [-s, c]]
            # We apply U^dagger to v (the original v, not the adjusted one)
            v_tensor_orig = v.reshape([2] * n)
            v_tensor_orig = np.moveaxis(v_tensor_orig, q, 0)
            
            new_v_prop = np.empty_like(v_tensor_orig)
            new_v_prop[0] = c * v_tensor_orig[0] + s * v_tensor_orig[1]
            new_v_prop[1] = -s * v_tensor_orig[0] + c * v_tensor_orig[1]
            
            v_tensor_orig = np.moveaxis(new_v_prop, 0, q)
            v = v_tensor_orig.reshape(dim)
            
        elif op[0] == 'cz':
            q = op[1]
            # CZ is diagonal: diag(1, 1, 1, -1) on qubits q, q+1
            # U^dagger = U
            # So v_before = U v_after
            # We apply CZ to v.
            v_tensor = v.reshape([2] * n)
            # Move q and q+1 to front? Or just use masks.
            # Masks are faster.
            mask1 = 1 << q
            mask2 = 1 << (q + 1)
            indices = np.arange(dim)
            both_one = (indices & mask1) != 0
            both_one &= (indices & mask2) != 0
            v[both_one] *= -1
            # v is already updated in place, which is v_before.
            # No gradient to compute.
            
    return energy - E0, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    E0 = _ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    best_result = None
    best_energy = np.inf
    
    # Multi-start L-BFGS-B
    # With analytic gradients, we can afford more restarts and iterations.
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    # Pre-allocate flat params
    flat_params_shape = (layers * n,)
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial point
        init_params = rng.uniform(-np.pi, np.pi, (layers, n))
        
        def objective(flat_params):
            if time.time() > deadline:
                return 1e10
            params = flat_params.reshape((layers, n))
            try:
                e, _ = _mps_simulate_energy_and_grad(params, n, layers, H, E0)
                return e
            except Exception:
                return 1e10
        
        def grad_func(flat_params):
            if time.time() > deadline:
                return np.zeros_like(flat_params)
            params = flat_params.reshape((layers, n))
            try:
                _, g = _mps_simulate_energy_and_grad(params, n, layers, H, E0)
                return g.flatten()
            except Exception:
                return np.zeros_like(flat_params)
        
        try:
            res = minimize(
                objective,
                init_params.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_result = res.x
                
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to zero
        return np.zeros((layers, n))
        
    return best_result.reshape((layers, n))
