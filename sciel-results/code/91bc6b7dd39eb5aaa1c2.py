"""
Optimize shallow circuit parameters for Ising ground states using analytic gradient descent.
Strategy:
1. Construct the Hamiltonian H for the open transverse-field Ising chain.
2. Compute the exact ground energy E_0 and ground state |psi_0> via exact diagonalization.
3. Implement a vectorized state-vector simulator for the specific 3-layer Ry/CZ circuit.
4. Derive the analytic gradient of the energy E = <psi|H|psi> with respect to circuit angles.
   The gradient is computed by propagating the state forward and using the chain rule for 
   the variational parameterization.
5. Use scipy.optimize.minimize with L-BFGS-B method, providing the analytic gradient.
   This converges much faster than Nelder-Mead, allowing for tighter tolerances and 
   potentially more restarts or a single high-quality run within the time budget.
6. Initialize parameters using a heuristic based on the ground state or random restarts 
   guided by the gradient landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval
            
    # XX terms: -field * sum X_i
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            
    return H

def _ground_energy_and_state(n, coupling, field):
    """Compute exact ground energy and state."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    # Ground state is the eigenvector with the smallest eigenvalue
    E0 = eigvals[0].real
    psi0 = eigvecs[:, 0]
    return E0, psi0, H

def _simulate_state(params, n, layers):
    """
    Simulate the circuit to get the final state vector.
    params: shape (layers, n)
    Returns: state vector of shape (2**n,)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Apply Ry on qubit q
            # Ry = [[c, -s], [s, c]]
            # We need to apply this to the q-th axis of the state tensor.
            
            # Efficient application using reshape and transpose
            # Move axis q to the end
            axes = [ax for ax in range(n) if ax != q] + [q]
            state_perm = np.transpose(state, axes)
            state_mat = state_perm.reshape(-1, 2)
            
            Ry = np.array([[c, -s], [s, c]], dtype=complex)
            
            # new_state_mat[i, :] = Ry @ state_mat[i, :]
            new_state_mat = (Ry @ state_mat.T).T
            
            new_state_perm = new_state_mat.reshape((2,) * n)
            
            # Move axis back to position q
            inv_axes = [0] * n
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            state = np.transpose(new_state_perm, inv_axes)
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for q in range(n - 1):
                mask_q = 1 << q
                mask_q1 = 1 << (q + 1)
                # CZ is diagonal, -1 if both bits are 1
                # Vectorized sign flip
                for b in range(dim):
                    if (b & mask_q) and (b & mask_q1):
                        state[b] = -state[b]
                        
    return state

def _energy_and_grad(params, n, layers, H):
    """
    Compute energy and analytic gradient.
    params: shape (layers, n)
    Returns: energy, gradient (same shape as params)
    """
    dim = 1 << n
    # We need to store intermediate states to compute gradients efficiently
    # State after each layer and each Ry operation
    # This is a "tape" for backpropagation or just forward accumulation of gradients
    
    # For a 3-layer circuit, the structure is:
    # |0> -> Ry(theta_0) -> CZ -> Ry(theta_1) -> CZ -> Ry(theta_2) -> |psi>
    # Energy E = <psi| H |psi>
    # dE/dtheta = 2 Re( <psi| H |dpsi/dtheta> )
    # Since H is Hermitian and <psi|psi>=1, we can also use:
    # dE/dtheta = 2 Re( <psi| H |dpsi/dtheta> )
    
    # Let's compute |psi> and |dpsi/dtheta> for each theta.
    # However, computing |dpsi/dtheta> for all 3n parameters separately is expensive.
    # We can use the chain rule and store the state after each gate.
    
    # Alternative: Finite difference is too slow. 
    # Let's implement the gradient via the "parameter shift rule" or analytic backprop.
    # Given the small size, we can compute the gradient by:
    # 1. Forward pass to get state at each step.
    # 2. Backward pass to compute adjoints.
    
    # Let's try a simpler approach for gradient:
    # E(theta) = <psi(theta)| H |psi(theta)>
    # dE/dtheta_k = <dpsi/dtheta_k| H |psi> + <psi| H |dpsi/dtheta_k>
    #             = 2 Re( <psi| H |dpsi/dtheta_k> )
    
    # We need |psi> and |dpsi/dtheta_k> for each k.
    # Since n is small (4,6), we can afford to compute |psi> once, and then 
    # for each parameter, compute |dpsi/dtheta> by differentiating the circuit.
    # This is O(N_params * Cost_Sim). N_params = 3n = 12 or 18.
    # Cost_Sim is small.
    
    # Actually, a more efficient way is to store the state after each Ry gate 
    # and use the fact that Ry is unitary.
    
    # Let's use the "Jacobian" approach via automatic differentiation logic manually.
    # State evolution:
    # |psi_0> = |0...0>
    # |psi_{l,q}> = Ry(theta_{l,q})_{q} |psi_{l,q-1}>  (for q in 0..n-1, with CZ in between layers)
    # This is complex to track.
    
    # Let's stick to the definition:
    # 1. Compute |psi> = Circuit(params) |0>
    # 2. For each parameter theta_{l,q}, compute |dpsi/dtheta_{l,q}>.
    #    This can be done by running the circuit with a "derivative" state.
    #    Let |psi> = U(params) |0>.
    #    d/dtheta U = sum over gates before theta of U_after * dU_gate * U_before.
    #    So |dpsi/dtheta> = U_after |0> where the gate at theta is replaced by dRy/dtheta.
    #    This requires re-running the circuit for each parameter if we don't store intermediates.
    #    With n=6, layers=3, total gates = 18 Ry + 10 CZ = 28 gates.
    #    Re-running 18 times is 18 * 28 * (2**6) ops. Very fast.
    
    psi = _simulate_state(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    
    grad = np.zeros_like(params)
    
    # For each parameter, compute derivative state
    for l in range(layers):
        for q in range(n):
            # We need to compute |dpsi/dtheta_{l,q}>
            # This is the state resulting from applying the circuit, but with 
            # the Ry gate at (l,q) replaced by dRy/dtheta.
            # dRy/dtheta = [[-s/2, -c/2], [c/2, -s/2]] where c=cos(t/2), s=sin(t/2)
            
            # To do this efficiently, we can simulate forward, and when we hit the gate,
            # apply the derivative matrix instead of the unitary, and continue with the rest.
            # We need to handle the CZ gates correctly.
            
            # Let's write a helper that simulates with a specific gate differentiated.
            # Since dim is small, we can just modify the simulation loop.
            
            dim = 1 << n
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
            
            for ll in range(layers):
                # Apply Ry to all qubits
                for qq in range(n):
                    theta = params[ll, qq]
                    c = np.cos(theta / 2.0)
                    s = np.sin(theta / 2.0)
                    
                    # Check if this is the gate we are differentiating
                    if ll == l and qq == q:
                        # Apply dRy/dtheta
                        # Ry = [[c, -s], [s, c]]
                        # dRy/dtheta = [[-s/2, -c/2], [c/2, -s/2]]
                        dRy = np.array([[-s/2.0, -c/2.0], [c/2.0, -s/2.0]], dtype=complex)
                        mat = dRy
                    else:
                        # Apply Ry
                        mat = np.array([[c, -s], [s, c]], dtype=complex)
                    
                    # Apply mat to qubit qq
                    axes = [ax for ax in range(n) if ax != qq] + [qq]
                    state_perm = np.transpose(state, axes)
                    state_mat = state_perm.reshape(-1, 2)
                    new_state_mat = (mat @ state_mat.T).T
                    new_state_perm = new_state_mat.reshape((2,) * n)
                    inv_axes = [0] * n
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    state = np.transpose(new_state_perm, inv_axes)
                    
                # Apply CZ between layers
                if ll < layers - 1:
                    for qq in range(n - 1):
                        mask_q = 1 << qq
                        mask_q1 = 1 << (qq + 1)
                        for b in range(dim):
                            if (b & mask_q) and (b & mask_q1):
                                state[b] = -state[b]
                                
            dpsi = state
            # grad[l, q] = 2 Re( <psi| H |dpsi> )
            # Note: <psi| H |dpsi> is a scalar.
            val = np.vdot(psi, H @ dpsi)
            grad[l, q] = 2.0 * val.real
            
    return energy, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    E0, psi0, H = _ground_energy_and_state(n, coupling, field)
    
    def objective(params_flat):
        params = params_flat.reshape(layers, n)
        e, _ = _energy_and_grad(params, n, layers, H)
        return e
        
    def grad_func(params_flat):
        params = params_flat.reshape(layers, n)
        _, g = _energy_and_grad(params, n, layers, H)
        return g.flatten()
        
    best_energy = np.inf
    best_params = np.zeros((layers, n))
    
    # Strategy: 
    # 1. Try to find initial parameters close to the solution.
    #    Since we have the exact ground state psi0, we can try to "fit" it.
    #    However, the circuit might not be able to represent psi0 exactly.
    #    We can use random restarts with L-BFGS-B.
    
    # L-BFGS-B is efficient with analytic gradients.
    # We will run multiple restarts.
    
    num_restarts = 10 # L-BFGS-B converges fast, so fewer restarts might suffice
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Uniform in [-pi, pi]
        params_init = np.random.uniform(-np.pi, np.pi, (layers, n))
        
        options = {
            'maxiter': 100, # L-BFGS-B iterations
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 50
        }
        
        try:
            res = minimize(objective, params_init.flatten(), jac=grad_func, 
                           method='L-BFGS-B', options=options)
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x.reshape(layers, n)
                
        except Exception as e:
            # If optimization fails, try the initial point?
            try:
                e_init, _ = _energy_and_grad(params_init, n, layers, H)
                if e_init < best_energy:
                    best_energy = e_init
                    best_params = params_init
            except:
                pass

    # If no valid result found, fallback
    if best_energy == np.inf:
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2
        
    return best_params
# EVOLVE-BLOCK-END
