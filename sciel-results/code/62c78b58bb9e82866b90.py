"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal (each pair added once)
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:  # only add each pair once
                H[b, b_flip] += -field
                H[b_flip, b] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _circuit_state_vector(n, theta, layers):
    """
    Build the state vector from the circuit.
    theta: (layers, n) array of Ry angles
    Circuit: |0>^n, then for each layer l:
      - Apply Ry(theta[l, i]) to qubit i for all i
      - If not last layer, apply CZ on adjacent pairs (i, i+1) for i in ascending order
    """
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            c = np.cos(theta[l, i] / 2.0)
            s = np.sin(theta[l, i] / 2.0)
            mask = (1 << i)
            idx0 = np.array([j for j in range(1 << n) if (j & mask) == 0])
            idx1 = idx0 | mask
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_a = 1 << i
                mask_b = 1 << (i + 1)
                for idx in range(1 << n):
                    if ((idx & mask_a) != 0) and ((idx & mask_b) != 0):
                        state[idx] = -state[idx]
    
    return state


def _circuit_energy_and_grad(n, coupling, field, theta, layers, H=None, ground_energy=None):
    """
    Compute energy and analytic gradient via backpropagation.
    Returns (energy, grad) where grad has shape (layers, n).
    """
    if H is None:
        H = _build_hamiltonian(n, coupling, field)
    if ground_energy is None:
        eigvals = np.linalg.eigvalsh(H)
        ground_energy = float(eigvals[0])
    
    dim = 1 << n
    # Store all intermediate states for backpropagation
    states = []
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        states.append(state.copy())
        for i in range(n):
            c = np.cos(theta[l, i] / 2.0)
            s = np.sin(theta[l, i] / 2.0)
            mask = (1 << i)
            idx0 = np.array([j for j in range(dim) if (j & mask) == 0])
            idx1 = idx0 | mask
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        if l < layers - 1:
            for i in range(n - 1):
                mask_a = 1 << i
                mask_b = 1 << (i + 1)
                for idx in range(dim):
                    if ((idx & mask_a) != 0) and ((idx & mask_b) != 0):
                        state[idx] = -state[idx]
    
    # Final state
    final_state = state
    energy = np.vdot(final_state, H @ final_state).real
    objective_val = (energy - ground_energy) / n
    
    # Backpropagation to compute gradients
    # We need dE/dtheta[l,i] for each parameter
    # E = <psi|H|psi>, so dE = 2*Re(<dpsi|H|psi>)
    # where dpsi is the change in final state due to parameter change
    
    grad = np.zeros((layers, n), dtype=float)
    
    # Adjoint method: propagate adjoint state backwards
    # adjoint_final = H @ final_state
    adjoint = H @ final_state
    
    # Work backwards through the circuit
    # We need to track the adjoint through each gate
    # For each Ry gate, the gradient contribution is:
    # dE/dtheta = 2*Re(<adjoint| dRy/dtheta | state_before>)
    
    # Reconstruct states and compute gradients
    # We'll go layer by layer in reverse
    current_adjoint = adjoint.copy()
    
    for l in range(layers - 1, -1, -1):
        # Undo CZ gates (they are self-adjoint)
        if l < layers - 1:
            for i in range(n - 2, -1, -1):
                mask_a = 1 << i
                mask_b = 1 << (i + 1)
                for idx in range(dim):
                    if ((idx & mask_a) != 0) and ((idx & mask_b) != 0):
                        current_adjoint[idx] = -current_adjoint[idx]
        
        # Undo Ry gates in reverse order
        # We need the state before each Ry gate
        # Let's recompute states forward to have them available
        pass
    
    # Alternative: use finite-difference-like approach but more efficient
    # Actually, let's just use the direct method with stored states
    
    # Recompute all intermediate states
    all_states = []  # all_states[l][i] = state before applying Ry(l,i)
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            all_states.append((l, i, state.copy()))
            c = np.cos(theta[l, i] / 2.0)
            s = np.sin(theta[l, i] / 2.0)
            mask = (1 << i)
            idx0 = np.array([j for j in range(dim) if (j & mask) == 0])
            idx1 = idx0 | mask
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        if l < layers - 1:
            for i in range(n - 1):
                mask_a = 1 << i
                mask_b = 1 << (i + 1)
                for idx in range(dim):
                    if ((idx & mask_a) != 0) and ((idx & mask_b) != 0):
                        state[idx] = -state[idx]
    
    # Now compute gradients using the parameter-shift rule or direct differentiation
    # For Ry(theta) = cos(theta/2)*I - i*sin(theta/2)*Y
    # dRy/dtheta = -0.5*sin(theta/2)*I - 0.5i*cos(theta/2)*Y
    # 
    # Energy gradient: dE/dtheta = 2*Re(<psi_final| (dU/dtheta) U_rest^dag H U_rest |psi_init>)
    # This is complex. Let's use a simpler approach:
    # For each parameter, perturb it slightly and compute the energy change.
    # But that's O(N) evaluations. Instead, let's use the fact that for small n,
    # we can compute the gradient analytically.
    
    # Actually, for n=4,6 and 3 layers, let's just use a more careful backprop
    # The key insight: we can compute the gradient by propagating the adjoint state backwards
    
    # Let me redo this properly
    # Forward pass: store all states
    fwd_states = []  # State after each operation
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    fwd_states.append(state.copy())
    
    ops = []  # List of (type, layer, qubit) for each operation
    
    for l in range(layers):
        for i in range(n):
            ops.append(('ry', l, i))
            c = np.cos(theta[l, i] / 2.0)
            s = np.sin(theta[l, i] / 2.0)
            mask = (1 << i)
            idx0 = np.array([j for j in range(dim) if (j & mask) == 0])
            idx1 = idx0 | mask
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
            fwd_states.append(state.copy())
        
        if l < layers - 1:
            for i in range(n - 1):
                ops.append(('cz', l, i))
                mask_a = 1 << i
                mask_b = 1 << (i + 1)
                for idx in range(dim):
                    if ((idx & mask_a) != 0) and ((idx & mask_b) != 0):
                        state[idx] = -state[idx]
                fwd_states.append(state.copy())
    
    # Backward pass
    adj = H @ final_state  # adjoint state
    grad = np.zeros((layers, n), dtype=float)
    
    for op_idx in range(len(ops) - 1, -1, -1):
        op_type, l, i = ops[op_idx]
        state_before = fwd_states[op_idx]
        state_after = fwd_states[op_idx + 1]
        
        if op_type == 'ry':
            c = np.cos(theta[l, i] / 2.0)
            s = np.sin(theta[l, i] / 2.0)
            mask = (1 << i)
            idx0 = np.array([j for j in range(dim) if (j & mask) == 0])
            idx1 = idx0 | mask
            
            # dRy/dtheta applied to state_before
            # Ry = [[c, -s], [s, c]] in the subspace
            # dRy/dtheta = [[-0.5s, -0.5c], [0.5c, -0.5s]]
            dstate = np.zeros(dim, dtype=complex)
            dstate[idx0] = -0.5 * s * state_before[idx0] - 0.5 * c * state_before[idx1]
            dstate[idx1] = 0.5 * c * state_before[idx0] - 0.5 * s * state_before[idx1]
            
            # Gradient contribution
            grad[l, i] = 2.0 * np.real(np.vdot(adj, dstate))
            
            # Propagate adjoint backwards through Ry
            # adj_before = Ry^dag @ adj_after
            # Ry^dag = Ry(-theta) = [[c, s], [-s, c]]
            adj_new = adj.copy()
            adj_new[idx0] = c * adj[idx0] + s * adj[idx1]
            adj_new[idx1] = -s * adj[idx0] + c * adj[idx1]
            adj = adj_new
        
        elif op_type == 'cz':
            # CZ is self-adjoint and diagonal
            mask_a = 1 << i
            mask_b = 1 << (i + 1)
            adj_new = adj.copy()
            for idx in range(dim):
                if ((idx & mask_a) != 0) and ((idx & mask_b) != 0):
                    adj_new[idx] = -adj[idx]
            adj = adj_new
    
    return objective_val, grad


def _objective(params, n, coupling, field, layers, ground_energy, H):
    """Objective function for optimization."""
    theta = params.reshape((layers, n))
    energy = np.vdot(_circuit_state_vector(n, theta, layers), H @ _circuit_state_vector(n, theta, layers)).real
    return (energy - ground_energy) / n


def _objective_grad(params, n, coupling, field, layers, ground_energy, H):
    """Objective function with gradient for optimization."""
    theta = params.reshape((layers, n))
    obj_val, grad = _circuit_energy_and_grad(n, coupling, field, theta, layers, H, ground_energy)
    return obj_val, grad.reshape(-1)


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 140  # Leave safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    best_params = np.zeros(layers * n)
    best_score = float('inf')
    
    # Try multiple initializations with different strategies
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/4
    initializations.append(np.full(layers * n, np.pi / 4))
    
    # 3. Last layer pi/2
    init3 = np.zeros(layers * n)
    init3[-n:] = np.pi / 2
    initializations.append(init3)
    
    # 4. Physics-based interpolation
    init4 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            init4[l * n + i] = np.pi / 2 * min(1.0, field)
    initializations.append(init4)
    
    # 5-8. Random initializations with different seeds
    for seed in [42, 123, 7, 99]:
        np.random.seed(seed)
        initializations.append(np.random.uniform(-np.pi/2, np.pi/2, layers * n))
    
    # 9. Symmetric initialization based on field strength
    init9 = np.zeros(layers * n)
    base_angle = np.arctan(field / coupling) if coupling > 0 else np.pi/2
    for l in range(layers):
        scale = 1.0 - 0.3 * l
        for i in range(n):
            init9[l * n + i] = base_angle * scale
    initializations.append(init9)
    
    # 10. Another physics-inspired init
    init10 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            pos_factor = 0.8 + 0.2 * np.sin(np.pi * i / max(1, n-1))
            init10[l * n + i] = (np.pi / 2) * min(1.0, field) * pos_factor * (1.0 - 0.2*l)
    initializations.append(init10)
    
    # 11. Uniform small angles
    initializations.append(np.full(layers * n, 0.1))
    
    # 12. Alternating signs
    init12 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            init12[l * n + i] = 0.3 * (1 if i % 2 == 0 else -1)
    initializations.append(init12)
    
    # Run L-BFGS-B with gradient from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _objective_grad,
                init,
                args=(n, coupling, field, layers, ground_energy, H),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 300,
                    'maxfun': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Final polish with Nelder-Mead from the best result
    if time.time() < deadline - 5:
        try:
            result4 = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, ground_energy, H),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 300,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            
            if result4.fun < best_score:
                best_score = result4.fun
                best_params = result4.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape((layers, n))
    return theta
# EVOLVE-BLOCK-END
