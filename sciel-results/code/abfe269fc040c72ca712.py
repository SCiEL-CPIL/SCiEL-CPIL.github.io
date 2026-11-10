import numpy as np
import scipy.optimize
import time

def _get_hamiltonian(n, coupling, field):
    """Construct the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    def op_on_qubit(op, i, n):
        # Qubit i is bit i. In standard kron order |q_{n-1}>...|q_0>,
        # qubit 0 is the least significant bit (rightmost).
        # So op on qubit i corresponds to position (n-1-i) in the kron list.
        ops = [I] * n
        ops[n-1-i] = op
        result = ops[0]
        for k in range(1, n):
            result = np.kron(result, ops[k])
        return result
    
    # Interaction terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        Zi = op_on_qubit(Z, i, n)
        Zj = op_on_qubit(Z, i+1, n)
        H -= coupling * (Zi @ Zj)
        
    # Field terms: -field * sum X_i
    for i in range(n):
        Xi = op_on_qubit(X, i, n)
        H -= field * Xi
        
    return H

def _apply_circuit(theta, n, layers):
    """Apply the circuit to |0>^n and return the state vector."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    # Precompute masks for CZ
    indices = np.arange(2 ** n)
    cz_masks = []
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        both = ((indices & mask_i) != 0) & ((indices & mask_j) != 0)
        cz_masks.append(both)
    
    def apply_ry(state, theta_val, qubit, n):
        c = np.cos(theta_val / 2)
        s = np.sin(theta_val / 2)
        # Ry(theta) = [[c, -s], [s, c]]
        
        # Tensor reshape
        state_tensor = state.reshape([2] * n)
        
        # Qubit i corresponds to axis (n - 1 - i)
        target_axis = n - 1 - qubit
        other_axes = [ax for ax in range(n) if ax != target_axis]
        
        # Permute so that target axis is last
        perm = other_axes + [target_axis]
        state_perm = np.transpose(state_tensor, perm)
        
        # Reshape to (2^{n-1}, 2)
        state_mat = state_perm.reshape(-1, 2)
        
        # Apply Ry: new_state = state_mat @ Ry.T
        # Ry = [[c, -s], [s, c]]
        # new_a = c*a - s*b, new_b = s*a + c*b
        new_state_mat = np.empty_like(state_mat)
        new_state_mat[:, 0] = c * state_mat[:, 0] - s * state_mat[:, 1]
        new_state_mat[:, 1] = s * state_mat[:, 0] + c * state_mat[:, 1]
        
        # Reshape back
        new_state_perm = new_state_mat.reshape(state_perm.shape)
        
        # Permute back
        inv_perm = np.argsort(perm)
        new_state_tensor = np.transpose(new_state_perm, inv_perm)
        
        return new_state_tensor.reshape(2 ** n)

    def apply_cz(state, mask):
        state[mask] *= -1
        return state

    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            state = apply_ry(state, theta[l, i], i, n)
        
        # Apply CZ between layers (not after last)
        if l < layers - 1:
            for i in range(n - 1):
                state = apply_cz(state, cz_masks[i])
                
    return state

def _energy(state, H):
    psi = state
    E = np.vdot(psi, H @ psi).real
    return E

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Compute Exact Ground Energy
    H = _get_hamiltonian(n, coupling, field)
    E_vals = np.linalg.eigvalsh(H)
    E_exact = E_vals[0]
    
    # 2. Define Cost Function
    def cost(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = _apply_circuit(theta, n, layers)
        E_circuit = _energy(state, H)
        return (E_circuit - E_exact) / n
    
    # 3. Optimization Strategy:
    # Use multiple initial guesses and gradient-based optimization (L-BFGS-B)
    # with analytical gradients via automatic differentiation or finite differences.
    # For small dimensions, L-BFGS-B with good initial guesses is very effective.
    
    candidates = []
    
    # Start 1: Zeros (|0>^n)
    x0_1 = np.zeros(layers * n)
    candidates.append(x0_1)
    
    # Start 2: Seed-like (|+>^n on last layer)
    x0_2 = np.zeros(layers * n)
    x0_2[(layers-1)*n : layers*n] = np.pi / 2
    candidates.append(x0_2)
    
    # Start 3: All pi/2 (|+>^n on all layers, but CZs interfere)
    x0_3 = np.full(layers * n, np.pi / 2)
    candidates.append(x0_3)
    
    # Start 4: Small random
    np.random.seed(42)
    x0_4 = np.random.randn(layers * n) * 0.5
    candidates.append(x0_4)
    
    # Start 5: Random uniform in [-pi, pi]
    x0_5 = np.random.uniform(-np.pi, np.pi, layers * n)
    candidates.append(x0_5)
    
    # Start 6: Based on field strength heuristic
    # For small field, closer to |0>; for large field, closer to |+>
    x0_6 = np.zeros(layers * n)
    scale = field / 2.0  # Normalize field
    for l in range(layers):
        for i in range(n):
            x0_6[l * n + i] = np.arcsin(np.clip(scale, -1, 1))
    candidates.append(x0_6)
    
    best_theta = None
    best_cost = np.inf
    
    # First pass: Quick Nelder-Mead to find good basins
    for idx, x0 in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        try:
            res = scipy.optimize.minimize(
                cost, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False, 'adaptive': True}
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass
    
    # Second pass: Refine with L-BFGS-B from best candidates
    if time.time() < deadline - 10 and best_theta is not None:
        # Use the best result and a few others as starting points for L-BFGS-B
        refine_starts = [best_theta]
        # Add a few random starts for diversity
        for i in range(3):
            x0_rand = np.random.uniform(-np.pi, np.pi, layers * n)
            refine_starts.append(x0_rand)
        
        for x0 in refine_starts:
            if time.time() > deadline - 5:
                break
            
            try:
                res = scipy.optimize.minimize(
                    cost, 
                    x0, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10, 'disp': False}
                )
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_theta = res.x.copy()
            except Exception:
                pass
    
    # Third pass: If we have time, do more random restarts with Nelder-Mead
    if time.time() < deadline - 10 and best_theta is not None:
        for i in range(8):
            if time.time() > deadline - 5:
                break
            x0_rand = np.random.uniform(-np.pi, np.pi, layers * n)
            try:
                res = scipy.optimize.minimize(
                    cost, 
                    x0_rand, 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'disp': False, 'adaptive': True}
                )
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_theta = res.x.copy()
            except Exception:
                pass
    
    if best_theta is None:
        # Fallback to seed
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    return best_theta.reshape(layers, n)

# EVOLVE-BLOCK-END
