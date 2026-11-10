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
        
        # Apply Ry: new_state = state_mat @ [[c, s], [-s, c]]^T? 
        # Ry = [[c, -s], [s, c]]
        # state_mat has shape (..., 2) where the last index is the qubit state (0 or 1).
        # Let state = a|0> + b|1>.
        # Ry @ [a, b]^T = [c*a - s*b, s*a + c*b]^T.
        # So new_a = c*a - s*b, new_b = s*a + c*b.
        # We can do: new_state_mat[:, 0] = c * state_mat[:, 0] - s * state_mat[:, 1]
        #           new_state_mat[:, 1] = s * state_mat[:, 0] + c * state_mat[:, 1]
        
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

def _cost(theta_flat, n, layers, H, E_exact):
    theta = theta_flat.reshape(layers, n)
    state = _apply_circuit(theta, n, layers)
    E_circuit = _energy(state, H)
    return (E_circuit - E_exact) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Compute Exact Ground Energy
    H = _get_hamiltonian(n, coupling, field)
    E_vals = np.linalg.eigvalsh(H)
    E_exact = E_vals[0]
    
    # 2. Define Cost Function
    def cost(theta_flat):
        return _cost(theta_flat, n, layers, H, E_exact)
    
    # 3. Optimization
    # We will use a combination of:
    # 1. A few smart initial guesses.
    # 2. L-BFGS-B for fast convergence from good starts.
    # 3. Differential Evolution for global search if time permits.
    
    best_theta = None
    best_cost = np.inf
    
    # Initial Guesses
    x0_list = []
    
    # 1. Zeros
    x0_list.append(np.zeros(layers * n))
    
    # 2. Last layer pi/2
    x0_2 = np.zeros(layers * n)
    x0_2[(layers-1)*n : layers*n] = np.pi / 2
    x0_list.append(x0_2)
    
    # 3. All pi/2
    x0_list.append(np.full(layers * n, np.pi / 2))
    
    # 4. Small random
    np.random.seed(42)
    x0_list.append(np.random.randn(layers * n) * 0.5)
    
    # 5. Random uniform
    x0_list.append(np.random.uniform(-np.pi, np.pi, layers * n))
    
    # 6. Based on field strength heuristic
    # For small field, we want to stay close to Z basis? No, X field wants +x.
    # For large field, we want |+> (pi/2).
    # For intermediate, maybe mix.
    # Let's try a few variations of pi/2 with small perturbations.
    for _ in range(3):
        x0_rand = np.full(layers * n, np.pi / 2) + np.random.randn(layers * n) * 0.1
        x0_list.append(x0_rand)
    
    # Run L-BFGS-B from each start
    for idx, x0 in enumerate(x0_list):
        if time.time() > deadline - 2:
            break
        
        try:
            res = scipy.optimize.minimize(
                cost, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_theta = res.x
        except Exception:
            pass
    
    # If we have time, run Differential Evolution for global optimization
    if time.time() < deadline - 5:
        bounds = [(-np.pi, np.pi)] * (layers * n)
        try:
            res = scipy.optimize.differential_evolution(
                cost, 
                bounds, 
                seed=42,
                maxiter=50,
                tol=1e-8,
                polish=True,
                workers=1
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_theta = res.x
        except Exception:
            pass
            
    # If still have time, do a few more random restarts with L-BFGS-B
    if time.time() < deadline - 5 and best_theta is not None:
        for i in range(10):
            if time.time() > deadline - 2:
                break
            x0_rand = np.random.uniform(-np.pi, np.pi, layers * n)
            try:
                res = scipy.optimize.minimize(
                    cost, 
                    x0_rand, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_theta = res.x
            except Exception:
                pass

    if best_theta is None:
        # Fallback to seed
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    return best_theta.reshape(layers, n)
