import numpy as np
import time

# Global cache for optimized angles
_cache = {}

def _build_hamiltonian_terms(n, coupling, field):
    """
    Precompute the coefficients for the Hamiltonian terms.
    H = -coupling * sum_{i} Z_i Z_{i+1} - field * sum_i X_i
    
    We represent the state as a vector of length 2^n.
    The Hamiltonian action on a state vector can be decomposed into:
    1. Diagonal part (ZZ terms): D_k = -coupling * sum_{i} z_k(i) z_k(i+1)
    2. Off-diagonal part (X terms): Connects k to k^flip_i with weight -field
    """
    dim = 2 ** n
    idx = np.arange(dim)
    
    # Diagonal part: ZZ terms
    diag_vals = np.zeros(dim)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        diag_vals += -coupling * z_i * z_ip1
        
    # Off-diagonal part: X terms
    # We store the list of (source, target, weight) for X terms
    # But for speed, we can just apply the flips.
    # X_i flips bit i.
    # <psi| H |psi> = sum_k psi^*_k (D_k psi_k + sum_i -field psi_{k^flip_i})
    
    return diag_vals

def _circuit_energy_and_grad(params, n, coupling, field, layers, grad_eps=1e-4):
    """
    Calculate energy and approximate gradient using finite differences.
    To be faster, we could implement analytic gradients, but for n=6,
    finite differences on 18 parameters with vectorized state evolution is feasible.
    However, analytic gradients are much better. Let's implement analytic 
    gradients via the parameter-shift rule or direct differentiation.
    
    Actually, for a fixed circuit, we can compute the gradient of <H> wrt theta
    using the chain rule.
    d/dtheta <psi(theta)|H|psi(theta)> = 2 Re[ <dpsi/dtheta| H |psi> ]
    dpsi/dtheta = i/2 * sum_j Z_j Ry_j(theta) ... this is complex.
    
    Alternative: Use the fact that Ry(theta) is a rotation.
    Let's stick to a robust, fast finite-difference gradient if analytic is too complex to code correctly in short time, 
    BUT we can do better: Central differences with a small step, vectorized over parameters?
    No, we need to evaluate the circuit for each perturbed parameter.
    
    Better Approach: 
    Since the circuit is shallow (3 layers), we can compute the gradient analytically 
    by tracking the state evolution.
    
    Let's implement a fast state evolution and use central differences for gradient.
    With n=6, dim=64. State evolution is very fast.
    18 params * 2 evals * 3 layers * (n Ry + n-1 CZ) ops.
    This is extremely fast.
    """
    def _eval_energy(p):
        thetas = p.reshape(layers, n)
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0
        
        for l in range(layers):
            # Apply Ry to all qubits
            for i in range(n):
                theta = thetas[l, i]
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                
                # Vectorized Ry application on flat state
                # We need to group indices by bit i
                # This is slow in a loop. Let's precompute masks.
                # For n=6, we can precompute the indices for each qubit flip.
                
            # Apply CZ
            if l < layers - 1:
                # CZ flips sign if both bits are 1
                pass
                
        # Compute energy
        pass

    # The above is a sketch. Let's implement a truly vectorized state evolution.
    
    dim = 2 ** n
    idx = np.arange(dim)
    
    # Precompute bit masks for Ry
    # For each qubit i, we need to apply Ry to the subspace where bit i is 0 and 1.
    # We can reshape the state to (2,)*n.
    # But reshaping and moving axes is slow if done in a loop.
    
    # Let's precompute the permutation or just use the tensor method which is clean.
    
    def _get_energy(p):
        thetas = p.reshape(layers, n)
        state = np.zeros([2] * n, dtype=complex)
        state[tuple([0] * n)] = 1.0
        
        for l in range(layers):
            for i in range(n):
                theta = thetas[l, i]
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                axis = n - 1 - i
                state = np.moveaxis(state, axis, -1)
                old_0 = state[..., 0]
                old_1 = state[..., 1]
                new_0 = c * old_0 - s * old_1
                new_1 = s * old_0 + c * old_1
                new_state = np.empty_like(state)
                new_state[..., 0] = new_0
                new_state[..., 1] = new_1
                state = np.moveaxis(new_state, -1, axis)
                
            if l < layers - 1:
                flat_state = state.reshape(-1)
                for i in range(n - 1):
                    mask_i = 1 << i
                    mask_ip1 = 1 << (i + 1)
                    cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                    flat_state[cond] *= -1
                state = flat_state.reshape([2] * n)
        
        # Energy calculation
        flat_state = state.reshape(-1)
        probs = np.abs(flat_state) ** 2
        
        # ZZ
        e_zz = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            e_zz += np.sum(probs * z_i * z_ip1)
            
        # X
        e_x = 0.0
        for i in range(n):
            mask = 1 << i
            idx_flipped = idx ^ mask
            term = np.sum(np.conj(flat_state) * flat_state[idx_flipped])
            e_x += np.real(term)
            
        return -coupling * e_zz - field * e_x

    # Compute energy
    e0 = _get_energy(params)
    
    # Compute gradient using central differences
    # To save time, we can use a smaller step or one-sided differences if needed, 
    # but central is more accurate.
    # With 18 params, 36 evaluations. Each eval is ~3*6*2 ops on 64-dim vectors. 
    # This is negligible.
    
    grad = np.zeros_like(params)
    h = 1e-5
    for i in range(len(params)):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] -= h
        grad[i] = (_get_energy(p_plus) - _get_energy(p_minus)) / (2 * h)
        
    return e0, grad

def _optimize(n, coupling, field, layers, timeout=115.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # Symmetry Reduction:
    # The Ising chain with open boundaries has reflection symmetry.
    # The ground state is symmetric (or anti-symmetric, but ground is symmetric).
    # The circuit can be constrained to be symmetric: theta[l, i] = theta[l, n-1-i].
    # This reduces parameters from n to ceil(n/2) per layer.
    # For n=4, params: 3*2=6 instead of 12.
    # For n=6, params: 3*3=9 instead of 18.
    # This makes gradient descent much more effective.
    
    # Let's implement symmetric parameterization.
    # We will optimize a reduced vector and expand it.
    
    n_sym = (n + 1) // 2
    n_sym_params = layers * n_sym
    
    def _expand_params(sym_params):
        # sym_params is (layers, n_sym)
        full_params = np.zeros((layers, n))
        for l in range(layers):
            for i in range(n_sym):
                val = sym_params[l, i]
                full_params[l, i] = val
                full_params[l, n - 1 - i] = val
        return full_params.flatten()

    def _symmetric_energy_grad(sym_p):
        full_p = _expand_params(sym_p)
        e, g = _circuit_energy_and_grad(full_p, n, coupling, field, layers)
        
        # Project gradient back to symmetric space
        # g_full is (layers, n)
        # g_sym[l, i] = g_full[l, i] + g_full[l, n-1-i]
        g_sym = np.zeros((layers, n_sym))
        for l in range(layers):
            for i in range(n_sym):
                g_sym[l, i] = g_full[l, i] + g_full[l, n - 1 - i]
        return e, g_sym.flatten()

    # Initial guess
    # Use a simple heuristic: small angles
    init_sym = np.zeros(n_sym_params)
    
    # Gradient Descent with Momentum (Adam-like)
    m = np.zeros(n_sym_params)
    v = np.zeros(n_sym_params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.1
    momentum = 0.9
    
    best_sym_p = init_sym.copy()
    best_e, _ = _symmetric_energy_grad(best_sym_p)
    
    t = 0
    while time.time() - start_time < timeout:
        e, g = _symmetric_energy_grad(best_sym_p)
        
        if e < best_e:
            best_e = e
            
        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))
        
        update = m_hat / (np.sqrt(v_hat) + eps)
        best_sym_p -= lr * update
        
        t += 1
        
        # Adaptive learning rate reduction
        if t % 100 == 0:
            lr *= 0.95
            if lr < 1e-5:
                break

    # Final polish with Nelder-Mead on the full parameter space starting from best symmetric
    # This helps escape the symmetric manifold if the ground state is slightly asymmetric 
    # (though for Ising it shouldn't be, numerical precision might benefit).
    # Actually, let's just do a few steps of NM on the symmetric space to be safe.
    
    # Convert best symmetric to full
    best_full = _expand_params(best_sym_p.reshape(layers, n_sym))
    
    # Optional: Local NM polish
    if time.time() - start_time < timeout - 10:
        import scipy.optimize as opt
        
        def obj_full(p):
            return _circuit_energy_and_grad(p, n, coupling, field, layers)[0]
            
        try:
            res = opt.minimize(
                obj_full, 
                best_full, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_e:
                best_full = res.x
        except:
            pass

    result_angles = best_full.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
