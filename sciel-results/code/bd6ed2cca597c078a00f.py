import numpy as np
import scipy.optimize as opt
import time

# Cache for optimized angles
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Precompute basis states bits for efficiency
    # idx is 0 to 2^n - 1
    # Bit i corresponds to (idx >> i) & 1
    
    # ZZ terms (Diagonal)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for k in range(dim):
            bit_i = (k >> i) & 1
            bit_ip1 = (k >> (i + 1)) & 1
            # Z eigenvalue is +1 for |0> (bit 0) and -1 for |1> (bit 1)
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[k, k] += -coupling * z_i * z_ip1
            
    # X terms (Off-diagonal)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            k_flipped = k ^ mask
            # X flips the bit. H[k, k_flipped] += -field
            # Since H is Hermitian, H[k_flipped, k] is also -field
            # We add to both to ensure symmetry, though adding once and symmetrizing is also valid.
            # Here we add directly.
            H[k, k_flipped] += -field
            H[k_flipped, k] += -field
            
    return H

def _get_exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state and energy.
    """
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Ground state is the first column
    ground_state = eigenvectors[:, 0]
    ground_energy = eigenvalues[0]
    return ground_state, ground_energy

def _compute_local_magnetizations(state, n):
    """
    Compute <Z_i> for each qubit i in the given state.
    """
    dim = 2 ** n
    probs = np.abs(state) ** 2
    idx = np.arange(dim)
    
    mags = np.zeros(n)
    for i in range(n):
        mask = 1 << i
        bit_vals = (idx >> i) & 1
        z_vals = 1 - 2 * bit_vals
        mags[i] = np.sum(probs * z_vals)
    return mags

def _circuit_energy(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value for the given circuit parameters.
    This function is optimized for speed using vectorized operations where possible.
    """
    thetas = params.reshape(layers, n)
    
    # Initial state |0>^n
    # Represent state as a tensor of shape (2,)*n for easy gate application
    # Index mapping: state[b_0, b_1, ..., b_{n-1}] corresponds to basis index sum b_i 2^i?
    # Standard numpy reshape of a vector of length 2^n:
    # Index k = b_{n-1} 2^{n-1} + ... + b_0 2^0
    # If we reshape to (2, 2, ..., 2), the first axis is the MSB (bit n-1).
    # Qubit i (bit i) corresponds to axis n-1-i in the tensor.
    
    state = np.zeros([2] * n, dtype=complex)
    # |0>^n means all bits are 0.
    # In the tensor, this is index [0, 0, ..., 0]
    state[tuple([0] * n)] = 1.0
    
    # Precompute Ry matrices? No, they depend on theta.
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Axis for qubit i
            axis = n - 1 - i
            
            # Move axis to end
            state = np.moveaxis(state, axis, -1)
            
            # Apply Ry:
            # |0> -> c|0> + s|1>
            # |1> -> -s|0> + c|1>
            # Wait, standard Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            # So new_0 = c*old_0 - s*old_1
            #    new_1 = s*old_0 + c*old_1
            
            old_0 = state[..., 0]
            old_1 = state[..., 1]
            
            new_0 = c * old_0 - s * old_1
            new_1 = s * old_0 + c * old_1
            
            # Construct new state
            # We need to be careful not to modify old_0/old_1 while assigning.
            # Since we create new arrays, this is safe.
            new_state_slice = np.empty_like(state)
            new_state_slice[..., 0] = new_0
            new_state_slice[..., 1] = new_1
            
            # Move axis back
            state = np.moveaxis(new_state_slice, -1, axis)
        
        # Apply CZ if not the last layer
        if l < layers - 1:
            # CZ is diagonal: flips sign if both qubits are 1.
            # We can do this by iterating over basis states or using masks on the flattened array.
            # Flattening is easier for masking.
            flat_state = state.reshape(-1)
            dim = 2 ** n
            idx = np.arange(dim)
            
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                flat_state[cond] *= -1
            
            state = flat_state.reshape([2] * n)
            
    # Calculate Energy
    # H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    
    flat_state = state.reshape(-1)
    dim = 2 ** n
    idx = np.arange(dim)
    probs = np.abs(flat_state) ** 2
    
    # ZZ Term
    E_zz = 0.0
    for i in range(n - 1):
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = z_i * z_ip1
        E_zz += np.sum(probs * zz_val)
        
    E_1 = -coupling * E_zz
    
    # X Term
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        # <psi| X_i |psi> = sum_k psi^*_k psi_{k^f}
        # Since X is real and symmetric, this is real.
        # sum_k psi^*_k psi_{k^f}
        term = np.sum(np.conj(flat_state) * flat_state[idx_flipped])
        E_x += np.real(term)
        
    E_2 = -field * E_x
    
    total_energy = E_1 + E_2
    return total_energy

def _optimize(n, coupling, field, layers, timeout=115.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # 1. Get exact ground state to inform initialization
    try:
        ground_state, ground_energy = _get_exact_ground_state(n, coupling, field)
        mags = _compute_local_magnetizations(ground_state, n)
        
        # Map magnetizations to angles for the LAST layer.
        # For a single qubit starting from |0>, Ry(theta) gives <Z> = cos(theta).
        # So theta = arccos(<Z>).
        # We assume the previous layers can be tuned to "prepare" the state such that
        # the final layer's simple rotation aligns with the local fields.
        # This is a heuristic, but provides a much better start than zeros or random.
        
        theta_last = np.arccos(np.clip(mags, -1, 1))
        
        # Initial Guess 1: Only last layer set
        init1 = np.zeros(n_params)
        init1[-n:] = theta_last
        
        # Initial Guess 2: All layers set to theta_last (symmetry breaking heuristic)
        init2 = np.tile(theta_last, layers)
        
        # Initial Guess 3: Zeros (baseline)
        init3 = np.zeros(n_params)
        
        # Initial Guess 4: Random
        init4 = np.random.uniform(0, 2*np.pi, n_params)
        
        inits = [init1, init2, init3, init4]
        
    except Exception:
        # Fallback to random if exact diagonalization fails (unlikely for n<=6)
        inits = [np.random.uniform(0, 2*np.pi, n_params)]
        
    best_params = inits[0]
    best_energy = _circuit_energy(best_params, n, coupling, field, layers)
    
    # 2. Optimize from each initial guess using Nelder-Mead
    # Nelder-Mead is derivative-free and handles the periodicity of angles well
    # without needing to wrap them explicitly in the objective (though wrapping helps).
    
    def objective(params):
        # Optional: Wrap angles to [-pi, pi] to help the optimizer?
        # Nelder-Mead works on the raw parameters.
        # The energy function is periodic.
        # We just return the energy.
        return _circuit_energy(params, n, coupling, field, layers)
        
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        try:
            res = opt.minimize(
                objective, 
                init, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # 3. Final polish with L-BFGS-B if time permits, starting from the best NM result
    # L-BFGS-B can converge faster to the local minimum if we are close.
    current_time = time.time()
    if current_time - start_time < timeout - 5:
        try:
            res_lbfgs = opt.minimize(
                objective, 
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_energy:
                best_energy = res_lbfgs.fun
                best_params = res_lbfgs.x
        except Exception:
            pass

    # Reshape to (layers, n)
    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    # Ensure types are consistent
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
