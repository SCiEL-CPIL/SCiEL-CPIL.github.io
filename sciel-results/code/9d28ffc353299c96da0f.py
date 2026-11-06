import numpy as np
import scipy.optimize as opt
import time

# Global cache for results
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Precompute bit values for efficiency
    idx = np.arange(dim)
    
    # ZZ terms (Diagonal)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = -coupling * z_i * z_ip1
        H[idx, idx] += zz_val
        
    # X terms (Off-diagonal)
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        H[idx, idx_flipped] += -field
        H[idx_flipped, idx] += -field
        
    return H

def _get_exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state and energy.
    """
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    ground_energy = eigenvalues[0]
    return ground_state, ground_energy

def _compute_product_state_init(ground_state, n):
    """
    Compute optimal initial angles for a product state approximation 
    of the ground state. This serves as a high-quality initialization.
    For each qubit, we minimize the distance to the reduced density matrix 
    or simply project the state onto a product basis to find the best 
    single-qubit rotations.
    
    A simpler and robust method: 
    The circuit ends with Ry(theta). 
    If we could make the state a product state, the best theta_i 
    minimizes the energy of the single-qubit Hamiltonian restricted to that site 
    in the mean field of the rest.
    
    However, a very effective initialization for Ry-CZ-Ry... circuits is 
    derived from the fact that the final layer is just Ry.
    Let's use the exact ground state to find the best product state approximation.
    We can do this by minimizing the norm ||Psi_exact - Psi_product||^2 
    with respect to the angles of the last layer only, assuming the rest 
    is optimized to produce a state that looks like the ground state 
    up to the last rotation.
    
    Actually, a more direct approach for initialization:
    The ground state of the Ising model has strong correlations.
    The product state approximation with angles theta_i such that 
    <Z_i> matches the exact <Z_i> is a decent start.
    But we can do better: minimize the energy of the product state 
    |theta> = tensor Ry(theta_i)|0> under the full Hamiltonian.
    This is a convex-like problem for product states (mean field).
    """
    # We will minimize the energy of a product state |theta_1>...|theta_n>
    # E(theta) = <theta|H|theta>
    # This is much faster than the full circuit.
    
    def product_state_energy(params):
        thetas = params
        # Build product state vector
        # |0> -> Ry(theta)|0> = [cos(theta/2), sin(theta/2)]
        # State is tensor product.
        # Energy terms:
        # 1. -coupling * sum <Z_i Z_j>
        #    For product states, <Z_i Z_j> = <Z_i><Z_j>
        #    <Z_i> = cos(theta_i)
        # 2. -field * sum <X_i>
        #    <X_i> = sin(theta_i)
        
        mags_z = np.cos(thetas)
        mags_x = np.sin(thetas)
        
        e_zz = 0.0
        for i in range(n - 1):
            e_zz += mags_z[i] * mags_z[i+1]
            
        e_x = np.sum(mags_x)
        
        return -coupling * e_zz - field * e_x
        
    # Minimize this product state energy to get good initial angles
    # We use this as the initialization for the full circuit optimization.
    # Note: The full circuit has CZ gates, so it can do better than a product state.
    # But the product state angles are a very good "center" for the search.
    
    # Multiple starts for the product state minimization to be safe
    best_prod_params = None
    best_prod_energy = np.inf
    
    for _ in range(5):
        init = np.random.uniform(0, 2*np.pi, n)
        try:
            res = opt.minimize(product_state_energy, init, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12})
            if res.fun < best_prod_energy:
                best_prod_energy = res.fun
                best_prod_params = res.x
        except:
            pass
            
    if best_prod_params is None:
        best_prod_params = np.zeros(n)
        
    return best_prod_params

def _circuit_energy(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value for the given circuit parameters.
    Optimized implementation using matrix operations where possible.
    """
    thetas = params.reshape(layers, n)
    
    # Start with |0...0>
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    # Precompute indices for CZ
    idx = np.arange(2**n)
    
    for l in range(layers):
        # Apply Ry(theta) to all qubits
        # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        # We apply this sequentially or in parallel? 
        # Since they act on different qubits, they commute.
        # We can update the state vector in one go per qubit.
        
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Split state into even/odd indices for qubit i
            # Bit i is 0 for indices where (k >> i) & 1 == 0
            # Bit i is 1 for indices where (k >> i) & 1 == 1
            
            # Create masks
            mask_i = 1 << i
            # Indices where bit i is 0
            idx_0 = idx[(idx & mask_i) == 0]
            # Indices where bit i is 1
            idx_1 = idx[(idx & mask_i) != 0]
            
            # The partner index for flipping bit i is k ^ mask_i
            # For k in idx_0, partner is k ^ mask_i which is in idx_1
            # For k in idx_1, partner is k ^ mask_i which is in idx_0
            
            # We need to be careful with in-place updates.
            # Let's extract the amplitudes.
            amp_0 = state[idx_0].copy()
            amp_1 = state[idx_1].copy()
            
            new_amp_0 = c * amp_0 - s * amp_1
            new_amp_1 = s * amp_0 + c * amp_1
            
            state[idx_0] = new_amp_0
            state[idx_1] = new_amp_1
            
        # Apply CZ between adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                
                # CZ applies -1 if both bits are 1
                # Condition: (k & mask_i) != 0 AND (k & mask_ip1) != 0
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                state[cond] *= -1
                
    # Calculate Energy
    # E = -coupling * sum <Z_i Z_{i+1}> - field * sum <X_i>
    
    probs = np.abs(state) ** 2
    
    E_zz = 0.0
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = z_i * z_ip1
        
        E_zz += np.sum(probs * zz_val)
        
    E_1 = -coupling * E_zz
    
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        # <X_i> = sum_k psi*_k psi_{k^i}
        term = np.sum(np.conj(state) * state[idx_flipped])
        E_x += np.real(term)
        
    E_2 = -field * E_x
    
    total_energy = E_1 + E_2
    return total_energy

def _optimize(n, coupling, field, layers, timeout=145.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # 1. Get Exact Ground State for Initialization
    try:
        ground_state, ground_energy = _get_exact_ground_state(n, coupling, field)
        # Use product state minimization to get a very good initial guess
        init_prod = _compute_product_state_init(ground_state, n)
        
        # The circuit has `layers` of Ry.
        # If we set all layers to the same angle, it's like a stronger rotation.
        # But CZ gates mix things.
        # A good initialization is to distribute the "rotation" needed.
        # Since the final layer is just Ry, let's set the last layer to init_prod 
        # and the others to 0? Or distribute?
        # Let's try: Last layer = init_prod, others = 0.
        init1 = np.zeros(n_params)
        init1[-n:] = init_prod
        
        # Also try distributing: theta_l = init_prod / layers
        init2 = np.tile(init_prod / layers, layers)
        
        # And the previous heuristic: arccos of magnetizations
        # Compute magnetizations from exact ground state
        dim = 2 ** n
        idx = np.arange(dim)
        probs = np.abs(ground_state) ** 2
        mags = np.zeros(n)
        for i in range(n):
            mask = 1 << i
            bit_vals = (idx >> i) & 1
            z_vals = 1 - 2 * bit_vals
            mags[i] = np.sum(probs * z_vals)
            
        theta_mag = np.arccos(np.clip(mags, -1, 1))
        init3 = np.zeros(n_params)
        init3[-n:] = theta_mag
        
    except Exception:
        init1 = np.random.uniform(0, 2*np.pi, n_params)
        init2 = init1.copy()
        init3 = init1.copy()
        
    inits = [init1, init2, init3]
    
    best_params = inits[0]
    best_energy = _circuit_energy(best_params, n, coupling, field, layers)
    
    def objective(params):
        return _circuit_energy(params, n, coupling, field, layers)
        
    # 2. Global Search with Differential Evolution (if time permits)
    # DE is robust against local minima.
    # We have ~12-18 parameters. DE population size 10-20 is reasonable.
    # Max iterations limited by time.
    
    # First, do a quick local refinement of the inits to establish a baseline
    for init in inits:
        if time.time() - start_time > timeout - 5:
            break
        try:
            res = opt.minimize(objective, init, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # Check time
    current_time = time.time()
    if current_time - start_time < timeout - 10:
        # Run Differential Evolution
        # Bounds: [0, 2*pi]
        bounds = [(0, 2*np.pi)] * n_params
        
        # Population size: 5 * n_params is standard, but we are time constrained.
        # Let's use popsize=8
        try:
            # Wrap objective to check time
            def de_objective(x):
                if time.time() - start_time > timeout - 5:
                    return np.inf
                return objective(x)
                
            res_de = opt.minimize(de_objective, best_params, method='differential_evolution',
                                  bounds=bounds, 
                                  maxiter=20, # Low maxiter, rely on timeout
                                  popsize=8, 
                                  tol=1e-4,
                                  seed=42,
                                  polish=False) # Skip polish inside DE, we'll do it after
            
            if res_de.fun < best_energy:
                best_energy = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
            
    # 3. Final Local Polish
    current_time = time.time()
    if current_time - start_time < timeout - 2:
        try:
            res_lbfgs = opt.minimize(objective, best_params, method='L-BFGS-B',
                                     options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            if res_lbfgs.fun < best_energy:
                best_energy = res_lbfgs.fun
                best_params = res_lbfgs.x
        except Exception:
            pass
            
    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
