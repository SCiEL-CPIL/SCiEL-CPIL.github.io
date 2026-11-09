import numpy as np
from scipy.optimize import minimize

# Cache for optimized parameters
_optimization_cache = {}

def _get_hamiltonian(n, coupling, field):
    """
    Construct the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    for state in range(dim):
        z_energy = 0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            z_energy += z_i * z_ip1
        H[state, state] += -coupling * z_energy
        
        for i in range(n):
            flipped_state = state ^ (1 << i)
            H[state, flipped_state] += -field

    return H

def _apply_circuit_vec(thetas, n, cz_masks):
    """
    Apply the circuit to the initial state |0...0> using vectorized operations.
    thetas: array of shape (layers, n)
    cz_masks: list of masks for CZ operations
    """
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    
    layers = thetas.shape[0]
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Reshape to isolate qubit i
            shape = [2**i, 2, 2**(n-i-1)]
            psi_reshaped = psi.reshape(shape)
            
            p0 = psi_reshaped[:, 0, :]
            p1 = psi_reshaped[:, 1, :]
            
            new_p0 = c * p0 - s * p1
            new_p1 = s * p0 + c * p1
            
            new_psi_reshaped = np.stack([new_p0, new_p1], axis=1)
            psi = new_psi_reshaped.reshape(-1)
            
        # Apply CZ if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                psi *= cz_masks[i]
                
    return psi

def _compute_energy(psi, H, n):
    """Compute expectation value <psi|H|psi>"""
    return np.real(np.vdot(psi, H @ psi))

def _compute_gradient(thetas, n, coupling, field, H, cz_masks, eps=1e-7):
    """
    Compute gradient of energy with respect to theta parameters using finite differences.
    thetas: array of shape (layers, n)
    Returns: gradient array of same shape
    """
    grad = np.zeros_like(thetas)
    for l in range(thetas.shape[0]):
        for i in range(thetas.shape[1]):
            theta_plus = thetas.copy()
            theta_plus[l, i] += eps
            psi_plus = _apply_circuit_vec(theta_plus, n, cz_masks)
            e_plus = _compute_energy(psi_plus, H, n)
            
            theta_minus = thetas.copy()
            theta_minus[l, i] -= eps
            psi_minus = _apply_circuit_vec(theta_minus, n, cz_masks)
            e_minus = _compute_energy(psi_minus, H, n)
            
            grad[l, i] = (e_plus - e_minus) / (2 * eps)
    
    return grad

def _physics_informed_init(n, coupling, field, layers):
    """
    Generate initial guesses based on physics of the Ising model.
    """
    inits = []
    
    # For strong field (field > 1), ground state is close to |+>^n
    if field > 1.0:
        # Start with small rotations toward |+>
        init = np.zeros((layers, n))
        init[-1, :] = np.pi / 4  # Last layer pushes toward |+>
        inits.append(init)
        
        # Alternative: symmetric initialization
        init2 = np.zeros((layers, n))
        init2[0, :] = np.pi / 8
        init2[1, :] = np.pi / 8
        init2[2, :] = np.pi / 4
        inits.append(init2)
    else:
        # For weak field, ground state has more Z-like character
        init = np.zeros((layers, n))
        init[-1, :] = np.pi / 2  # Last layer creates superposition
        inits.append(init)
        
        # Alternative
        init2 = np.zeros((layers, n))
        init2[0, :] = np.pi / 6
        init2[1, :] = np.pi / 3
        init2[2, :] = np.pi / 2
        inits.append(init2)
    
    # Zero initialization
    inits.append(np.zeros((layers, n)))
    
    # Random initializations
    np.random.seed(123)
    for _ in range(8):
        inits.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    return inits

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _optimization_cache:
        return _optimization_cache[key]
        
    H = _get_hamiltonian(n, coupling, field)
    
    # Precompute CZ masks
    cz_masks = []
    for i in range(n - 1):
        mask = np.ones(2**n, dtype=complex)
        for state in range(2**n):
            if ((state >> i) & 1) and ((state >> (i+1)) & 1):
                mask[state] = -1
        cz_masks.append(mask)
        
    def objective(x):
        thetas = x.reshape((layers, n))
        psi = _apply_circuit_vec(thetas, n, cz_masks)
        energy = _compute_energy(psi, H, n)
        return energy
    
    def objective_and_grad(x):
        thetas = x.reshape((layers, n))
        psi = _apply_circuit_vec(thetas, n, cz_masks)
        energy = _compute_energy(psi, H, n)
        grad = _compute_gradient(thetas, n, coupling, field, H, cz_masks)
        return energy, grad.flatten()
    
    # Multi-start optimization with L-BFGS-B
    best_result = None
    best_val = np.inf
    
    inits = _physics_informed_init(n, coupling, field, layers)
    
    for x0 in inits:
        try:
            res = minimize(
                objective_and_grad, 
                x0.flatten(), 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.reshape((layers, n))
        except Exception:
            continue
    
    optimal_thetas = best_result
    
    # Store in cache
    _optimization_cache[key] = optimal_thetas
    
    return optimal_thetas
