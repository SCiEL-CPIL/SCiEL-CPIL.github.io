import numpy as np
from scipy.optimize import minimize
import time

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

def _compute_energy(psi, H):
    """Compute expectation value <psi|H|psi>"""
    return np.real(np.vdot(psi, H @ psi))

def _compute_gradient_shift(thetas, n, coupling, field, H, cz_masks):
    """
    Compute gradient of energy with respect to theta parameters using the Parameter Shift Rule.
    dE/dtheta = 0.5 * (E(theta + pi/2) - E(theta - pi/2))
    """
    grad = np.zeros_like(thetas)
    shift = np.pi / 2
    
    for l in range(thetas.shape[0]):
        for i in range(thetas.shape[1]):
            theta_plus = thetas.copy()
            theta_plus[l, i] += shift
            psi_plus = _apply_circuit_vec(theta_plus, n, cz_masks)
            e_plus = _compute_energy(psi_plus, H)
            
            theta_minus = thetas.copy()
            theta_minus[l, i] -= shift
            psi_minus = _apply_circuit_vec(theta_minus, n, cz_masks)
            e_minus = _compute_energy(psi_minus, H)
            
            grad[l, i] = 0.5 * (e_plus - e_minus)
            
    return grad

def _physics_informed_init(n, coupling, field, layers):
    """
    Generate initial guesses based on physics of the Ising model.
    """
    inits = []
    
    # Strategy 1: Uniform rotation
    # For field > coupling, state is closer to |+> (theta=pi/2)
    # For field < coupling, state is closer to |0> (theta=0) or mixed
    # A good heuristic: theta = arctan(field/coupling) ?
    
    # Let's try a few deterministic inits
    # 1. All zeros
    inits.append(np.zeros((layers, n)))
    
    # 2. All pi/2
    inits.append(np.full((layers, n), np.pi/2))
    
    # 3. Based on field strength
    # If field is strong, we want to rotate towards X.
    # Ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
    # |+> is theta=pi/2.
    # |0> is theta=0.
    # For Ising, the ground state is a superposition.
    # Let's try theta = field / (coupling + field) * pi/2
    ratio = field / (coupling + field) if (coupling + field) != 0 else 0.5
    init_val = ratio * np.pi / 2
    inits.append(np.full((layers, n), init_val))
    
    # 4. Alternating signs (to capture potential anti-ferromagnetic correlations if coupling was positive, but here it's negative -ferro)
    # Wait, H = -J sum ZZ. J=1 > 0. So it's Ferromagnetic.
    # All spins want to align.
    
    # 5. Random inits
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
        
    # 6. Small random perturbations around the heuristic
    for _ in range(5):
        inits.append(np.full((layers, n), init_val) + np.random.uniform(-0.1, 0.1, (layers, n)))

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
        energy = _compute_energy(psi, H)
        return energy
    
    def objective_and_grad(x):
        thetas = x.reshape((layers, n))
        psi = _apply_circuit_vec(thetas, n, cz_masks)
        energy = _compute_energy(psi, H)
        grad = _compute_gradient_shift(thetas, n, coupling, field, H, cz_masks)
        return energy, grad.flatten()
    
    # Multi-start optimization with L-BFGS-B using exact gradients
    best_result = None
    best_val = np.inf
    
    inits = _physics_informed_init(n, coupling, field, layers)
    
    # Time limit check
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    for x0 in inits:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                x0.flatten(), 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.reshape((layers, n))
        except Exception:
            continue
    
    optimal_thetas = best_result
    if optimal_thetas is None:
        # Fallback to the best init if optimization failed
        optimal_thetas = inits[0]
        
    # Store in cache
    _optimization_cache[key] = optimal_thetas
    
    return optimal_thetas
