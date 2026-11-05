"""Optimized quantum circuit parameters for Ising ground states.
Strategy: Multi-start L-BFGS-B with exact parameter-shift gradients and mirror symmetry.
Exploits the smoothness of the energy landscape and the symmetry of the open-chain Hamiltonian.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_START_TIME = time.time()


def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i using Kronecker products."""
    total_dim = 2 ** n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    H = np.zeros((total_dim, total_dim), dtype=complex)

    # ZZ terms
    for i in range(n - 1):
        op = np.array([[1.0]])
        for q in range(n):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op

    # X terms
    for i in range(n):
        op = np.array([[1.0]])
        for q in range(n):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op

    return H


def _get_ground_energy(H):
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0].real)


def _circuit_state(params_sym, n, layers):
    """
    Build circuit state from symmetric parameters.
    params_sym shape: (layers, n//2)
    Full params shape: (layers, n) where p[l, i] = p[l, n-1-i] = params_sym[l, i]
    """
    total_dim = 2 ** n
    state = np.zeros(total_dim, dtype=complex)
    state[0] = 1.0

    # Precompute bit masks
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            # Map full index q to symmetric index
            sym_idx = q if q < n // 2 else n - 1 - q
            theta = params_sym[layer, sym_idx]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            new_state = np.zeros_like(state)
            for idx in range(total_dim):
                if (idx & mask) == 0:
                    new_state[idx] += c * state[idx]
                    new_state[idx | mask] += s * state[idx]
                else:
                    new_state[idx] += c * state[idx]
                    new_state[idx ^ mask] += -s * state[idx]
            state = new_state

        # Apply CZ between adjacent pairs
        if layer < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                for idx in range(total_dim):
                    if (idx & mask1) and (idx & mask2):
                        state[idx] = -state[idx]

    return state


def _circuit_energy(params_sym, n, coupling, field, layers, H):
    state = _circuit_state(params_sym, n, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _physics_seed_sym(n, coupling, field, layers):
    """
    Generate physics-based symmetric initial parameters.
    """
    params_sym = np.zeros((layers, n // 2))
    
    # Estimate local effective field ratio for unique sites
    # Sites are 0, 1, ..., n//2 - 1
    for i in range(n // 2):
        # Determine number of neighbors for site i in the full chain
        # If i < n//2, it's in the first half.
        # Neighbor count:
        # If i == 0: 1 neighbor (site 1)
        # Else: 2 neighbors (i-1, i+1)
        # Note: For even n, the middle bond is (n/2-1, n/2). 
        # Symmetric sites are 0..n/2-1.
        # Site 0 is boundary. Site n/2-1 is inner.
        if i == 0:
            neighbors = 1.0
        else:
            neighbors = 2.0
            
        ratio = field / (coupling * neighbors)
        base_angle = np.arctan(ratio)
        
        # Distribute across layers
        weights = np.array([0.7, 0.2, 0.1])
        for l in range(layers):
            params_sym[l, i] = base_angle * weights[l]
            
    return params_sym


def angles(n, coupling, field, layers):
    """Optimize circuit parameters for the given Ising chain."""
    global _START_TIME
    _START_TIME = time.time()

    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(H)
    
    # Symmetric parameter space
    n_sym = n // 2
    num_params = layers * n_sym
    
    # Bounds: [-pi/2, pi/2]
    bounds = [(-np.pi/2, np.pi/2)] * num_params
    
    def objective_sym(x_sym):
        p_sym = x_sym.reshape(layers, n_sym)
        e = _circuit_energy(p_sym, n, coupling, field, layers, H)
        return (e - ground_energy) / n

    def gradient_sym(x_sym):
        p_sym = x_sym.reshape(layers, n_sym)
        grad = np.zeros_like(x_sym)
        for i in range(num_params):
            p_plus = p_sym.copy()
            p_minus = p_sym.copy()
            
            # Apply shift to the specific parameter
            # Index i corresponds to layer = i // n_sym, site = i % n_sym
            l = i // n_sym
            s = i % n_sym
            
            p_plus[l, s] += np.pi / 2
            p_minus[l, s] -= np.pi / 2
            
            e_plus = _circuit_energy(p_plus, n, coupling, field, layers, H)
            e_minus = _circuit_energy(p_minus, n, coupling, field, layers, H)
            
            # Gradient of energy
            grad[i] = (e_plus - e_minus) / 2.0
            
            # Gradient of objective (divide by n)
            grad[i] /= n
            
        return grad

    # Generate diverse initial points in symmetric space
    candidates = []
    
    # 1. Physics-based
    candidates.append(_physics_seed_sym(n, coupling, field, layers).flatten())
    
    # 2. Zero
    candidates.append(np.zeros(num_params))
    
    # 3. Random small
    np.random.seed(42)
    candidates.append(np.random.uniform(-0.5, 0.5, size=num_params))
    
    # 4. Random larger
    np.random.seed(43)
    candidates.append(np.random.uniform(-1.0, 1.0, size=num_params))
    
    # 5. Uniform positive
    candidates.append(np.full(num_params, 0.3))
    
    # 6. Uniform negative
    candidates.append(np.full(num_params, -0.3))
    
    best_params_sym = None
    best_score = np.inf
    
    # Evaluate candidates to find best starting point
    for p in candidates:
        if not _check_deadline():
            break
        try:
            score = objective_sym(p)
            if score < best_score:
                best_score = score
                best_params_sym = p.copy()
        except Exception:
            continue
            
    if best_params_sym is None:
        best_params_sym = candidates[0]
        best_score = objective_sym(best_params_sym)

    # Multi-start L-BFGS-B with exact gradients
    # We have ~140s. Each L-BFGS-B run with exact gradients is fast.
    # We can afford 5-10 restarts.
    
    start_indices = list(range(len(candidates)))
    # Ensure we start from the best candidate first
    start_indices.sort(key=lambda i: objective_sym(candidates[i]))
    
    for idx in start_indices:
        if not _check_deadline(margin=10.0):
            break
            
        try:
            result = minimize(
                objective_sym,
                candidates[idx],
                jac=gradient_sym,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params_sym = result.x.copy()
                
        except Exception:
            continue

    # Reconstruct full symmetric parameters
    best_params_full = np.zeros((layers, n))
    p_sym_best = best_params_sym.reshape(layers, n_sym)
    for l in range(layers):
        for i in range(n_sym):
            val = p_sym_best[l, i]
            best_params_full[l, i] = val
            best_params_full[l, n - 1 - i] = val

    # Ensure finite output
    best_params_full = np.nan_to_num(best_params_full, nan=0.0, posinf=0.0, neginf=0.0)

    return best_params_full
# EVOLVE-BLOCK-END
