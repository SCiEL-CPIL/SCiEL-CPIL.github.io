"""Model-and-optimize: Fast circuit simulation with Nelder-Mead + L-BFGS-B hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 145.0
_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix using vectorized bit manipulation."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    idx = np.arange(dim)
    
    # Diagonal: -coupling * sum Z_i Z_{i+1}
    diag = np.zeros(dim, dtype=np.float64)
    for i in range(n - 1):
        bi = (idx >> i) & 1
        bp = (idx >> (i + 1)) & 1
        zi = 1.0 - 2.0 * bi
        zp = 1.0 - 2.0 * bp
        diag += -coupling * zi * zp
    H[np.arange(dim), np.arange(dim)] = diag
    
    # Off-diagonal: -field * sum X_i
    for i in range(n):
        other = idx ^ (1 << i)
        H[idx, other] += -field
    
    return H


def _get_ground_energy(n, coupling, field):
    """Compute the exact ground energy only."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _circuit_forward(params, n, layers):
    """
    Forward pass using vectorized operations.
    State is stored as complex array indexed by bitstring.
    Qubit i corresponds to bit i (LSB = qubit 0).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    thetas = params.reshape(layers, n)
    
    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            if abs(theta) < 1e-15:
                continue
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            bit_mask = 1 << q
            idx = np.arange(dim)
            mask0 = (idx & bit_mask) == 0
            idx0 = idx[mask0]
            partner = idx0 | bit_mask
            
            old_0 = state[idx0].copy()
            old_1 = state[partner].copy()
            
            state[idx0] = c * old_0 - s * old_1
            state[partner] = s * old_0 + c * old_1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                bit_q = 1 << q
                bit_qp1 = 1 << (q + 1)
                idx = np.arange(dim)
                mask = (idx & bit_q) & (idx & bit_qp1)
                state[mask] *= -1.0
    
    return state


def _energy(params, n, coupling, field, layers, H, e0):
    """Compute mean energy gap per spin."""
    psi = _circuit_forward(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    return (energy - e0) / n


def _energy_fd(params, n, coupling, field, layers, H, e0, eps=1e-7):
    """Compute energy and finite-difference gradient."""
    val = _energy(params, n, coupling, field, layers, H, e0)
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (_energy(params_plus, n, coupling, field, layers, H, e0) - 
                   _energy(params_minus, n, coupling, field, layers, H, e0)) / (2 * eps)
    return val, grad


def _physics_informed_init(n, coupling, field, layers):
    """
    Generate physics-informed initial parameters.
    """
    inits = []
    
    # Base angle from field strength
    base_angle = min(np.pi / 2, field * 0.8)
    
    # Pattern 1: All layers same base angle
    x = np.full((layers, n), base_angle)
    inits.append(x.flatten())
    
    # Pattern 2: Alternating signs (staggered)
    x = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            sign = 1 if (q % 2 == 0) else -1
            x[l, q] = sign * base_angle
    inits.append(x.flatten())
    
    # Pattern 3: Layer-dependent (deeper layers larger angle)
    x = np.zeros((layers, n))
    for l in range(layers):
        x[l, :] = base_angle * (l + 1) / layers
    inits.append(x.flatten())
    
    # Pattern 4: Middle layer larger
    x = np.full((layers, n), base_angle * 0.5)
    x[1, :] = base_angle
    inits.append(x.flatten())
    
    # Pattern 5: Near pi/2 for strong field
    if field > 1.2:
        x = np.full((layers, n), np.pi / 2 - 0.1)
        inits.append(x.flatten())
    
    # Pattern 6: Small angles for weak field
    if field < 0.6:
        x = np.full((layers, n), 0.2)
        inits.append(x.flatten())
    
    # Random perturbations of base
    np.random.seed(42)
    for _ in range(15):
        x = base_angle + np.random.randn(layers, n) * 0.4
        x = np.clip(x, -np.pi, np.pi)
        inits.append(x.flatten())
    
    return inits


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    e0 = _get_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    np_params = layers * n
    
    def obj_only(x):
        return _energy(x, n, coupling, field, layers, H, e0)
    
    def jac_fd(x):
        return _energy_fd(x, n, coupling, field, layers, H, e0)[1]
    
    # Generate physics-informed restarts
    restarts = _physics_informed_init(n, coupling, field, layers)
    
    best_val = np.inf
    best_x = np.zeros(np_params)
    
    # Evaluate all restarts first to find best starting point
    for x0 in restarts:
        if time.time() - _start_time > _TIME_LIMIT - 15:
            break
        try:
            val = obj_only(x0)
            if val < best_val:
                best_val = val
                best_x = x0.copy()
        except Exception:
            pass
    
    # Nelder-Mead from best restart (derivative-free, robust to local structure)
    if time.time() - _start_time < _TIME_LIMIT - 20:
        try:
            res = opt.minimize(
                obj_only, best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # L-BFGS-B with finite-difference gradient for polish
    if time.time() - _start_time < _TIME_LIMIT - 15:
        try:
            res = opt.minimize(
                obj_only, best_x, jac=jac_fd,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Additional restarts from perturbed best solution if time allows
    if time.time() - _start_time < _TIME_LIMIT - 25:
        np.random.seed(99)
        for _ in range(10):
            if time.time() - _start_time > _TIME_LIMIT - 20:
                break
            x0 = best_x + np.random.randn(np_params) * 0.3
            x0 = np.clip(x0, -np.pi, np.pi)
            try:
                # Nelder-Mead for robustness
                res = opt.minimize(
                    obj_only, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Final polish with L-BFGS-B
    if time.time() - _start_time < _TIME_LIMIT - 10:
        try:
            res2 = opt.minimize(
                obj_only, best_x, jac=jac_fd,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-20, 'gtol': 1e-14, 'maxls': 50}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass
    
    theta = best_x.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
