"""Rewritten program for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _get_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]

def _apply_cz_layer(state, n):
    dim = 1 << n
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for b in range(dim):
            if (b & mask_i) and (b & mask_ip1):
                state[b] *= -1.0
    return state

def _circuit_state(theta, n, layers):
    dim = 1 << n
    state = np.zeros((2,) * n, dtype=float)
    state[(0,) * n] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            state = np.moveaxis(state, i, 0)
            new_shape = (2,) + state.shape[1:]
            state = state.reshape(new_shape)
            
            old0 = state[0].copy()
            old1 = state[1].copy()
            state[0] = c * old0 - s * old1
            state[1] = s * old0 + c * old1
            
            state = state.reshape((2,) * n)
            state = np.moveaxis(state, 0, i)
        
        if l < layers - 1:
            state_flat = state.flatten()
            state_flat = _apply_cz_layer(state_flat, n)
            state = state_flat.reshape((2,) * n)
            
    return state.flatten()

def _energy(theta, n, coupling, field, H, E0):
    state = _circuit_state(theta, n, 3)
    e_val = np.dot(state, H @ state)
    return e_val

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _get_ground_energy(n, coupling, field)
    
    # Initial guess: product |+> state (seed)
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    
    best_theta = theta0.copy()
    best_energy = _energy(best_theta, n, coupling, field, H, E0)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: L-BFGS-B from seed
    if time.time() < deadline:
        try:
            res = opt.minimize(
                _energy, 
                theta0.flatten(), 
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Strategy 2: Multiple random restarts with L-BFGS-B
    num_restarts = 12
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r % 3 == 0:
            current_theta = rng.uniform(-np.pi, np.pi, size=(layers, n))
        elif r % 3 == 1:
            current_theta = best_theta + rng.normal(0, 0.3, size=(layers, n))
        else:
            # Try different structured initializations
            if r % 6 == 2:
                current_theta = np.zeros((layers, n))
                current_theta[-1] = rng.uniform(0, np.pi, size=n)
            else:
                current_theta = np.zeros((layers, n))
                current_theta[0] = rng.uniform(-np.pi/4, np.pi/4, size=n)
                current_theta[1] = rng.uniform(-np.pi/4, np.pi/4, size=n)
                current_theta[-1] = rng.uniform(np.pi/4, np.pi, size=n)
        
        try:
            res = opt.minimize(
                _energy, 
                current_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            continue
    
    # Strategy 3: Nelder-Mead from best found so far (if time permits)
    if time.time() < deadline - 15:
        try:
            res = opt.minimize(
                _energy, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Final polish with L-BFGS-B from best
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _energy, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Ensure output is finite real array
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = theta0
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
