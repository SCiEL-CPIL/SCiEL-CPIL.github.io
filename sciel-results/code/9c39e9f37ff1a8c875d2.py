"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_cache = {}
_H_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _cache:
        return _cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    _cache[key] = e0
    return e0

def _circuit_state(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            new_state = np.zeros(dim, dtype=np.complex128)
            for idx in range(dim):
                bit = (idx >> q) & 1
                if bit == 0:
                    new_state[idx] += c * state[idx]
                else:
                    new_state[idx ^ (1 << q)] += s * state[idx]
            state = new_state
        if layer < layers - 1:
            for i in range(n - 1):
                m1 = 1 << i
                m2 = 1 << (i + 1)
                for idx in range(dim):
                    if (idx & m1) and (idx & m2):
                        state[idx] *= -1.0
    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115.0
    e0 = _exact_ground_energy(n, coupling, field)
    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)
    for init_seed in range(8):
        if time.time() > deadline:
            break
        np.random.seed(init_seed + 100)
        x_init = np.random.uniform(-1.5, 1.5, size=(layers * n))
        def obj(p):
            return _objective(p, n, coupling, field, layers, e0)
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
