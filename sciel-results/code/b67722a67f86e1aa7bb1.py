import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_phase_cache = {}
_idx_cache = {}
_gate_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
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
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = float(eigvals[0])
    _e0_cache[key] = e0
    return e0


def _get_phases(n):
    key = n
    if key in _phase_cache:
        return _phase_cache[key]
    dim = 1 << n
    phases = []
    idx = np.arange(dim)
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        phase = np.ones(dim, dtype=np.float64)
        mask = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        phase[mask] = -1.0
        phases.append(phase)
    _phase_cache[key] = phases
    return phases


def _get_idx_pairs(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_pairs = {}
    idx = np.arange(dim)
    for q in range(n):
        mask = 1 << q
        idx_0 = idx[(idx & mask) == 0]
        idx_1 = idx_0 + mask
        idx_pairs[q] = (idx_0, idx_1)
    _idx_cache[key] = idx_pairs
    return idx_pairs


def _circuit_state_vec(n, thetas, layers):
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0j
    phases = _get_phases(n)
    idx_pairs = _get_idx_pairs(n)
    for layer in range(layers):
        for q in range(n):
            theta = float(thetas[layer, q])
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = idx_pairs[q]
            v0 = psi[idx_0].copy()
            v1 = psi[idx_1].copy()
            psi[idx_0] = c * v0 - s * v1
            psi[idx_1] = s * v0 + c * v1
        if layer < layers - 1:
            for i in range(n - 1):
                psi *= phases[i]
    return psi


def _circuit_energy_vec(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_vec(n, thetas, layers)
    return float(np.real(np.vdot(psi, H @ psi)))


def _symmap(params, n, layers):
    half_n = n // 2
    out = np.zeros((layers, n))
    for l in range(layers):
        for i in range(half_n):
            v = params[l * half_n + i]
            out[l, i] = v
            out[l, n - 1 - i] = v
    return out


def _obj_sym(params, n, coupling, field, layers, e0):
    th = _symmap(params, n, layers)
    e = _circuit_energy_vec(n, coupling, field, th, layers)
    return (e - e0) / n


def _obj_grad(params, n, coupling, field, layers, e0, eps=1e-5):
    th = _symmap(params, n, layers)
    e0_e = _circuit_energy_vec(n, coupling, field, th, layers)
    val = (e0_e - e0) / n
    grad = np.zeros_like(params)
    for k in range(params.size):
        pp = params.copy()
        pm = params.copy()
        pp[k] += eps
        pm[k] -= eps
        ep = _circuit_energy_vec(n, coupling, field, _symmap(pp, n, layers), layers)
        em = _circuit_energy_vec(n, coupling, field, _symmap(pm, n, layers), layers)
        grad[k] = ((ep - e0) - (em - e0)) / (2.0 * eps * n)
    return val, grad


def _heuristic_inits_sym(n, coupling, field, layers):
    half_n = n // 2
    inits = []
    inits.append(np.zeros(layers * half_n))
    inits.append(np.full(layers * half_n, np.pi / 2))
    inits.append(np.full(layers * half_n, np.pi))
    ratio = field / coupling if coupling != 0 else field
    theta_est = np.arctan(ratio) * 0.8
    inits.append(np.full(layers * half_n, theta_est))
    inits.append(np.full(layers * half_n, 0.5))
    inits.append(np.full(layers * half_n, 1.0))
    rng = np.random.RandomState(7)
    for _ in range(8):
        inits.append(rng.uniform(0, np.pi, size=(layers * half_n)))
    return inits


def _nm_wrap(params, n, coupling, field, layers, e0):
    return _obj_sym(params, n, coupling, field, layers, e0)


def angles(n, coupling, field, layers):
    start = time.time()
    deadline = start + 135.0  # keep a safety margin under 150s
    e0 = _exact_ground_energy(n, coupling, field)

    use_sym = (n % 2 == 0)
    if use_sym:
        dim_params = layers * (n // 2)
        obj = lambda p: _obj_sym(p, n, coupling, field, layers, e0)
        inits = _heuristic_inits_sym(n, coupling, field, layers)
    else:
        dim_params = layers * n
        obj = lambda p: (lambda t: (_circuit_energy_vec(n, coupling, field, t, layers) - e0) / n)(p.reshape(layers, n))
        inits = [np.zeros(dim_params), np.full(dim_params, np.pi / 2)]
        rng = np.random.RandomState(11)
        for _ in range(8):
            inits.append(rng.uniform(0, np.pi, size=(dim_params)))

    best_x = inits[0].copy()
    best_val = obj(best_x)

    # evaluate all initial points quickly
    for x0 in inits:
        if time.time() > deadline:
            break
        v = obj(x0)
        if v < best_val:
            best_val = v
            best_x = x0.copy()

    # First round: Nelder-Mead (derivative-free, robust local search) from best + a few others
    to_try = [best_x]
    seen = set()
    seen.add(tuple(np.round(best_x, 6)))
    for x0 in inits:
        t = tuple(np.round(x0, 6))
        if t not in seen:
            to_try.append(x0)
            seen.add(t)
            if len(to_try) >= 6:
                break

    for x0 in to_try:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                obj,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 6000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Second round: L-BFGS-B using finite-difference gradients from current best
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _obj_sym,
                best_x,
                args=(n, coupling, field, layers, e0),
                method='L-BFGS-B',
                jac=lambda p: _obj_grad(p, n, coupling, field, layers, e0)[1],
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 80}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish with tighter tolerances
    if time.time() < deadline - 3.0:
        try:
            res = minimize(
                _obj_sym,
                best_x,
                args=(n, coupling, field, layers, e0),
                method='L-BFGS-B',
                jac=lambda p: _obj_grad(p, n, coupling, field, layers, e0)[1],
                options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 120}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    if use_sym:
        full = _symmap(best_x, n, layers)
    else:
        full = best_x.reshape((layers, n))

    if not np.all(np.isfinite(full)):
        return np.zeros((layers, n))
    return full
