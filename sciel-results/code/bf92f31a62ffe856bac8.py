import numpy as np
import time
from scipy.optimize import minimize


def _cut_values(n, edges):
    m = 2 ** n
    states = np.arange(m, dtype=np.int64)
    bits = (states[:, None] >> np.arange(n)[None, :]) & 1
    z = 1 - 2 * bits
    cv = np.zeros(m, dtype=np.float64)
    for (i, j) in edges:
        cv += (z[:, i] != z[:, j]).astype(np.float64)
    return cv


def _expected_cut(n, cut_vals, gammas_betas):
    m = 2 ** n
    state = np.full(m, 1.0 / np.sqrt(m), dtype=np.complex128).reshape((2,) * n)
    for gamma, beta in gammas_betas:
        phase = np.exp(-1j * gamma * cut_vals)
        state = state.reshape(m) * phase
        state = state.reshape((2,) * n)
        if beta != 0.0:
            c = np.cos(beta); s = np.sin(beta)
            G = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            for axis in range(n):
                st = np.moveaxis(state, axis, 0)
                shp = st.shape
                st = st.reshape(2, -1)
                st = G @ st
                state = np.moveaxis(st.reshape(shp), 0, axis)
    probs = np.abs(state.reshape(m)) ** 2
    return float(np.dot(probs, cut_vals))


def angles(n, edges, p):
    t0 = time.time()
    call_deadline = 14.0  # 6 calls * 14s = 84s, well under 150s wall-clock
    cut_vals = _cut_values(n, edges)

    def f(x):
        return -_expected_cut(n, cut_vals, x.reshape(p, 2))

    best_x = np.tile([0.35, 0.25], (p, 1)).reshape(-1).astype(float)
    best_f = f(best_x)

    def timed_out():
        return (time.time() - t0) > (call_deadline - 1.0)

    # Coarse grid search to identify promising basins
    # Using a slightly finer grid than before (8x8) to improve initial points
    grid_g = np.linspace(0.0, np.pi, 8)
    grid_b = np.linspace(0.0, np.pi, 8)
    for g in grid_g:
        for b in grid_b:
            if timed_out():
                break
            cand = np.tile([g, b], (p, 1)).reshape(-1)
            val = f(cand)
            if val < best_f:
                best_f, best_x = val, cand.copy()
        if timed_out():
            break

    # Multi-start Nelder-Mead optimization
    # Nelder-Mead proved superior to L-BFGS-B in previous iterations for this task
    rng = np.random.default_rng(713)
    nstarts = 40  # Increased from 14 to 40 to explore more basins
    for k in range(nstarts):
        if timed_out():
            break
        # Generate random start points
        start = np.concatenate([
            rng.uniform(0.0, np.pi, size=p),
            rng.uniform(0.0, np.pi, size=p),
        ])
        
        # Also try starts near the current best to polish
        if k % 2 == 1 and k > 0:
             noise = rng.normal(0, 0.1, size=2*p)
             start = best_x + noise
             # Clip to valid range [0, pi]
             start = np.clip(start, 0.0, np.pi)

        try:
            res = minimize(f, start, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass
        if timed_out():
            break
    
    # Final polishing pass with tighter tolerances on the best point found
    if not timed_out():
        try:
            res = minimize(f, best_x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass

    return best_x.reshape(p, 2)
