"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _cut_values(n, edges):
    """Precompute cut values for all 2^n states."""
    m = 1 << n
    states = np.arange(m, dtype=np.int64)
    bits = (states[:, None] >> np.arange(n)[None, :]) & 1
    z = 1 - 2 * bits
    cv = np.zeros(m, dtype=np.float64)
    for (i, j) in edges:
        cv += (z[:, i] != z[:, j]).astype(np.float64)
    return cv

def _expected_cut(n, cut_vals, gammas_betas):
    """Compute expected cut for given QAOA parameters."""
    m = 1 << n
    state = np.full(m, 1.0 / np.sqrt(m), dtype=np.complex128).reshape((2,) * n)
    for gamma, beta in gammas_betas:
        phase = np.exp(-1j * gamma * cut_vals)
        state = state.reshape(m) * phase
        state = state.reshape((2,) * n)
        if beta != 0.0:
            c = np.cos(beta)
            s = np.sin(beta)
            G = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            for axis in range(n):
                st = np.moveaxis(state, axis, 0)
                shp = st.shape
                st = st.reshape(2, -1)
                st = G @ st
                state = np.moveaxis(st.reshape(shp), 0, axis)
    probs = np.abs(state.reshape(m)) ** 2
    return float(np.dot(probs, cut_vals))

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut by brute force for small n."""
    if n > 15:
        return 0.5 * len(edges)
    
    m = 1 << n
    states = np.arange(m, dtype=np.int64)
    best = 0
    for z in states:
        cut = 0
        for (i, j) in edges:
            bit_i = (z >> i) & 1
            bit_j = (z >> j) & 1
            if bit_i != bit_j:
                cut += 1
        if cut > best:
            best = cut
    
    return best

def _optimize_angles(n, edges, p, time_limit):
    """
    Optimize QAOA angles using:
    1. Coarse grid search for basin identification
    2. Differential evolution for global exploration
    3. Multi-start Nelder-Mead for local polishing
    4. Final high-precision polish
    """
    cut_vals = _cut_values(n, edges)
    
    def objective(x):
        gammas_betas = x.reshape(p, 2)
        return -_expected_cut(n, cut_vals, gammas_betas)
    
    # Bounds: gamma in [0, pi], beta in [0, pi]
    bounds = [(0.0, np.pi)] * (2 * p)
    
    # Initial guess
    best_x = np.tile([0.35, 0.25], (p, 1)).reshape(-1).astype(float)
    best_f = objective(best_x)
    
    t_start = time.time()
    
    def timed_out():
        return (time.time() - t_start) > (time_limit - 2.0)
    
    # Phase 1: Coarse grid search to identify promising basins
    if _time_left() > 10 and not timed_out():
        grid_g = np.linspace(0.0, np.pi, 6)
        grid_b = np.linspace(0.0, np.pi, 6)
        for g in grid_g:
            for b in grid_b:
                if timed_out():
                    break
                cand = np.tile([g, b], (p, 1)).reshape(-1)
                val = objective(cand)
                if val < best_f:
                    best_f, best_x = val, cand.copy()
            if timed_out():
                break
    
    # Phase 2: Differential Evolution for global exploration
    if _time_left() > 15 and not timed_out():
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=False,
                init='sobol',
                updating='deferred'
            )
            if de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start Nelder-Mead optimization
    rng = np.random.default_rng(713)
    nstarts = 30
    
    for k in range(nstarts):
        if timed_out():
            break
        
        # Generate random start points
        if k % 3 == 0:
            start = np.concatenate([
                rng.uniform(0.0, np.pi, size=p),
                rng.uniform(0.0, np.pi, size=p),
            ])
        elif k % 3 == 1:
            # Start near current best
            noise = rng.normal(0, 0.15, size=2 * p)
            start = best_x + noise
            start = np.clip(start, 0.0, np.pi)
        else:
            # Start from common QAOA initial values
            start = np.tile([0.5, 0.3], (p, 1)).reshape(-1)
            start = start + rng.normal(0, 0.1, size=2 * p)
            start = np.clip(start, 0.0, np.pi)
        
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish on the best point found
    if not timed_out():
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13})
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass
    
    # Phase 5: Second polish if time remains
    if not timed_out():
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass
    
    return best_x

def angles(n, edges, p):
    """
    Return QAOA angles for a given graph.
    
    Args:
        n: number of vertices
        edges: list of (i, j) edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check time
    if _time_left() < 5:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        x_opt = _optimize_angles(n, edges, p, _time_left() - 5)
        gammas = x_opt[0::2]
        betas = x_opt[1::2]
        result = np.column_stack([gammas, betas])
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
