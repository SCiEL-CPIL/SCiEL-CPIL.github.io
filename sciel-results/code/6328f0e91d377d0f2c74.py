"""
QAOA p=2 graph-dependent angle optimizer.
Strategy: dense graph-scaled grid search over the 4D parameter space,
followed by multi-start Nelder-Mead refinement from the top grid points,
then a final high-precision polish.  Uses vectorized state evolution and
analytic cost-phase application for speed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 142.0  # safety margin before 150 s kill

_value_cache = {}


def _remaining():
    return _DEADLINE - time.time()


def _build_state_data(n, edges):
    dim = 1 << n
    states = np.arange(dim)
    bits = np.zeros((dim, n), dtype=np.int8)
    for j in range(n):
        bits[:, j] = (states >> j) & 1

    cut_counts = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        cut_counts += (bits[:, u] != bits[:, v]).astype(np.float64)

    # Pre-compute X-operator index pairs for each qubit
    x_pairs = []
    for i in range(n):
        mask = 1 << i
        idx_0 = np.where((states & mask) == 0)[0]
        idx_1 = idx_0 + mask
        x_pairs.append((idx_0, idx_1))

    return states, cut_counts, x_pairs


def _expected_cut(gammas, betas, cut_counts, x_pairs):
    """Fast vectorized QAOA forward pass for p=2 (or general p)."""
    dim = len(cut_counts)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    for layer in range(len(gammas)):
        g = gammas[layer]
        b = betas[layer]

        # Cost: phase kick
        if abs(g) > 1e-12:
            psi *= np.exp(-1j * g * cut_counts)

        # Mixer: X rotations
        if abs(b) > 1e-12:
            c = np.cos(b)
            s = np.sin(b)
            for idx_0, idx_1 in x_pairs:
                p0 = psi[idx_0]
                p1 = psi[idx_1]
                psi[idx_0] = c * p0 - 1j * s * p1
                psi[idx_1] = -1j * s * p0 + c * p1

    probs = np.real(psi * np.conj(psi))
    return np.dot(probs, cut_counts)


def angles(n, edges, p):
    edge_set = frozenset(frozenset((u, v)) for u, v in edges)
    cache_key = (n, edge_set, p)

    if cache_key in _value_cache:
        return _value_cache[cache_key]

    states, cut_counts, x_pairs = _build_state_data(n, edges)
    m = len(edges)

    def obj(params):
        return -_expected_cut(
            [params[0], params[2]],
            [params[1], params[3]],
            cut_counts, x_pairs
        )

    # --- Graph-scaled parameter ranges ---
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    # Denser graphs: optimal angles smaller
    sf = max(0.4, 1.0 - 0.6 * density)

    g_lo, g_hi = 0.02, 0.8 * sf
    b_lo, b_hi = 0.01, 0.5 * sf

    # --- Stage 1: Dense grid search ---
    n_grid = 14  # 14^4 = 38416 evaluations
    g_vals = np.linspace(g_lo, g_hi, n_grid)
    b_vals = np.linspace(b_lo, b_hi, n_grid)

    best_val = -np.inf
    best_params = np.array([0.3, 0.1, 0.3, 0.1])

    # Store top-K grid points for multi-start refinement
    top_k = 8
    top_points = []

    count = 0
    for i in range(n_grid):
        if _remaining() < 25.0:
            break
        for j in range(n_grid):
            if _remaining() < 25.0:
                break
            for k in range(n_grid):
                if _remaining() < 25.0:
                    break
                for l in range(n_grid):
                    count += 1
                    params = np.array([g_vals[i], b_vals[j], g_vals[k], b_vals[l]])
                    val = _expected_cut(
                        [params[0], params[2]],
                        [params[1], params[3]],
                        cut_counts, x_pairs
                    )
                    if val > best_val:
                        best_val = val
                        best_params = params

                    # Maintain top-K
                    if len(top_points) < top_k:
                        top_points.append((val, params.copy()))
                        top_points.sort(key=lambda x: -x[0])
                    elif val > top_points[-1][0]:
                        top_points[-1] = (val, params.copy())
                        top_points.sort(key=lambda x: -x[0])

    if best_val == -np.inf:
        # Fallback
        best_params = np.array([0.3, 0.1, 0.3, 0.1])
        best_val = _expected_cut([0.3, 0.3], [0.1, 0.1], cut_counts, x_pairs)

    # --- Stage 2: Multi-start Nelder-Mead from top grid points ---
    if _remaining() > 15.0:
        for val, params in top_points[:min(top_k, 4)]:
            if _remaining() < 10.0:
                break
            try:
                res = minimize(
                    obj, params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
                res_val = -res.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = res.x
            except Exception:
                pass

    # --- Stage 3: Final high-precision polish ---
    if _remaining() > 5.0:
        try:
            res2 = minimize(
                obj, best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 800,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            res2_val = -res2.fun
            if res2_val > best_val:
                best_val = res2_val
                best_params = res2.x
        except Exception:
            pass

    # --- Stage 4: One more ultra-tight polish if time allows ---
    if _remaining() > 3.0:
        try:
            res3 = minimize(
                obj, best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True,
                }
            )
            res3_val = -res3.fun
            if res3_val > best_val:
                best_val = res3_val
                best_params = res3.x
        except Exception:
            pass

    # Format output
    out = np.zeros((p, 2))
    out[0, 0] = best_params[0]
    out[0, 1] = best_params[1]
    if p > 1:
        out[1, 0] = best_params[2]
        out[1, 1] = best_params[3]

    _value_cache[cache_key] = out
    return out
# EVOLVE-BLOCK-END
