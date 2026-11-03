"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

DEADLINE = 138.0  # safety margin from the 150s kill

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])


def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return (bary[:, 0:1] * _VERTS[0]
            + bary[:, 1:2] * _VERTS[1]
            + bary[:, 2:3] * _VERTS[2])


def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _norm_bary(bary):
    """Clip to non-negative and renormalize each row to sum 1."""
    bary = np.maximum(0.0, bary)
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return bary / s


def _vectorized_min_area_cart(points):
    """Minimum area over all 165 triangles, vectorized."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))


def _soft_min_area_cart(points, alpha):
    """Soft minimum (log-sum-exp) of triangle areas for smoothness."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    return float(min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a)))))


def _bary_objective(flat_bary, alpha):
    """Objective in barycentric space: -soft_min + boundary penalty."""
    bary = _norm_bary(flat_bary.reshape(N, 3))
    cart = _bary_to_cart(bary)
    soft = _soft_min_area_cart(cart, alpha)
    x = cart[:, 0]
    y = cart[:, 1]
    # After normalization the point is always inside, so these are ~0; keep for safety.
    pen = (np.sum(np.maximum(0.0, -y) ** 2)
           + np.sum(np.maximum(0.0, y - S3 * x) ** 2)
           + np.sum(np.maximum(0.0, S3 * x + y - S3) ** 2))
    return -soft + 100.0 * pen


def _random_bary(rng):
    r = rng.random((N, 3))
    r = r / r.sum(axis=1, keepdims=True)
    return r


def _lattice_bary(K):
    """Dense lattice in barycentric space; return best K*K*K candidate subset of N points."""
    # Generate all lattice barycentric points
    coords = []
    for a in range(K + 1):
        for b in range(K + 1 - a):
            c = K - a - b
            coords.append((a / K, b / K, c / K))
    coords = np.array(coords)
    cart_all = _bary_to_cart(coords)
    # Pick N points that maximize the minimum pairwise area contribution:
    # greedy farthest-point sampling over the lattice.
    chosen = [0]
    min_d = np.full(len(coords), np.inf)
    for _ in range(N - 1):
        min_d[0] = -1.0  # exclude already chosen (set min_d to 0 for chosen)
        # recompute distance to nearest chosen
        for c in chosen:
            d = np.sum((cart_all - cart_all[c]) ** 2, axis=1)
            min_d = np.minimum(min_d, d)
        for c in chosen:
            min_d[c] = -1.0
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
    return coords[chosen].copy()


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    def elapsed():
        return time.time() - start_time

    # Best tracked in Cartesian for clarity.
    best_cart = _bary_to_cart(_lattice_bary(12))
    best_score = _vectorized_min_area_cart(best_cart)

    # --- Phase 1: soft-min annealing with Nelder-Mead, multiple restarts ---
    alphas = [8, 20, 50, 120, 300, 800, 2000]
    for alpha in alphas:
        if elapsed() > DEADLINE - 12:
            break
        # number of restarts scales with remaining time
        remaining = DEADLINE - 12 - elapsed()
        n_restarts = max(2, int(remaining // 4))
        for r in range(n_restarts):
            if elapsed() > DEADLINE - 12:
                break
            if r == 0:
                x0 = _cart_to_bary(best_cart).flatten()
            elif r == 1:
                x0 = _lattice_bary(12).flatten()
            else:
                x0 = _random_bary(rng).flatten()
            try:
                res = minimize(_bary_objective, x0, method='Nelder-Mead',
                               args=(alpha,),
                               options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-7})
            except Exception:
                continue
            new_bary = _norm_bary(res.x.reshape(N, 3))
            new_cart = _bary_to_cart(new_bary)
            new_score = _vectorized_min_area_cart(new_cart)
            if new_score > best_score:
                best_score = new_score
                best_cart = new_cart

    # --- Phase 2: local polish on the hard (vectorized) min in Cartesian ---
    if elapsed() < DEADLINE - 6:
        def cart_obj(flat):
            cart = flat.reshape(N, 2)
            # Project into triangle
            x = cart[:, 0].copy()
            y = cart[:, 1].copy()
            y = np.clip(y, 0.0, S3H)
            x = np.clip(x, y / S3, 1.0 - y / S3)
            cart = np.column_stack([x, y])
            return -_vectorized_min_area_cart(cart)

        best_bary = _cart_to_bary(best_cart)
        for attempt in range(3):
            if elapsed() > DEADLINE - 6:
                break
            if attempt == 0:
                x0 = best_cart.flatten()
            else:
                pert = best_cart + rng.normal(0, 0.02, size=(N, 2))
                # project
                x = np.clip(pert[:, 0], 0, 1)
                y = np.clip(pert[:, 1], 0, S3H)
                x = np.clip(x, y / S3, 1.0 - y / S3)
                x0 = np.column_stack([x, y]).flatten()
            try:
                res = minimize(cart_obj, x0, method='Nelder-Mead',
                               options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-10})
            except Exception:
                continue
            final_cart = res.x.reshape(N, 2)
            x = np.clip(final_cart[:, 0], 0, 1)
            y = np.clip(final_cart[:, 1], 0, S3H)
            x = np.clip(x, y / S3, 1.0 - y / S3)
            final_cart = np.column_stack([x, y])
            fs = _vectorized_min_area_cart(final_cart)
            if fs > best_score:
                best_score = fs
                best_cart = final_cart

    # Final projection to guarantee validity
    x = np.clip(best_cart[:, 0], 0, 1)
    y = np.clip(best_cart[:, 1], 0, S3H)
    x = np.clip(x, y / S3, 1.0 - y / S3)
    best_cart = np.column_stack([x, y])

    return best_cart
# EVOLVE-BLOCK-END
