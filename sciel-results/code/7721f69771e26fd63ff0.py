"""Combined program: soft-max diameter surrogate + L-BFGS-B + diverse inits + high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 8

def _softmax_diameter(dists, beta):
    """Smooth approximation to max(dists). dists is a flat array of positive distances."""
    b = beta * dists
    b = b - np.max(b)  # numerical stability
    return (1.0 / beta) * np.log(np.sum(np.exp(b)))

def _neg_log_score(points_flat, beta):
    """Negative log of the scale-invariant score using soft-max diameter surrogate.
    Score = prod(d_i) / D^(n(n-1)) where D is the diameter (max distance).
    log_score = sum(log d_i) - n(n-1) * log(D).
    We minimize -log_score.
    """
    pts = points_flat.reshape(N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    if np.any(d < 1e-12):
        return 1e10
    log_prod = np.sum(np.log(d))
    D = _softmax_diameter(d, beta)
    if D < 1e-12:
        return 1e10
    log_score = log_prod - N * (N - 1) * np.log(D)
    return -log_score

def _true_score(points):
    """True scale-invariant score (product of ordered-pair distances / diameter^(n(n-1)))."""
    pts = np.array(points)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    diameter = np.max(d)
    if diameter < 1e-12:
        return -np.inf
    log_prod = np.sum(np.log(d))
    log_score = log_prod - N * (N - 1) * np.log(diameter)
    return np.exp(log_score)

def _scale_to_diameter2(pts):
    pts = np.array(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-12:
        pts = pts * (2.0 / diameter)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    # ---- Build diverse initializations ----
    inits = []

    # 1. Regular octagon
    angles = 2 * np.pi * np.arange(N) / N
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(octagon.copy())

    # 2. Antipodal pairs (4 pairs on the unit circle)
    antipodal = np.column_stack([np.cos(angles), np.sin(angles)])
    # Make them come in antipodal pairs: points 0&4, 1&5, 2&6, 3&7 are already antipodal in octagon
    # Create a different antipodal set with unequal radii
    rng = np.random.default_rng(7)
    base_angles = rng.uniform(0, np.pi, 4)  # 4 directions, each gives a pair
    radii = rng.uniform(0.5, 1.0, 4)
    antipodal_pts = np.zeros((8, 2))
    for i in range(4):
        a = base_angles[i]
        r = radii[i]
        antipodal_pts[2*i] = [r*np.cos(a), r*np.sin(a)]
        antipodal_pts[2*i+1] = [-r*np.cos(a), -r*np.sin(a)]
    inits.append(antipodal_pts)

    # 3. Rectangular grid (2x4)
    grid = np.array([
        [-1.0, -0.5], [1.0, -0.5], [-0.5, 0.5], [0.5, 0.5],
        [-1.0, 0.5], [1.0, 0.5], [-0.5, -0.5], [0.5, -0.5]
    ])
    inits.append(grid)

    # 4. Two squares (4 on outer radius, 4 on inner radius, rotated by pi/4)
    angles_sq1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    angles_sq2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    r1, r2 = 1.0, 0.6
    two_sq = np.column_stack([
        np.concatenate([r1*np.cos(angles_sq1), r2*np.cos(angles_sq2)]),
        np.concatenate([r1*np.sin(angles_sq1), r2*np.sin(angles_sq2)])
    ])
    inits.append(two_sq)

    # 5. Perturbed octagon
    inits.append(octagon + rng.normal(0, 0.2, size=(8, 2)))

    # 6. Random
    inits.append(rng.uniform(-1, 1, size=(8, 2)))

    best_pts = octagon.copy()
    best_score = _true_score(octagon)

    def _try_optimize(x0, beta, maxiter=2000):
        """Run L-BFGS-B from x0 with given beta. Returns (points, score) if improved."""
        nonlocal best_pts, best_score
        try:
            result = minimize(
                _neg_log_score,
                x0.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'gtol': 1e-12, 'ftol': 1e-12}
            )
            candidate = result.x.reshape(N, 2)
            cand_score = _true_score(candidate)
            if cand_score > best_score:
                best_pts = candidate.copy()
                best_score = cand_score
                return True
        except Exception:
            pass
        return False

    # ---- Phase 1: Coarse optimization from each initialization with moderate beta ----
    for init in inits:
        if time.time() > deadline - 20:
            break
        _try_optimize(init, beta=50.0, maxiter=1500)

    # ---- Phase 2: High-precision polish with increasing beta values ----
    # Higher beta makes the soft-max closer to the true max, refining the solution
    polish_betas = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    for beta in polish_betas:
        if time.time() > deadline - 10:
            break
        _try_optimize(best_pts, beta=beta, maxiter=3000)

    # ---- Phase 3: A few random restarts if time permits, with high beta ----
    rng2 = np.random.default_rng(123)
    for restart in range(30):
        if time.time() > deadline - 8:
            break
        # Perturb the best configuration
        noise = rng2.normal(0, 0.1, size=(8, 2))
        x0 = best_pts + noise
        _try_optimize(x0, beta=500.0, maxiter=2000)

    # ---- Final: scale to diameter 2 and return ----
    pts = _scale_to_diameter2(best_pts)
    return [[float(p[0]), float(p[1])] for p in pts]
# EVOLVE-BLOCK-END
