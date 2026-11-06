import numpy as np
from scipy.optimize import minimize
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    n = 8
    num_pairs = n * (n - 1) // 2  # 28

    # Free indices (0 and 4 are fixed at (0,0) and (2,0))
    fixed_idx = [0, 4]
    free_idx = [i for i in range(n) if i not in fixed_idx]
    n_free = len(free_idx)  # 6

    def unpack(x):
        p = np.zeros((n, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2 * i], x[2 * i + 1]]
        return p

    def objective(x, beta=100.0):
        """
        Minimize: -sum(log d_ij) + num_pairs * log(soft_max(d_ij, beta))
        For points already at diameter 2 (fixed pair), soft_max ≈ 2,
        so this is essentially -sum(log d_ij) + constant.
        The soft-max term provides a smooth penalty if diameter < 2.
        """
        p = unpack(x)
        # Compute all pairwise distances
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        # Upper triangle only
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)

        sum_log_d = np.sum(np.log(d))

        # Soft-max via log-sum-exp: log(sum(exp(beta * d))) / beta
        # This approximates max(d) smoothly
        max_d = np.max(d)
        # log-sum-exp: max_d + log(sum(exp(beta*(d - max_d)))) / beta
        scaled = beta * (d - max_d)
        # Clip to avoid overflow
        scaled = np.clip(scaled, -500, 500)
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_max = max_d + log_sum / beta

        # Objective: minimize -sum_log_d + num_pairs * log(soft_max)
        # = -sum_log_d + num_pairs * log(soft_max)
        val = -sum_log_d + num_pairs * np.log(soft_max)
        return val

    def objective_simple(x):
        """Simple version without soft-max, just for reference."""
        p = unpack(x)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)
        sum_log_d = np.sum(np.log(d))
        return -sum_log_d

    # Initial guess: Regular octagon with P0=(0,0), P4=(2,0)
    # Regular octagon on unit circle centered at origin:
    # Angles: 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # P0 = (1,0), P4 = (-1,0)
    # Transform: new_pts = -pts + [1, 0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_unit = np.column_stack((np.cos(angles), np.sin(angles)))
    init_pts = -pts_unit + np.array([1.0, 0.0])
    # Verify: init_pts[0] = -(1,0)+(1,0) = (0,0), init_pts[4] = -(-1,0)+(1,0) = (2,0)

    x0 = np.array([init_pts[i] for i in free_idx]).flatten()

    best_x = x0.copy()
    best_obj = objective(x0)

    # Phase 1: Local optimization from regular octagon with BFGS (smooth objective)
    try:
        res = minimize(objective, x0, method='BFGS',
                       options={'maxiter': 500, 'gtol': 1e-8, 'disp': False})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Phase 2: Multi-restart with structured perturbations
    np.random.seed(42)
    random.seed(42)

    # Structured perturbations: move pairs symmetrically
    # The regular octagon has D4 symmetry. We can perturb while preserving some structure.
    restarts = []

    # Small random perturbations
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for trial in range(3):
            noise = np.random.normal(0, scale, size=(n, 2))
            noise[0] = 0
            noise[4] = 0
            perturbed = init_pts + noise
            x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
            restarts.append(x0_pert)

    # Larger perturbations
    for scale in [0.8, 1.0, 1.5]:
        for trial in range(2):
            noise = np.random.normal(0, scale, size=(n, 2))
            noise[0] = 0
            noise[4] = 0
            perturbed = init_pts + noise
            x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
            restarts.append(x0_pert)

    # Random starts
    for trial in range(10):
        x0_rand = np.random.uniform(-1, 3, size=n_free * 2)
        restarts.append(x0_rand)

    for x0_r in restarts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, x0_r, method='BFGS',
                           options={'maxiter': 200, 'gtol': 1e-7, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # Phase 3: High-precision polish pass
    if time.time() < deadline - 5:
        try:
            # Use L-BFGS-B with tighter tolerances
            res = minimize(objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Additional polish with even tighter tolerance if time permits
    if time.time() < deadline - 3:
        try:
            res = minimize(objective, best_x, method='BFGS',
                           options={'maxiter': 2000, 'gtol': 1e-14, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    p = unpack(best_x)

    # Ensure distinct points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-8:
                p[j] += np.array([1e-6, 0])

    # Scale to ensure diameter is exactly 2
    max_d = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(p[i] - p[j])
            if d > max_d:
                max_d = d

    if max_d > 0 and abs(max_d - 2.0) > 1e-10:
        scale_factor = 2.0 / max_d
        p = p * scale_factor

    # Return as list of lists of floats
    return [[float(p[i][0]), float(p[i][1])] for i in range(n)]
