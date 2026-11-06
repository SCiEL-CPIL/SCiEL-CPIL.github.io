import numpy as np
from scipy.optimize import minimize
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    def objective(x):
        """Minimize negative sum of log distances."""
        p = unpack(x)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)
        return -np.sum(np.log(d))

    # Initial guess: Regular octagon with P0=(0,0), P4=(2,0)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_unit = np.column_stack((np.cos(angles), np.sin(angles)))
    init_pts = -pts_unit + np.array([1.0, 0.0])
    x0 = np.array([init_pts[i] for i in free_idx]).flatten()

    best_x = x0.copy()
    best_obj = objective(x0)

    # Phase 1: Local optimization from regular octagon
    try:
        res = minimize(objective, x0, method='BFGS',
                       options={'maxiter': 500, 'gtol': 1e-10, 'disp': False})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Phase 2: Multi-restart with diverse initial points
    np.random.seed(42)
    random.seed(42)

    restarts = []

    # Small random perturbations of regular octagon
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        for trial in range(2):
            noise = np.random.normal(0, scale, size=(n, 2))
            noise[0] = 0
            noise[4] = 0
            perturbed = init_pts + noise
            x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
            restarts.append(x0_pert)

    # Structured: two concentric circles
    # Inner circle radius r1, outer radius r2, with 4 points each
    for r1 in [0.3, 0.5, 0.7]:
        for r2 in [1.0, 1.2, 1.4]:
            # 4 points on inner circle at angles 0, pi/2, pi, 3pi/2
            # 4 points on outer circle at angles pi/4, 3pi/4, 5pi/4, 7pi/4
            inner_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
            outer_angles = inner_angles + np.pi/4
            inner_pts = np.column_stack((r1*np.cos(inner_angles), r1*np.sin(inner_angles)))
            outer_pts = np.column_stack((r2*np.cos(outer_angles), r2*np.sin(outer_angles)))
            # Center at (1,0) so that diameter is approximately 2
            all_pts = np.vstack([inner_pts, outer_pts]) + np.array([1.0, 0.0])
            # Fix P0 and P4: swap if needed
            # Just use as is, the optimizer will adjust
            x0_struct = np.array([all_pts[i] for i in free_idx]).flatten()
            restarts.append(x0_struct)

    # Random starts
    for trial in range(15):
        x0_rand = np.random.uniform(-1, 3, size=n_free * 2)
        restarts.append(x0_rand)

    for x0_r in restarts:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(objective, x0_r, method='BFGS',
                           options={'maxiter': 300, 'gtol': 1e-8, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'gtol': 1e-14, 'ftol': 1e-16, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 4: Additional polish
    if time.time() < deadline - 3:
        try:
            res = minimize(objective, best_x, method='BFGS',
                           options={'maxiter': 2000, 'gtol': 1e-15, 'disp': False})
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

    return [[float(p[i][0]), float(p[i][1])] for i in range(n)]
