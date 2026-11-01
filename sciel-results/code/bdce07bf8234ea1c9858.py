"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint


def _score(centers):
    """Compute the radius for a given set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Min distance to boundary
    r_bound = 0.5 - np.max(np.abs(centers), axis=1)
    min_r_bound = np.min(r_bound)
    if min_r_bound <= 0:
        return 0.0
    # Min pairwise distance
    min_pair = np.inf
    for i in range(N):
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if dists.size > 0:
            local_min = np.min(dists)
            if local_min < min_pair:
                min_pair = local_min
    if not np.isfinite(min_pair):
        min_pair = 0.0
    return min(min_pair / 2.0, min_r_bound)


def _neg_score(centers):
    return -_score(centers)


def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    best_centers = None
    best_score = 0.0
    
    # Initial configurations to try
    initial_configs = []
    
    # 1. Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_configs.append(grid[:N].copy())
    
    # 2. Hexagonal packing approximation
    # Place circles in a hexagonal lattice pattern
    hex_centers = []
    # Try different hexagonal arrangements
    for dx, dy, offset_x, offset_y in [(0.15, 0.13, 0, 0), (0.14, 0.12, 0.01, 0.01), (0.16, 0.14, -0.01, -0.01)]:
        pts = []
        for row in range(8):
            for col in range(8):
                x = -0.5 + dx/2 + col * dx + (row % 2) * dx/2
                y = -0.5 + dy/2 + row * dy
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
        if len(pts) >= N:
            # Shuffle and take first N for variety
            np.random.seed(row)
            idx = np.random.permutation(len(pts))[:N]
            initial_configs.append(np.array([pts[i] for i in idx]))
    
    # 3. Random configurations
    np.random.seed(42)
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        random_pts = np.random.uniform(-0.45, 0.45, size=(N, 2))
        initial_configs.append(random_pts)
    
    # 4. Concentric ring pattern
    rings = []
    rings.append([0.0, 0.0])
    for r_idx, (n_pts, radius) in enumerate([(6, 0.15), (12, 0.25), (18, 0.35), (14, 0.42)]):
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        for a in angles:
            rings.append([radius * np.cos(a), radius * np.sin(a)])
    if len(rings) >= N:
        initial_configs.append(np.array(rings[:N]))
    
    # Optimize from each initial configuration
    for init_idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Clip to valid region
        x0 = np.clip(x0, -0.49, 0.49)
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                _neg_score,
                x0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            centers = result.x.reshape(N, 2)
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead for local refinement if L-BFGS-B didn't do well
        if time.time() > deadline - 20:
            break
        try:
            x0_nm = np.clip(x0, -0.49, 0.49).flatten()
            result_nm = minimize(
                _neg_score,
                x0_nm,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            centers_nm = result_nm.x.reshape(N, 2)
            s_nm = _score(centers_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_centers = centers_nm.copy()
        except Exception:
            pass
    
    # Final refinement with more iterations if time permits
    if best_centers is not None and time.time() < deadline - 10:
        try:
            result_final = minimize(
                _neg_score,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            centers_final = result_final.x.reshape(N, 2)
            s_final = _score(centers_final)
            if s_final > best_score:
                best_score = s_final
                best_centers = centers_final.copy()
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback: return the grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
