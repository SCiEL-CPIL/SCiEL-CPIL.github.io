"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _make_initial_grid(n, W, H):
    """Create a grid of n initial positions inside [0,W]x[0,H]."""
    # Use a grid layout
    side = int(np.ceil(np.sqrt(n * W / H))) if H > 0 else int(np.ceil(np.sqrt(n)))
    cols = side
    rows = int(np.ceil(n / cols))
    dx = W / cols
    dy = H / rows
    pts = []
    for i in range(cols):
        for j in range(rows):
            if len(pts) >= n:
                break
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            pts.append((x, y))
    # If we don't have enough, add random points
    while len(pts) < n:
        pts.append((np.random.uniform(0, W), np.random.uniform(0, H)))
    return np.array(pts[:n])


def _radius_from_positions(pts, W, H):
    """Given center positions, compute max radius such that all circles fit in [0,W]x[0,H]."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.concatenate([
        pts[:, 0],          # left
        W - pts[:, 0],      # right
        pts[:, 1],          # bottom
        H - pts[:, 1]       # top
    ])
    min_wall = np.min(wall_dists)
    
    if n == 1:
        return min_wall
    
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return 0.5 * min(min_wall, min_pair)


def _score(pts, W, H):
    """Compute the sum of radii (which is the objective to maximize)."""
    r = _radius_from_positions(pts, W, H)
    return r * len(pts)


def _neg_score(pts, W, H):
    """Negative of the score for minimization."""
    return -_score(pts, W, H)


def _is_valid(pts, W, H, tol=1e-7):
    """Check if the packing is valid: all circles disjoint and inside rectangle."""
    n = len(pts)
    if n == 0:
        return True
    
    # Check bounds
    if np.any(pts[:, 0] < -tol) or np.any(pts[:, 0] > W + tol) or \
       np.any(pts[:, 1] < -tol) or np.any(pts[:, 1] > H + tol):
        return False
    
    r = _radius_from_positions(pts, W, H)
    if r < -tol:
        return False
    
    # Check pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 2 * r - tol:
        return False
    
    return True


def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_result = None
    
    # Perimeter = 2*(W+H) = 4, so W+H = 2
    # We'll try multiple aspect ratios
    aspect_ratios = [
        1.0,      # square 1x1
        0.5,      # 0.667x1.333
        2.0,      # 2.667x0.667
        0.75,     # 1.143x0.857
        1.333,    # 1.455x0.545
        0.6,      # 1.25x0.75
        1.667,    # 1.6x0.8
        0.4,      # 1.429x0.571
    ]
    
    # Shuffle aspect ratios for diversity
    np.random.seed(42)
    np.random.shuffle(aspect_ratios)
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        W = 2.0 * ar / (1 + ar)
        H = 2.0 / (1 + ar)
        
        # Try multiple initial configurations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Different initializations
            if trial == 0:
                pts = _make_initial_grid(N, W, H)
            elif trial == 1:
                # Slightly perturbed grid
                pts = _make_initial_grid(N, W, H)
                pts += np.random.uniform(-0.02, 0.02, size=pts.shape)
            elif trial == 2:
                # Random initialization
                pts = np.random.uniform(0.1, W - 0.1, size=(N, 2))
            elif trial == 3:
                # Two-cluster initialization
                pts1 = np.random.uniform(0, W * 0.4, size=(N//2 + 1, 2))
                pts2 = np.random.uniform(W * 0.6, W, size=(N - N//2 - 1, 2))
                pts = np.vstack([pts1, pts2])
            else:
                # Three-cluster
                k1 = N // 3
                k2 = N // 3
                k3 = N - k1 - k2
                pts1 = np.random.uniform(0, W * 0.35, size=(k1, 2))
                pts2 = np.random.uniform(W * 0.35, W * 0.65, size=(k2, 2))
                pts3 = np.random.uniform(W * 0.65, W, size=(k3, 2))
                pts = np.vstack([pts1, pts2, pts3])
            
            # Optimize
            try:
                result = minimize(
                    _neg_score,
                    pts,
                    args=(W, H),
                    method='L-BFGS-B',
                    bounds=[(0, W)] * N + [(0, H)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if result.success or result.fun < 0:
                    score = -result.fun
                    if score > best_score:
                        best_score = score
                        best_result = result.x.copy()
                        best_W, best_H = W, H
            except Exception:
                pass
        
        # If we found a good result, do a refinement pass
        if best_result is not None and time.time() < deadline - 10:
            try:
                # Refine with tighter tolerances
                result2 = minimize(
                    _neg_score,
                    best_result,
                    args=(best_W, best_H),
                    method='L-BFGS-B',
                    bounds=[(0, best_W)] * N + [(0, best_H)] * N,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
                )
                if result2.fun < 0:
                    score2 = -result2.fun
                    if score2 > best_score:
                        best_score = score2
                        best_result = result2.x.copy()
            except Exception:
                pass
    
    # If no valid result found, fall back to simple grid
    if best_result is None:
        W = 1.0
        H = 1.0
        best_result = _make_initial_grid(N, W, H)
        best_W, best_H = W, H
    
    # Compute final radii and construct output
    r = _radius_from_positions(best_result, best_W, best_H)
    r = max(r, 0)
    
    # Clamp positions to be strictly inside
    best_result[:, 0] = np.clip(best_result[:, 0], r, best_W - r)
    best_result[:, 1] = np.clip(best_result[:, 1], r, best_H - r)
    
    # Recompute radius after clamping
    r = _radius_from_positions(best_result, best_W, best_H)
    r = max(r, 1e-10)
    
    # Final validity check
    if not _is_valid(best_result, best_W, best_H, tol=1e-5):
        # Try to fix by reducing radius slightly
        for shrink in [0.99, 0.98, 0.95, 0.9]:
            r2 = r * shrink
            # Check if valid with smaller radius
            diff = best_result[:, None, :] - best_result[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            wall_dists = np.concatenate([
                best_result[:, 0],
                best_W - best_result[:, 0],
                best_result[:, 1],
                best_H - best_result[:, 1]
            ])
            if np.min(dists) >= 2 * r2 and np.min(wall_dists) >= r2:
                r = r2
                break
    
    # Construct output: (x, y, radius)
    output = np.column_stack([best_result, np.full(N, r)])
    
    return output

# EVOLVE-BLOCK-END
