"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # distance to each of the 4 walls
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    # nearest neighbour distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _neg_score(pts):
    return -_score(pts)

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points.
    """
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 7, 6, 7, 6, 7, 6, 3],
        [5, 6, 7, 6, 7, 6, 7, 6]
    ]
    
    best_pts = None
    best_score = -1
    
    for rows in configs:
        if sum(rows) != N:
            continue
            
        pts = []
        max_row = max(rows)
        spacing = 1.0 / max_row
        h = spacing * np.sqrt(3.0) / 2.0
        
        total_height = (len(rows) - 1) * h
        start_y = -total_height / 2.0
        
        for i, n in enumerate(rows):
            y = start_y + i * h
            offset = (0.5 * spacing) if (i % 2 == 1) else 0.0
            xs = np.arange(n) * spacing
            center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
            xs = xs - center + offset
            for x in xs:
                pts.append([x, y])
        
        pts = np.array(pts)
        # Scale to fit within [-0.45, 0.45] to allow expansion
        max_extent = max(np.max(np.abs(pts)), 1e-6)
        scale = 0.45 / max_extent
        pts = pts * scale
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    return best_pts

def _make_grid_initial():
    """
    Create a grid-based initial configuration.
    """
    # Try different grid arrangements
    best_pts = None
    best_score = -1
    
    for rows in [7, 8, 9, 10]:
        for cols in [7, 8, 9, 10]:
            if rows * cols != N:
                continue
            
            xs = np.linspace(-0.4, 0.4, cols)
            ys = np.linspace(-0.4, 0.4, rows)
            xx, yy = np.meshgrid(xs, ys)
            pts = np.column_stack([xx.ravel(), yy.ravel()])
            
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    return best_pts

def _make_random_initial(seed=42):
    """
    Create random initial configurations and pick the best.
    """
    np.random.seed(seed)
    best_pts = None
    best_score = -1
    
    for trial in range(20):
        pts = np.random.uniform(-0.45, 0.45, (N, 2))
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    return best_pts

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    # Generate multiple initial configurations
    hex_pts = _make_hex_initial()
    grid_pts = _make_grid_initial()
    rand_pts = _make_random_initial()
    
    # Pick the best initial configuration
    candidates = []
    if hex_pts is not None:
        candidates.append(hex_pts)
    if grid_pts is not None:
        candidates.append(grid_pts)
    if rand_pts is not None:
        candidates.append(rand_pts)
    
    if not candidates:
        np.random.seed(42)
        candidates.append(np.random.uniform(-0.4, 0.4, (N, 2)))
    
    best_pts = max(candidates, key=lambda p: _score(p))
    best_score = _score(best_pts)
    
    # Phase 1: Powell optimization on the best initial
    try:
        res = minimize(_neg_score, best_pts.flatten(), method="Powell",
                       options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10, 'disp': False})
        if time.time() > deadline:
            return best_pts.reshape(N, 2)
            
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    except Exception:
        pass
    
    # Phase 2: Try Powell on other initial configurations if time permits
    for cand in candidates:
        if time.time() > deadline - 20.0:
            break
        if np.allclose(cand, best_pts, atol=1e-6):
            continue
        try:
            res = minimize(_neg_score, cand.flatten(), method="Powell",
                           options={'maxiter': 1000, 'ftol': 1e-10, 'xtol': 1e-10, 'disp': False})
            if time.time() > deadline:
                return best_pts.reshape(N, 2)
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polishing
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                            options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 5000, 'adaptive': True})
            if time.time() > deadline:
                return best_pts.reshape(N, 2)
            new_pts = res2.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase 4: Local search - try small perturbations
    if time.time() < deadline - 5.0:
        np.random.seed(123)
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            # Random small perturbation
            perturbation = np.random.normal(0, 0.005, (N, 2))
            perturbed = best_pts + perturbation
            # Clamp to square
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
            
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_pts = perturbed
    
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
