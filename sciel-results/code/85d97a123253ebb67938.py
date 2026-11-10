# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _get_grid_candidates():
    """
    Generate a set of candidate integer lattice points within a bounded box.
    We use a 5x5 grid, giving 25 candidates. We need to choose 13.
    """
    candidates = []
    for i in range(5):
        for j in range(5):
            candidates.append([i, j])
    return np.array(candidates)

def _generate_structured_configs():
    """
    Generate diverse structured initial configurations based on lattice subsets.
    """
    configs = []
    grid = _get_grid_candidates()
    
    # Strategy 1: Regular 13-gon (baseline for comparison, but scaled to fit grid-like distribution)
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    configs.append(pts)
    
    # Strategy 2: Hexagonal/Lattice subsets
    # Try to pick 13 points from the 5x5 grid that are well-spread.
    # We can use a few heuristic subsets.
    
    # Subset A: Remove corners and center from 5x5? No, 25-4-1=20.
    # Let's just generate random subsets of the grid and score them.
    # But for structured init, let's try specific patterns.
    
    # Pattern 1: "Frame" - perimeter of 5x5 is 16 points. Pick 13.
    perimeter = []
    for i in range(5):
        perimeter.append([i, 0])
        perimeter.append([i, 4])
    for i in range(1, 4):
        perimeter.append([0, i])
        perimeter.append([4, i])
    perimeter = np.array(perimeter)
    # Pick every other point roughly?
    # Just take the first 13 for now, optimization will fix it.
    configs.append(perimeter[:13].astype(float) / 4.0) # Scale to [0,1]
    
    # Pattern 2: "Cross" - Middle row, middle col, plus some corners
    cross = []
    for i in range(5):
        cross.append([i, 2])
        cross.append([2, i])
    # Remove duplicates (center is double added)
    cross = np.unique(np.array(cross), axis=0)
    # Add corners
    for c in [[0,0], [4,0], [0,4], [4,4]]:
        if not any(np.allclose(c, p) for p in cross):
            cross.append(c)
    cross = np.array(cross)
    # Scale to [0,1]
    configs.append(cross.astype(float) / 4.0)
    
    # Pattern 3: Random grid subsets (5 diverse ones)
    rng = np.random.default_rng(123)
    for _ in range(5):
        indices = rng.choice(25, size=13, replace=False)
        pts = grid[indices].astype(float) / 4.0
        configs.append(pts)
        
    # Pattern 4: Concentric rings (non-grid based, but good structure)
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(6):
        a = np.pi / 3 * i
        pts.append([0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)])
    r2 = 0.4
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6
        pts.append([0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)])
    configs.append(np.array(pts))
    
    return configs

def _hill_climb(points: np.ndarray, start_time: float, deadline: float, max_steps: int = 20000):
    """
    Perform a local search using random perturbations.
    """
    best_points = points.copy()
    best_score = _compute_score(best_points)
    step_size = 0.05
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
    
    for step in range(max_steps):
        if time.time() - start_time > deadline - 5:
            break
            
        # Generate a neighbor
        candidate = best_points + rng.normal(0, step_size, best_points.shape)
        # Keep within bounds
        candidate = np.clip(candidate, 0.01, 0.99)
        
        cand_score = _compute_score(candidate)
        
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
            # Slightly increase step size if we improved? Or keep constant.
            # Constant is safer for this metric.
        else:
            # Slightly decrease step size if we didn't improve?
            # This is a simple hill climb, not SA.
            pass
            
    return best_points, best_score

def _refine_with_gradient(points: np.ndarray, start_time: float, deadline: float):
    """
    Use a simple gradient-free refinement (Nelder-Mead) on the best candidate.
    """
    from scipy.optimize import minimize
    
    def objective(params):
        pts = params.reshape(13, 2)
        return -_compute_score(pts)
        
    x0 = points.flatten()
    # Nelder-Mead can be slow, use it sparingly or with tight limits.
    # Given the non-smoothness, it might not help much, but let's try.
    # Actually, let's just do a few more hill climb steps with smaller step size.
    
    # Let's do a second pass of hill climb with smaller step size
    best_points = points.copy()
    best_score = _compute_score(best_points)
    step_size = 0.01
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
    
    for step in range(10000):
        if time.time() - start_time > deadline - 2:
            break
            
        candidate = best_points + rng.normal(0, step_size, best_points.shape)
        candidate = np.clip(candidate, 0.01, 0.99)
        
        cand_score = _compute_score(candidate)
        
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
            
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using a hybrid Lattice-Structured + Local Search approach.
    """
    start_time = time.time()
    deadline = 600 # seconds
    
    configs = _generate_structured_configs()
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Evaluate all structured configs
    for cfg in configs:
        score = _compute_score(cfg)
        if score > best_score:
            best_score = score
            best_points = cfg.copy()
            
    # Phase 2: Local Search (Hill Climbing) from each config
    # We have ~10 configs. Allocate time for each.
    time_per_config = (deadline - 10) / len(configs)
    
    for i, cfg in enumerate(configs):
        if time.time() - start_time > deadline - 10:
            break
            
        # Only refine if it's a promising candidate (score > some threshold) or top few
        # For simplicity, refine all, but with time limits.
        
        # Coarse search
        refined_pts, refined_score = _hill_climb(
            cfg, 
            start_time, 
            deadline, 
            max_steps=int(time_per_config * 100) # ~100 steps per second estimate
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
            
        # Fine search
        refined_pts2, refined_score2 = _refine_with_gradient(
            refined_pts, 
            start_time, 
            deadline
        )
        
        if refined_score2 > best_score:
            best_score = refined_score2
            best_points = refined_pts2.copy()
            
    # Final sanity check
    if best_points is None:
        best_points = configs[0]
        
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
