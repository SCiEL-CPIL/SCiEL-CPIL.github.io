"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Active-Set Local Search: Instead of random perturbations (SA), we identify the specific 
   triangles that define the current minimum area (the "active set").
2. We coordinate moves of the vertices involved in these critical triangles.
3. We use a vectorized evaluation to quickly test small perturbations.
4. We employ an adaptive step-size that expands on success and shrinks on failure, 
   allowing us to "climb" the rugged landscape of the min-area function.
5. Multi-start with diverse initializations (Grid, Random, Greedy).
"""
import numpy as np
import time
import random
import math

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        x = np.clip(x, 0.0, 1.0)
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        y = np.clip(y, 0.0, max_y)
        pts[i] = [x, y]
    return pts

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _greedy_init(rng):
    """Greedy initialization: place points one by one to maximize min distance."""
    pts = []
    # Start with a centroid-like point
    pts.append(np.array([0.5, S3H/2]))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        # Sample candidates
        for _ in range(2000):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            cand = np.array([x, y])
            
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _grid_init():
    """Structured grid-like initialization."""
    # 3 vertices
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H]
    ]
    # 3 edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.25, S3H/2])
    pts.append([0.75, S3H/2])
    # 3 medial triangle vertices (centroids of corner sub-triangles)
    pts.append([0.25, S3H/6])
    pts.append([0.75, S3H/6])
    pts.append([0.5, S3H/3])
    # 2 interior points
    pts.append([0.5, S3H/4])
    pts.append([0.5, S3H/2])
    
    return np.array(pts[:N])

def _random_init(rng):
    """Generate random points inside the triangle."""
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        x = u + 0.5 * v
        y = S3H * v
        pts.append([x, y])
    return np.array(pts)

def _active_set_search(pts, rng, time_budget):
    """
    Active-Set Local Search.
    1. Find min area.
    2. Identify all triangles with area close to min (active set).
    3. Try to move vertices of these triangles in random directions.
    4. Accept if min area increases.
    """
    pts = pts.copy()
    current_score = _min_triangle_area(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    start_time = time.time()
    step_size = 0.1
    min_step = 1e-6
    
    # Pre-compute barycentric-like weights? No, just cartesian.
    
    while time.time() - start_time < time_budget:
        # Find active set
        areas = 0.5 * np.abs(
            (pts[_j_idx] - pts[_i_idx])[:, 0] * (pts[_k_idx] - pts[_i_idx])[:, 1] -
            (pts[_j_idx] - pts[_i_idx])[:, 1] * (pts[_k_idx] - pts[_i_idx])[:, 0]
        )
        
        # Tolerance for "active" set: within 1% of min
        tol = max(1e-6, 0.01 * current_score)
        active_mask = areas <= (current_score + tol)
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            # Fallback: just the minimum
            min_idx = np.argmin(areas)
            active_indices = np.array([min_idx])
        
        # Pick a random subset of active triangles to "work on"
        # This prevents trying to satisfy all constraints at once if they conflict
        num_to_move = min(len(active_indices), rng.randint(1, 3))
        chosen_triangles = rng.sample(active_indices.tolist(), num_to_move)
        
        # Collect unique points involved in these triangles
        involved_points = set()
        for tri_idx in chosen_triangles:
            i, j, k = _triplets[tri_idx]
            involved_points.add(i)
            involved_points.add(j)
            involved_points.add(k)
        
        # Generate random perturbations for these points
        new_pts = pts.copy()
        for p_idx in involved_points:
            angle = rng.uniform(0, 2 * math.pi)
            # Scale magnitude: maybe smaller for more constrained points?
            mag = rng.uniform(0, step_size)
            dx = mag * math.cos(angle)
            dy = mag * math.sin(angle)
            new_pts[p_idx][0] += dx
            new_pts[p_idx][1] += dy
            
        # Project to triangle
        new_pts = _project_to_triangle(new_pts)
        
        new_score = _min_triangle_area(new_pts)
        
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            # Expand step size on success
            step_size = min(step_size * 1.1, 0.5)
        else:
            # Shrink step size on failure
            step_size *= 0.8
            if step_size < min_step:
                # Reset step size and try a bigger random jump to escape local min?
                # Or just restart this local search?
                # Let's try a random restart of the step size
                step_size = 0.05
    
    return best_pts, best_score

def _refine_with_nm(pts, time_budget):
    """Final polish with Nelder-Mead."""
    start_time = time.time()
    
    def nm_obj(pts_flat):
        pts_tmp = pts_flat.reshape(N, 2)
        pts_tmp = _project_to_triangle(pts_tmp)
        return -_min_triangle_area(pts_tmp)
    
    try:
        from scipy.optimize import minimize
        result = minimize(
            nm_obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if result.success:
            pts_opt = result.x.reshape(N, 2)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                score = _min_triangle_area(pts_opt)
                return pts_opt, score
    except Exception:
        pass
    
    return pts, _min_triangle_area(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
        
    # 2. Grid
    try:
        p_grid = _grid_init()
        p_grid = _project_to_triangle(p_grid)
        initializations.append(p_grid)
    except Exception:
        pass
        
    # 3. Random
    for _ in range(5):
        try:
            p_rand = _random_init(rng)
            initializations.append(p_rand)
        except Exception:
            pass
            
    # Phase 1: Active Set Search
    # We have ~135s for search.
    # Let's run the best few initializations.
    
    # Sort initializations by initial score?
    scored_inits = []
    for init_pts in initializations:
        s = _min_triangle_area(init_pts)
        scored_inits.append((s, init_pts))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    # Run search on top 3
    # Allocate time: ~40s each
    for i, (init_score, init_pts) in enumerate(scored_inits[:3]):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        # If the initial score is already very good, maybe less time needed?
        # But let's keep it simple.
        search_time = 40.0
        
        sa_pts, sa_score = _active_set_search(init_pts, rng, search_time)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            
        # Check if we are improving
        # If not, maybe try a different init?
        
    # Phase 2: Refine the best solution
    if best_pts is not None:
        if time.time() - start_time < DEADLINE - 5:
            refined_pts, refined_score = _refine_with_nm(best_pts, 10.0)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
                
    # Fallback
    if best_pts is None:
        for s, p in scored_inits:
            if s > best_score:
                best_score = s
                best_pts = p.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(_random_init(rng))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
