"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Direct derivative-free optimization (Nelder-Mead) on the true non-smooth min-area objective.
2. 'Error-Repair' Nudge Search: Identify points involved in the minimum-area triangle(s)
   and apply targeted random perturbations to them to break degeneracies.
3. Multi-start with diverse initializations (greedy, structured, random).
4. Time-budgeted iterations to fully utilize the 120s limit.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # Safety margin before 150s limit

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

def _get_min_area_indices(pts):
    """Return the indices of the triangles that achieve the minimum area."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    # Tolerance for "minimum" triangles
    mask = areas < (min_area + 1e-9)
    return np.where(mask)[0]

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

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(1000):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """Create a structured initialization based on known Heilbronn-like configurations."""
    # Use a hexagonal-like pattern scaled to the triangle
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def _nudge_search(pts, rng, iterations=50, step_size=0.01):
    """
    Error-Repair Nudge Search:
    1. Find the minimum area triangle(s).
    2. Identify the points involved in these triangles.
    3. Randomly nudge these points slightly.
    4. If the new configuration improves the minimum area, keep it.
    """
    best_pts = pts.copy()
    best_score = _min_triangle_area(best_pts)
    
    for _ in range(iterations):
        # Find indices of min-area triangles
        min_idx = _get_min_area_indices(best_pts)
        
        # Collect unique point indices involved in these triangles
        involved_points = set()
        for idx in min_idx:
            involved_points.add(_i_idx[idx])
            involved_points.add(_j_idx[idx])
            involved_points.add(_k_idx[idx])
        
        # Try nudging one of the involved points
        if not involved_points:
            continue
            
        point_idx = rng.choice(list(involved_points))
        
        # Generate a random nudge
        nudge = np.array([rng.gauss(0, step_size), rng.gauss(0, step_size)])
        new_pts = best_pts.copy()
        new_pts[point_idx] = new_pts[point_idx] + nudge
        
        # Project back to triangle
        new_pts = _project_to_triangle(new_pts)
        
        # Check validity and score
        if _check_inside(new_pts):
            new_score = _min_triangle_area(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
    
    return best_pts, best_score

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Nelder-Mead on the true min-area objective, interleaved with targeted nudge search.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start initializations
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Structured
    p_struct = _structured_init()
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # 3. Random starts
    for _ in range(5):
        p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_rand = _project_to_triangle(p_rand)
        initializations.append(p_rand)
    
    # For each initialization, run a combined Nelder-Mead + Nudge optimization
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        current_pts = init_pts.copy()
        current_score = _min_triangle_area(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
        
        # Phase 1: Nelder-Mead on true min-area objective
        # We use a wrapper that projects points back to the triangle
        def nm_obj(pts_flat):
            pts_tmp = pts_flat.reshape(N, 2)
            pts_tmp = _project_to_triangle(pts_tmp)
            return -_min_triangle_area(pts_tmp)
        
        try:
            # Nelder-Mead options
            nm_options = {
                'maxiter': 100,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
            
            result = minimize(
                nm_obj,
                current_pts.flatten(),
                method='Nelder-Mead',
                options=nm_options
            )
            
            if result.success:
                new_pts = result.x.reshape(N, 2)
                new_pts = _project_to_triangle(new_pts)
                if _check_inside(new_pts):
                    new_score = _min_triangle_area(new_pts)
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_pts = current_pts.copy()
                            
        except Exception:
            pass
        
        # Phase 2: Nudge Search (Error Repair)
        # Run nudge search for a few seconds or a fixed number of iterations
        nudge_start = time.time()
        while time.time() - nudge_start < 2.0: # 2 seconds per nudge phase
            current_pts, current_score = _nudge_search(
                current_pts, rng, iterations=100, step_size=0.005
            )
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
            # Check global deadline
            if time.time() - start_time > DEADLINE - 5:
                break
                
        # If we have time, do another Nelder-Mead pass from the nudged position
        if time.time() - start_time < DEADLINE - 10:
            try:
                result = minimize(
                    nm_obj,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'adaptive': True}
                )
                if result.success:
                    new_pts = result.x.reshape(N, 2)
                    new_pts = _project_to_triangle(new_pts)
                    if _check_inside(new_pts):
                        new_score = _min_triangle_area(new_pts)
                        if new_score > best_score:
                            best_score = new_score
                            best_pts = new_pts.copy()
            except Exception:
                pass

    # Fallback if nothing worked
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
