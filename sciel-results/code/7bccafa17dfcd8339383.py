import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

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
        for _ in range(500):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """
    Structured initialization: Vertices, Midpoints, Center, and some others.
    """
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(p_struct)

def _active_set_local_search(pts, rng, time_limit):
    """
    Direct Search / Derivative-Free Optimization strategy.
    1. Identify the active set of triplets (those with area close to the minimum).
    2. Perform a local search (Nelder-Mead or random nudges) to increase the minimum area.
    3. Project back to the triangle if necessary.
    """
    current_pts = pts.copy()
    current_score = _min_triangle_area(current_pts)
    
    # We will use a simple iterative improvement loop
    # At each step, we identify the worst triplet and try to move points to improve it.
    # Or, we can use Nelder-Mead on the full point set, but constrained to stay inside.
    
    # Let's try a combination:
    # 1. Nelder-Mead with a penalty for going outside.
    # 2. If stuck, do random nudges on the active set points.
    
    def objective(x):
        # x is flattened points
        p = x.reshape(N, 2)
        # Penalty for being outside
        penalty = 0
        for pi in p:
            x, y = pi
            if y < 0: penalty += 100 * y**2
            if y > np.sqrt(3) * x: penalty += 100 * (y - np.sqrt(3) * x)**2
            if y > np.sqrt(3) * (1 - x): penalty += 100 * (y - np.sqrt(3) * (1 - x))**2
            if x < 0: penalty += 100 * x**2
            if x > 1: penalty += 100 * (x - 1)**2
        return -_min_triangle_area(p) + penalty

    x0 = current_pts.flatten()
    
    # Use Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < objective(x0):
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_triangle(new_pts)
            new_score = _min_triangle_area(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
    except Exception:
        pass
    
    # Additional local refinement: Random nudges on active set
    for _ in range(50):
        if time.time() > time_limit:
            break
            
        # Find active triplets
        p_i = current_pts[_i_idx]
        p_j = current_pts[_j_idx]
        p_k = current_pts[_k_idx]
        dx1 = p_j - p_i
        dx2 = p_k - p_i
        cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Threshold for active set
        active_thresh = current_score * 1.01
        active_mask = areas < active_thresh
        
        if not np.any(active_mask):
            break
            
        active_triplets = _triplets[active_mask]
        
        # Pick a random active triplet
        tri = active_triplets[rng.randint(0, len(active_triplets)-1)]
        i, j, k = tri
        
        # Try to move one of the points slightly to increase the area of this triplet
        # and hopefully not decrease others too much.
        # We'll just do a random small perturbation of one of the 3 points.
        point_idx = rng.choice([i, j, k])
        orig_p = current_pts[point_idx].copy()
        
        # Random direction
        angle = rng.uniform(0, 2*np.pi)
        step_size = rng.uniform(0.001, 0.01)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_p = orig_p + np.array([dx, dy])
        
        # Check if new point is inside
        if not _check_inside(np.array([new_p])):
            # Project
            new_p = _project_to_triangle(np.array([new_p]))[0]
            
        # Test
        test_pts = current_pts.copy()
        test_pts[point_idx] = new_p
        test_score = _min_triangle_area(test_pts)
        
        if test_score > current_score:
            current_pts = test_pts
            current_score = test_score
        else:
            # If it didn't improve, try a smaller step or different direction?
            # For simplicity, just move on.
            pass
            
    return current_pts, current_score

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Derivative-Free Active-Set Local Search.
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
        
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 3. Structured
    p_struct = _structured_init()
    initializations.append(p_struct)
    
    # 4. Vertices + Center + Random
    p_vcr = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, S3H/3],
        _random_point_in_triangle(rng), _random_point_in_triangle(rng),
        _random_point_in_triangle(rng), _random_point_in_triangle(rng),
        _random_point_in_triangle(rng), _random_point_in_triangle(rng),
        _random_point_in_triangle(rng)
    ])
    p_vcr = _project_to_triangle(p_vcr)
    initializations.append(p_vcr)
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Refine this initialization
        refined_pts, score = _active_set_local_search(init_pts, rng, DEADLINE - 5)
        
        if score > best_score:
            best_score = score
            best_pts = refined_pts.copy()
            
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
