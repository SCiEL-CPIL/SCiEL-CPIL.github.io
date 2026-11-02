"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use SLSQP with a Smoothed Minimum (Soft-Min) objective function.
   This avoids the non-differentiability of the min operator and the high
   constraint count of the previous formulation.
2. Anneal the smoothing parameter (beta) to move from a global landscape
   to a sharp local minimum of the true min-area.
3. Multi-start with diverse initializations.
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

def _soft_min_area(pts, beta):
    """
    Compute the soft-minimum of triangle areas.
    soft_min(a) = -1/beta * log(sum(exp(-beta * a_i)))
    As beta -> inf, this approaches min(a_i).
    """
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Numerical stability: subtract max area
    a_max = np.max(areas)
    if a_max == 0:
        return 0.0
    
    # exp(-beta * (a_i - a_max)) = exp(-beta*a_i + beta*a_max)
    # log(sum(exp(-beta*a_i))) = beta*a_max + log(sum(exp(-beta*(a_i - a_max))))
    # soft_min = -1/beta * (beta*a_max + log_sum_exp) = -a_max - 1/beta * log_sum_exp
    
    scaled = -beta * (areas - a_max)
    # Clip to avoid underflow/overflow in exp
    scaled = np.clip(scaled, -700, 700)
    
    log_sum = np.log(np.sum(np.exp(scaled)))
    
    soft_min_val = -a_max - (1.0 / beta) * log_sum
    return soft_min_val

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
    """Create a structured initialization based on known Heilbronn-like configurations."""
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses SLSQP with a smoothed minimum objective and annealing.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start optimization
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
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # 4. More random starts
    for _ in range(5):
        p_rand2 = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_rand2 = _project_to_triangle(p_rand2)
        initializations.append(p_rand2)
    
    # Define constraints for triangle boundaries
    def triangle_constraints(x):
        pts = x.reshape(N, 2)
        c1 = pts[:, 1] # y >= 0
        c2 = np.sqrt(3) * pts[:, 0] - pts[:, 1] # y <= sqrt(3)*x
        c3 = np.sqrt(3) * (1 - pts[:, 0]) - pts[:, 1] # y <= sqrt(3)*(1-x)
        c4 = pts[:, 0] # x >= 0
        c5 = 1 - pts[:, 0] # x <= 1
        # Combine into one vector
        return np.concatenate([c1, c2, c3, c4, c5])
    
    cons = [
        {'type': 'ineq', 'fun': triangle_constraints}
    ]
    
    bounds = [(0, 1), (0, S3H)] * N
    
    # Annealing schedule for beta
    # Start with low beta (smooth) and increase to high beta (sharp)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        current_pts = init_pts.copy()
        
        # Anneal beta
        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            def objective(x):
                pts = x.reshape(N, 2)
                # Soft min is what we want to maximize, so we minimize -soft_min
                return -_soft_min_area(pts, beta)
            
            try:
                result = minimize(
                    objective,
                    current_pts.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 50, 'ftol': 1e-8}
                )
                
                if result.success:
                    new_pts = result.x.reshape(N, 2)
                    new_pts = _project_to_triangle(new_pts)
                    
                    # Check if valid and improved
                    if _check_inside(new_pts):
                        # Update current points for next beta step
                        current_pts = new_pts
                        # We don't update best_score here yet, only at the end of annealing
                        # or if we want to track intermediate improvements.
                        # Let's track it to be safe.
                        score = _min_triangle_area(current_pts)
                        if score > best_score:
                            best_score = score
                            best_pts = current_pts.copy()
                            
            except Exception as e:
                pass
                
        # Final evaluation of this chain
        score = _min_triangle_area(current_pts)
        if score > best_score:
            best_score = score
            best_pts = current_pts.copy()

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
