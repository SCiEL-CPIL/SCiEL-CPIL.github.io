"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import time
import random

N = 11
H = np.sqrt(3) / 2.0
AREA_EQUI = np.sqrt(3) / 4.0

# Pre-compute indices for all combinations of 3 points
# C(11, 3) = 165 triangles
# We will use a vectorized approach to compute areas.
# Instead of looping in Python, we can use numpy broadcasting on the full set of points,
# but 11 points is small enough that a semi-vectorized loop or a pre-computed index matrix is fast.

# Pre-compute the 165 pairs of indices (i, j, k)
combos = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            combos.append((i, j, k))
combos = np.array(combos)
idx_i = combos[:, 0]
idx_j = combos[:, 1]
idx_k = combos[:, 2]

def _min_triangle_area_vec(points):
    """
    Vectorized computation of the minimum triangle area.
    points: shape (N, 2)
    """
    # Extract coordinates for all triangles
    pi = points[idx_i]  # shape (165, 2)
    pj = points[idx_j]
    pk = points[idx_k]
    
    # Area = 0.5 * | (xj - xi)(yk - yi) - (yj - yi)(xk - xi) |
    # Cross product of vectors (pj - pi) and (pk - pi)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _is_inside_triangle(points):
    """
    Check if all points are inside the unit equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    # Barycentric coordinates or half-plane tests
    # For efficiency, we can just check the constraints:
    # y >= 0
    # y <= sqrt(3) * x  (left side? no, right side is x - y/sqrt(3) <= 1? )
    # Let's use the standard inequalities for triangle (0,0), (1,0), (0.5, H)
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (line from (0,0) to (0.5, H) is y = sqrt(3)x. Wait, slope is H/0.5 = sqrt(3). So y <= sqrt(3)x is the region BELOW the line? No.
    #    The triangle is ABOVE y=0.
    #    The left edge is from (0,0) to (0.5, H). Equation: y = sqrt(3)x. Points inside are to the RIGHT/BOTTOM?
    #    Let's test center (0.5, H/3). H/3 = sqrt(3)/6. sqrt(3)*0.5 = sqrt(3)/2. H/3 < sqrt(3)/2. So y < sqrt(3)x is INSIDE?
    #    Wait, the vertex is (0.5, H). x=0.5, y=H. H = sqrt(3)/2. sqrt(3)*0.5 = sqrt(3)/2. So on the line.
    #    Point (0.1, 0.1). y=0.1, sqrt(3)*x = 0.173. 0.1 < 0.173. Inside?
    #    Let's check the other edge. (1,0) to (0.5, H). Slope = (H-0)/(0.5-1) = -sqrt(3).
    #    Equation: y - 0 = -sqrt(3)(x - 1) => y = -sqrt(3)x + sqrt(3).
    #    Test center (0.5, H/3). H/3 = 0.288. -sqrt(3)(0.5) + sqrt(3) = sqrt(3)/2 = 0.866. 0.288 < 0.866.
    #    So the conditions are:
    #    y >= 0
    #    y <= sqrt(3) * x  (This is the constraint for the left edge? No, the left edge is x = y/sqrt(3). So x >= y/sqrt(3) => y <= sqrt(3)x)
    #    y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)
    
    y = points[:, 1]
    x = points[:, 0]
    
    cond1 = y >= 0
    cond2 = y <= np.sqrt(3) * x
    cond3 = y <= np.sqrt(3) * (1 - x)
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def _project_to_triangle(points):
    """
    Project points slightly outside the triangle back inside to maintain validity.
    Simple heuristic: if outside, move towards centroid.
    """
    centroid = np.array([0.5, H / 3.0])
    valid = _is_inside_triangle(points)
    if not valid:
        # Find invalid points
        y = points[:, 1]
        x = points[:, 0]
        cond1 = y >= 0
        cond2 = y <= np.sqrt(3) * x
        cond3 = y <= np.sqrt(3) * (1 - x)
        
        # For simplicity and speed, if any point is invalid, we can nudge the whole set or just return a penalty.
        # In DE, we usually use a penalty or repair.
        # Let's try a simple repair: clip to bounds and if still invalid, shrink towards center.
        # However, checking validity every iteration is expensive.
        # Alternative: use a penalty in the objective function.
        pass
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    rng = np.random.RandomState(42)
    
    # Initialize population
    # Population size 50
    pop_size = 50
    points_pop = np.zeros((pop_size, N, 2))
    
    # Generate random points inside triangle
    # Rejection sampling or barycentric
    # Barycentric: p = u*v1 + w*v2 + (1-u-w)*v3
    # v1=(0,0), v2=(1,0), v3=(0.5, H)
    # p = u*(1,0) + w*(0.5, H) = (u + 0.5w, w*H)
    # Conditions: u>=0, w>=0, u+w<=1
    
    for i in range(pop_size):
        for j in range(N):
            while True:
                u = rng.random()
                w = rng.random()
                if u + w <= 1.0:
                    points_pop[i, j] = [u + 0.5 * w, w * H]
                    break
    
    best_score = -np.inf
    best_points = None
    
    # Differential Evolution
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    # Nelder-Mead local improvement on top candidates?
    # Let's stick to DE for global search, it's robust.
    
    # To improve speed, we can use a "fast" DE variant or just limit generations.
    # Each generation: pop_size evaluations.
    # Evaluation cost: 165 triangles * simple math. ~1000 ops. Very fast.
    # 145 seconds is a lot. We can do thousands of generations.
    
    # However, to ensure we find the global optimum, we might need more sophisticated local search.
    # Let's add a local hill-climbing step on the best candidate of each generation.
    
    def _local_optimize(points, max_iter=100):
        """Simple coordinate descent / hill climbing."""
        current = points.copy()
        current_score = _min_triangle_area_vec(current)
        
        for it in range(max_iter):
            improved = False
            # Try moving each point in small random directions
            # Or use a fixed grid of small moves?
            # Let's use random perturbations
            for _ in range(5):
                candidate = current.copy()
                # Pick a random point
                idx = rng.randint(0, N - 1)
                # Perturb
                dx = rng.normal(0, 0.01)
                dy = rng.normal(0, 0.01)
                candidate[idx] += [dx, dy]
                
                # Check validity
                if not _is_inside_triangle(candidate):
                    continue
                    
                cand_score = _min_triangle_area_vec(candidate)
                if cand_score > current_score:
                    current = candidate
                    current_score = cand_score
                    improved = True
            
            if not improved:
                break
                
        return current, current_score

    for gen in range(10000):
        if time.time() > deadline:
            break
            
        # Update population
        new_pop = np.zeros_like(points_pop)
        
        # Keep best
        scores = np.array([_min_triangle_area_vec(points_pop[i]) for i in range(pop_size)])
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_points = points_pop[best_idx].copy()
        
        # Local optimization on the current best
        if gen % 10 == 0:
            loc_best, loc_score = _local_optimize(best_points, max_iter=20)
            if loc_score > best_score:
                best_score = loc_score
                best_points = loc_best
                # Update the best in population
                points_pop[best_idx] = best_points
                scores[best_idx] = best_score
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
                best_points = points_pop[best_idx].copy()

        for i in range(pop_size):
            # Differential Evolution mutation
            # r1, r2, r3 distinct from i
            while True:
                r1 = rng.randint(0, pop_size - 1)
                r2 = rng.randint(0, pop_size - 1)
                r3 = rng.randint(0, pop_size - 1)
                if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                    break
            
            v = points_pop[r1] + F * (points_pop[r2] - points_pop[r3])
            
            # Crossover
            trial = points_pop[i].copy()
            j_rand = rng.randint(0, N - 1)
            for j in range(N):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Check validity
            if _is_inside_triangle(trial):
                trial_score = _min_triangle_area_vec(trial)
            else:
                trial_score = -np.inf
            
            if trial_score > scores[i]:
                points_pop[i] = trial
                scores[i] = trial_score
        
        # Update best
        new_best_idx = np.argmax(scores)
        if scores[new_best_idx] > best_score:
            best_score = scores[new_best_idx]
            best_points = points_pop[new_best_idx].copy()

    # Final check
    if best_points is None:
        # Fallback to random
        pts = np.zeros((N, 2))
        for j in range(N):
            while True:
                u = rng.random()
                w = rng.random()
                if u + w <= 1.0:
                    pts[j] = [u + 0.5 * w, w * H]
                    break
        return pts

    # Ensure best_points is valid
    if not _is_inside_triangle(best_points):
        # Fallback
        pts = np.zeros((N, 2))
        for j in range(N):
            while True:
                u = rng.random()
                w = rng.random()
                if u + w <= 1.0:
                    pts[j] = [u + 0.5 * w, w * H]
                    break
        return pts

    return best_points
