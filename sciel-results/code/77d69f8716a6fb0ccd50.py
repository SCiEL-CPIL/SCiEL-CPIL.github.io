import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

TRI_IDX = np.array(list(combinations(range(N), 3)))

def min_triangle_area(points):
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def project_to_triangle(p):
    x, y = p
    if y < 0: y = 0
    if y > H: y = H
    x_min = y / SQRT3
    x_max = 1 - y / SQRT3
    if x < x_min: x = x_min
    if x > x_max: x = x_max
    return np.array([x, y])

def is_inside_triangle(p, eps=1e-9):
    x, y = p
    return (y >= -eps) and (y <= H + eps) and (x >= y/SQRT3 - eps) and (x <= 1 - y/SQRT3 + eps)

def generate_structured_initial():
    points = [
        [0.0, 0.0], [1.0, 0.0], [0.5, H],
        [0.5, 0.0], [0.25, H/2], [0.75, H/2],
        [0.5, H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.35, H/4], [0.65, H/4]
    ]
    return np.array(points[:N])

def generate_random_points(seed):
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1:
            x = u * 1.0 + v * 0.5
            y = v * H
            points.append([x, y])
    return np.array(points)

def greedy_max_min(seed):
    rng = np.random.RandomState(seed)
    points = []
    # Start with a random point
    u, v = rng.random(2)
    while u + v >= 1:
        u, v = rng.random(2)
    points.append([u * 1.0 + v * 0.5, v * H])
    
    for _ in range(N - 1):
        # Generate many candidates
        candidates = []
        for _ in range(500):
            u, v = rng.random(2)
            if u + v < 1:
                candidates.append([u * 1.0 + v * 0.5, v * H])
        candidates = np.array(candidates)
        
        # Find candidate with max min distance to existing points
        best_candidate = None
        best_min_dist = -1
        for c in candidates:
            dists = np.linalg.norm(points - c, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c
        points.append(best_candidate)
        
    return np.array(points)

def surrogate_optimize_min_distance(points, deadline, max_iter=1000):
    """Maximize minimum pairwise distance as a surrogate."""
    points = points.copy()
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Compute all pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                dists[i, j] = d
                dists[j, i] = d
        
        # Find min distance pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        if min_idx[0] == min_idx[1]:
            break
        i, j = min_idx
        
        # Move i and j apart
        # Direction from j to i
        direction = points[i] - points[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            # Random kick
            direction = np.random.randn(2)
            norm = 1.0
        direction /= norm
        
        # Step size
        step = 0.05
        new_i = points[i] + direction * step
        new_j = points[j] - direction * step
        
        new_i = project_to_triangle(new_i)
        new_j = project_to_triangle(new_j)
        
        # Check if it actually improved min distance (or at least didn't worsen the global min too much)
        # Simple heuristic: just accept if the specific pair distance increased
        old_dist = np.linalg.norm(points[i] - points[j])
        new_dist = np.linalg.norm(new_i - new_j)
        
        if new_dist > old_dist:
            points[i] = new_i
            points[j] = new_j
        else:
            # Try smaller step
            step *= 0.5
            if step < 1e-4:
                # Try random perturbation of the pair
                dx = np.random.randn(2) * 0.01
                new_i = project_to_triangle(points[i] + dx)
                new_j = project_to_triangle(points[j] - dx)
                if np.linalg.norm(new_i - new_j) > old_dist:
                    points[i] = new_i
                    points[j] = new_j
                else:
                    # Stuck, break or random restart for this pair
                    pass
    return points

def objective_flat(x):
    p = x.reshape(-1, 2)
    # Project
    p = np.array([project_to_triangle(pi) for pi in p])
    return -min_triangle_area(p)

def local_optimize_nm(points, deadline, max_iter=500):
    """Nelder-Mead optimization."""
    x0 = points.flatten()
    # Options for Nelder-Mead
    options = {'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    
    # We need to wrap the objective to check deadline?
    # scipy doesn't support callbacks easily for stopping.
    # We can just run it and hope it finishes, or check time after.
    # To be safe, we can run it in a way that we can interrupt, but standard minimize is blocking.
    # Given 120s limit, and NM is fast for 22 vars, it should be fine.
    
    res = minimize(objective_flat, x0, method='Nelder-Mead', options=options)
    
    if res.success or res.fun < objective_flat(x0):
        best_points = res.x.reshape(-1, 2)
        best_points = np.array([project_to_triangle(p) for p in best_points])
        return best_points
    else:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_points = None
    best_value = -1
    
    # 1. Structured Initial
    init_structured = generate_structured_initial()
    # Surrogate warmup
    warmup_1 = surrogate_optimize_min_distance(init_structured, deadline, max_iter=200)
    # NM Polish
    opt_1 = local_optimize_nm(warmup_1, deadline, max_iter=1000)
    val_1 = min_triangle_area(opt_1)
    if val_1 > best_value:
        best_value = val_1
        best_points = opt_1
    
    # 2. Greedy Max-Min Initial
    if time.time() < deadline - 20:
        init_greedy = greedy_max_min(seed=42)
        warmup_2 = surrogate_optimize_min_distance(init_greedy, deadline, max_iter=200)
        opt_2 = local_optimize_nm(warmup_2, deadline, max_iter=1000)
        val_2 = min_triangle_area(opt_2)
        if val_2 > best_value:
            best_value = val_2
            best_points = opt_2
            
    # 3. Random Initials with Surrogate
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        init_rand = generate_random_points(seed * 100 + 1)
        warmup_r = surrogate_optimize_min_distance(init_rand, deadline, max_iter=200)
        opt_r = local_optimize_nm(warmup_r, deadline, max_iter=1000)
        val_r = min_triangle_area(opt_r)
        if val_r > best_value:
            best_value = val_r
            best_points = opt_r
            
    # 4. Final Polish on Best
    if best_points is not None and time.time() < deadline - 10:
        # Perturb slightly
        perturbed = best_points + np.random.randn(*best_points.shape) * 0.01
        perturbed = np.array([project_to_triangle(p) for p in perturbed])
        opt_final = local_optimize_nm(perturbed, deadline, max_iter=1000)
        val_final = min_triangle_area(opt_final)
        if val_final > best_value:
            best_value = val_final
            best_points = opt_final
            
    if best_points is None:
        best_points = init_structured
        
    # Final verification
    for i in range(N):
        if not is_inside_triangle(best_points[i]):
            best_points[i] = project_to_triangle(best_points[i])
            
    return best_points
