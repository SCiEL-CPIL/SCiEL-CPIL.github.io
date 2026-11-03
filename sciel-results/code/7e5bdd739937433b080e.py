"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
N = 11
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def _bary_to_cart(u, v):
    """Convert barycentric (u,v) to Cartesian. u=weight of V1, v=weight of V2."""
    return u * V1 + v * V2

def _cart_to_bary(p):
    """Convert Cartesian point to barycentric (u,v)."""
    px, py = p
    v = 2.0 * py / np.sqrt(3)
    u = px - 0.5 * v
    return u, v

def _clamp_bary(u, v):
    """Clamp barycentric coords to valid region u>=0, v>=0, u+v<=1."""
    if u < 0: u = 0.0
    if v < 0: v = 0.0
    if u + v > 1:
        s = u + v
        u /= s
        v /= s
    return u, v

def _min_triangle_area_bary(bary_pts):
    """Compute min triangle area from barycentric points (n,2 array of (u,v))."""
    n = len(bary_pts)
    # Convert to Cartesian for area computation
    cart = np.array([_bary_to_cart(u, v) for u, v in bary_pts])
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((cart[j][0] - cart[i][0]) * (cart[k][1] - cart[i][1]) -
                                  (cart[k][0] - cart[i][0]) * (cart[j][1] - cart[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_vec(bary_pts):
    """Vectorized min triangle area computation."""
    n = len(bary_pts)
    cart = np.array([_bary_to_cart(u, v) for u, v in bary_pts])
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((cart[j][0] - cart[i][0]) * (cart[k][1] - cart[i][1]) -
                                  (cart[k][0] - cart[i][0]) * (cart[j][1] - cart[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _objective_bary(x):
    """Objective: negative min triangle area in barycentric coords."""
    bary_pts = x.reshape(-1, 2)
    # Clamp to valid region
    clamped = np.array([_clamp_bary(u, v) for u, v in bary_pts])
    return -_min_triangle_area_bary(clamped)

def _greedy_farthest_seed(n_points, seed=42):
    """Greedy farthest-point sampling for initial configuration."""
    rng = np.random.RandomState(seed)
    # Start with a random point
    u0 = rng.random()
    v0 = rng.random() * (1 - u0)
    bary_pts = [(u0, v0)]
    
    for _ in range(n_points - 1):
        # Generate candidate points
        candidates = []
        for _ in range(200):
            u = rng.random()
            v = rng.random() * (1 - u)
            candidates.append((u, v))
        
        # Find candidate that maximizes min distance to existing points
        best_cand = None
        best_min_dist = -1
        for cand in candidates:
            min_dist = float('inf')
            for existing in bary_pts:
                dist = np.sqrt((cand[0] - existing[0])**2 + (cand[1] - existing[1])**2)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_cand = cand
        
        bary_pts.append(best_cand)
    
    return np.array(bary_pts)

def _structured_init_bary(seed=42):
    """Structured initialization in barycentric coordinates."""
    rng = np.random.RandomState(seed)
    # Use a grid-like structure in the simplex
    # Points at vertices, edge midpoints, and interior
    bary_pts = []
    
    # Vertices
    bary_pts.append((0, 0))      # V0
    bary_pts.append((1, 0))      # V1
    bary_pts.append((0, 1))      # V2
    
    # Edge midpoints
    bary_pts.append((0.5, 0))    # mid V0-V1
    bary_pts.append((0.5, 0.5))  # mid V1-V2
    bary_pts.append((0, 0.5))    # mid V0-V2
    
    # Centroid
    bary_pts.append((1/3, 1/3))
    
    # 4 more interior points
    # Place them at 1/4 and 3/4 along medians
    bary_pts.append((1/6, 1/6))
    bary_pts.append((2/3, 1/6))
    bary_pts.append((1/6, 2/3))
    bary_pts.append((1/2, 0))    # adjust to fill
    
    # Ensure we have exactly 11
    while len(bary_pts) < N:
        u = rng.random()
        v = rng.random() * (1 - u)
        bary_pts.append((u, v))
    bary_pts = bary_pts[:N]
    
    return np.array(bary_pts)

def _random_init_bary(seed=42):
    """Random initialization in barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random() * (1 - u)
        bary_pts.append((u, v))
    return np.array(bary_pts)

def _simulated_annealing(bary_pts, n_iter=5000, t0=0.1, tf=0.001, seed=42):
    """Simulated annealing in barycentric space."""
    rng = np.random.RandomState(seed)
    current = bary_pts.copy()
    current_score = _min_triangle_area_bary(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        t = t0 * (tf / t0) ** (i / n_iter)
        
        # Pick a random point to perturb
        idx = rng.randint(N)
        du = rng.normal(0, 0.05)
        dv = rng.normal(0, 0.05)
        
        new_u = current[idx][0] + du
        new_v = current[idx][1] + dv
        new_u, new_v = _clamp_bary(new_u, new_v)
        
        new_pts = current.copy()
        new_pts[idx] = (new_u, new_v)
        new_score = _min_triangle_area_bary(new_pts)
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best, best_score

def _local_refine_bary(bary_pts, max_iter=500):
    """Local refinement using coordinate descent in barycentric space."""
    pts = bary_pts.copy()
    current_score = _min_triangle_area_bary(pts)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            best_u, best_v = pts[i][0], pts[i][1]
            best_s = current_score
            
            # Try small perturbations
            for du in [-0.02, -0.01, 0.01, 0.02]:
                for dv in [-0.02, -0.01, 0.01, 0.02]:
                    nu = pts[i][0] + du
                    nv = pts[i][1] + dv
                    nu, nv = _clamp_bary(nu, nv)
                    test_pts = pts.copy()
                    test_pts[i] = (nu, nv)
                    s = _min_triangle_area_bary(test_pts)
                    if s > best_s:
                        best_s = s
                        best_u, best_v = nu, nv
            
            if (best_u, best_v) != (pts[i][0], pts[i][1]):
                pts[i] = (best_u, best_v)
                current_score = best_s
                improved = True
        
        if not improved:
            break
    
    return pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_bary = None
    best_score = -float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # Structured
    for seed in [42, 123, 456, 789, 1000]:
        inits.append(_structured_init_bary(seed))
    
    # Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        inits.append(_random_init_bary(seed))
    
    # Greedy farthest
    for seed in [11, 22, 33, 44, 55]:
        inits.append(_greedy_farthest_seed(N, seed))
    
    # Run simulated annealing + local refinement on each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        # Simulated annealing
        sa_pts, sa_score = _simulated_annealing(init, n_iter=3000, t0=0.08, tf=0.001, seed=i)
        
        # Local refinement
        refined_pts, refined_score = _local_refine_bary(sa_pts, max_iter=200)
        
        if refined_score > best_score:
            best_score = refined_score
            best_bary = refined_pts.copy()
    
    # Final local refinement on the best
    if best_bary is not None and time.time() < deadline - 5.0:
        final_pts, final_score = _local_refine_bary(best_bary, max_iter=500)
        if final_score > best_score:
            best_score = final_score
            best_bary = final_pts.copy()
    
    # Convert to Cartesian
    if best_bary is None:
        best_bary = inits[0]
    
    cart_pts = np.array([_bary_to_cart(u, v) for u, v in best_bary])
    
    return cart_pts
