"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _softmin_objective(x, beta=50.0):
    """Soft-min surrogate: -softmin(areas). Differentiable approximation of -min(areas)."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    areas = _all_triangle_areas(points)
    # Soft min: -logsumexp(-beta * areas) / beta
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    softmin = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -softmin

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _clamp_cart(points):
    """Clamp Cartesian points to be inside the triangle."""
    bary = _cart_to_bary(points)
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)

def _hierarchical_subdivision_init():
    """Deterministic hierarchical subdivision into 11 equal-area sub-regions.
    
    Strategy: Divide the equilateral triangle into 3 main regions (top, bottom-left, bottom-right),
    then further subdivide to get 11 centroids.
    
    We use a recursive approach:
    - Level 0: 1 triangle
    - Level 1: 4 triangles (connect midpoints)
    - Level 2: 9 triangles (further subdivide)
    - We select 11 centroids from a mixed subdivision
    """
    # Start with the main triangle
    # Use a structured approach: place points at centroids of sub-triangles
    
    # Method: Use a grid-like subdivision
    # Divide base into 3 segments, create parallel lines
    
    # For 11 points, use a combination:
    # 3 vertices + 3 edge midpoints + 1 centroid + 4 interior points
    
    # Better: Use a hexagonal packing pattern adapted to triangle
    # Or: Use the 11-point optimal configuration known for Heilbronn problem
    
    # Let's use a deterministic subdivision:
    # Split into 3 corner triangles + 1 central hexagon region
    # Then subdivide the hexagon into 4 more regions
    
    # Actually, let's use a simpler approach:
    # Create a 3x3 grid in barycentric coordinates and pick 11 points
    
    # Barycentric grid points (u, v) where u+v <= 1
    # Use a fine grid and select well-spaced points
    
    # Alternative: Use the known near-optimal 11-point configuration
    # Based on literature, a good configuration places points symmetrically
    
    # Symmetric configuration with 3-fold symmetry:
    # 1 center point + 3 at distance r1 + 3 at distance r2 + 4 more
    
    # Let me use a more systematic approach:
    # Divide triangle into 11 regions using Voronoi-like subdivision
    
    # For simplicity and determinism, use this structured pattern:
    # Vertices (3), edge midpoints (3), centroid (1), and 4 interior points
    
    bary = np.array([
        [0.0, 0.0],      # V0
        [1.0, 0.0],      # V1
        [0.0, 1.0],      # V2
        [0.5, 0.0],      # midpoint V0-V1
        [0.5, 0.5],      # midpoint V1-V2
        [0.0, 0.5],      # midpoint V0-V2
        [1/3, 1/3],      # centroid
        # 4 interior points arranged symmetrically
        [0.25, 0.15],
        [0.60, 0.15],
        [0.15, 0.55],
        [0.45, 0.40],
    ])
    
    return _clamp_bary(bary)

def _repulsion_relaxation(bary, n_iter=100, alpha=0.01, beta_rep=1.0):
    """Repulsion-based relaxation to maximize minimum pairwise distance.
    
    Iteratively move each point away from others, with damping to stay in triangle.
    """
    bary = bary.copy()
    points = _bary_to_cart(bary)
    
    for it in range(n_iter):
        # Compute pairwise distances
        for i in range(N):
            # Force from all other points (repulsion)
            force = np.zeros(2)
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    # Repulsion: inverse square law
                    force += beta_rep * diff / (dist ** 2)
            
            # Move point in direction of force
            new_point = points[i] + alpha * force
            new_point = _clamp_cart(new_point)
            points[i] = new_point
        
        # Convert back to barycentric
        bary = _cart_to_bary(points)
        bary = _clamp_bary(bary)
    
    return bary

def _l_bfgs_softmin(bary0, maxiter=500, beta=50.0):
    """Run L-BFGS-B optimization using soft-min surrogate."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _softmin_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'eps': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _nelder_mead_optimize_bary(bary0, maxiter=200):
    """Run Nelder-Mead optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        
        def obj(x):
            bary = _clamp_bary(x.reshape(N, 2))
            points = _bary_to_cart(bary)
            return -_min_triangle_area_vectorized(points)
        
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _simulated_annealing(bary0, max_iters=5000, t_start=0.1, t_end=0.001, seed=42):
    """Simulated annealing in barycentric coordinates with point-move neighborhood."""
    rng = np.random.RandomState(seed)
    bary = bary0.copy()
    points = _bary_to_cart(bary)
    current_score = _min_triangle_area_vectorized(points)
    best_bary = bary.copy()
    best_score = current_score
    
    n_triples = len(_get_indices())
    
    for it in range(max_iters):
        # Cooling schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Choose a random point to move
        idx = rng.randint(N)
        
        # Generate a new position for that point
        new_bary = bary.copy()
        new_bary[idx] = bary[idx] + rng.normal(0, 0.05, size=2)
        new_bary = _clamp_bary(new_bary)
        
        new_points = _bary_to_cart(new_bary)
        new_score = _min_triangle_area_vectorized(new_points)
        
        # Accept if better or with probability exp((new - current) / t)
        if new_score >= current_score:
            bary = new_bary
            points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_bary = bary.copy()
        else:
            delta = new_score - current_score
            if delta > 0:
                bary = new_bary
                points = new_points
                current_score = new_score
            else:
                accept_prob = np.exp(delta / t)
                if rng.random() < accept_prob:
                    bary = new_bary
                    points = new_points
                    current_score = new_score
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    def score_bary(bary):
        points = _bary_to_cart(bary)
        return _min_triangle_area_vectorized(points)
    
    # Phase 1: Hierarchical subdivision init + repulsion relaxation + soft-min optimization
    if time_left() > 5.0:
        bary_hier = _hierarchical_subdivision_init()
        bary_relaxed = _repulsion_relaxation(bary_hier, n_iter=200, alpha=0.005, beta_rep=0.5)
        score_relaxed = score_bary(bary_relaxed)
        
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_bary = bary_relaxed.copy()
        
        # Refine with soft-min L-BFGS
        if time_left() > 3.0:
            bary_opt = _l_bfgs_softmin(bary_relaxed, maxiter=300, beta=50.0)
            score_opt = score_bary(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
    
    # Phase 2: Multiple structured initializations with local optimization
    initial_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in initial_seeds:
        if time_left() < 3.0:
            break
        
        # Use hierarchical init as base, add jitter
        bary_init = _hierarchical_subdivision_init()
        rng = np.random.RandomState(seed)
        jitter = rng.normal(0, 0.03, size=bary_init.shape)
        bary_init = _clamp_bary(bary_init + jitter)
        
        score_init = score_bary(bary_init)
        
        # Local optimization with soft-min L-BFGS
        if time_left() > 2.0:
            bary_opt = _l_bfgs_softmin(bary_init, maxiter=200, beta=40.0)
            score_opt = score_bary(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()
    
    # Phase 3: Simulated annealing from best solution
    if time_left() > 10.0:
        sa_bary, sa_score = _simulated_annealing(
            best_bary, max_iters=3000, t_start=0.05, t_end=0.0005, seed=42
        )
        if sa_score > best_score:
            best_score = sa_score
            best_bary = sa_bary.copy()
            
            # Refine with local optimization
            if time_left() > 5.0:
                bary_refined = _l_bfgs_softmin(best_bary, maxiter=300, beta=60.0)
                score_refined = score_bary(bary_refined)
                if score_refined > best_score:
                    best_score = score_refined
                    best_bary = bary_refined.copy()
    
    # Phase 4: More simulated annealing restarts
    rng = np.random.RandomState(123)
    sa_restarts = 0
    max_sa_restarts = 8
    
    while sa_restarts < max_sa_restarts and time_left() > 5.0:
        sa_restarts += 1
        
        # Perturb the best solution for SA start
        pert = rng.normal(0, 0.03, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        sa_bary, sa_score = _simulated_annealing(
            bary_start, max_iters=2000, t_start=0.03, t_end=0.0003, seed=42+sa_restarts
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_bary = sa_bary.copy()
            
            # Quick local refinement
            if time_left() > 3.0:
                bary_refined = _l_bfgs_softmin(best_bary, maxiter=200, beta=50.0)
                score_refined = score_bary(bary_refined)
                if score_refined > best_score:
                    best_score = score_refined
                    best_bary = bary_refined.copy()
    
    # Phase 5: Fine-tuning with very small perturbations
    if time_left() > 2.0:
        for ft in range(10):
            if time_left() < 1.0:
                break
            pert = rng.normal(0, 0.005, size=(N, 2))
            bary_ft = _clamp_bary(best_bary + pert)
            bary_opt = _l_bfgs_softmin(bary_ft, maxiter=200, beta=70.0)
            score_opt = score_bary(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _hierarchical_subdivision_init()
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
