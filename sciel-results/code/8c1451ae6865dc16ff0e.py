"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N,3) to Cartesian (N,2)."""
    # bary[i] = [a, b, c] where a+b+c=1, point = a*V0 + b*V1 + c*V2
    verts = np.array([V0, V1, V2])  # (3, 2)
    return bary @ verts  # (N, 2)

def _min_area_ratio_from_bary(bary):
    """Compute min triangle area / TRI_AREA from barycentric coordinates (N,3)."""
    n = len(bary)
    min_ratio = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area ratio = |det of 3x3 matrix with rows [1, bi, bj, bk]| / 2
                # Actually, area of triangle formed by points with bary coords
                # equals |det([[1,a1,b1,c1],[1,a2,b2,c2],[1,a3,b3,c3]])| * TRI_AREA / 2
                # Let me use the determinant formula directly
                b1, b2, b3 = bary[i], bary[j], bary[k]
                # det = |b1[0]*(b2[1]*b3[2]-b2[2]*b3[1]) - b1[1]*(b2[0]*b3[2]-b2[2]*b3[0]) + b1[2]*(b2[0]*b3[1]-b2[1]*b3[0])|
                # But simpler: area_ratio = 0.5 * |det of [[1,b1[0],b1[1],b1[2]], ...]| 
                # Actually the cleanest: compute in Cartesian
                p1 = b1[0]*V0 + b1[1]*V1 + b1[2]*V2
                p2 = b2[0]*V0 + b2[1]*V1 + b2[2]*V2
                p3 = b3[0]*V0 + b3[1]*V1 + b3[2]*V2
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                ratio = area / TRI_AREA
                if ratio < min_ratio:
                    min_ratio = ratio
    return min_ratio

def _objective_bary(x):
    """Objective: negative min area ratio. x is (N,3) flattened."""
    bary = x.reshape(N, 3)
    # Project to simplex (a+b+c=1, a,b,c >= 0)
    bary = np.maximum(bary, 0)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-15, 1.0, sums)
    bary = bary / sums
    ratio = _min_area_ratio_from_bary(bary)
    return -ratio

def _project_to_bary_simplex(x):
    """Project flattened (N*3,) vector to valid barycentric coords."""
    bary = x.reshape(N, 3)
    bary = np.maximum(bary, 0)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-15, 1.0, sums)
    bary = bary / sums
    return bary

def _structured_init_fps(seed=42):
    """Farthest-point sampling in barycentric space for structured initialization."""
    rng = np.random.RandomState(seed)
    # Start with 3 vertices
    bary = np.zeros((N, 3))
    bary[0] = [1.0, 0.0, 0.0]  # V0
    bary[1] = [0.0, 1.0, 0.0]  # V1
    bary[2] = [0.0, 0.0, 1.0]  # V2
    bary[3] = [1.0/3, 1.0/3, 1.0/3]  # centroid
    
    for i in range(4, N):
        # Generate random candidates
        candidates = rng.dirichlet([1.0, 1.0, 1.0], size=200)
        best_idx = -1
        best_min_dist = -1
        for c in candidates:
            # Compute min distance to existing points
            dists = np.sqrt(((bary[:i] - c) ** 2).sum(axis=1))
            min_d = dists.min()
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = c
        bary[i] = best_idx
    
    return bary

def _local_optimize_slsqp(bary_init, time_limit=10.0):
    """Optimize using SLSQP with equality constraint sum=1 and inequality constraints >= 0."""
    x0 = bary_init.flatten()
    
    # Equality constraints: sum of each row = 1
    eq_cons = []
    for i in range(N):
        eq_cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: x[3*i] + x[3*i+1] + x[3*i+2] - 1.0
        })
    
    # Inequality constraints: each bary >= 0
    ineq_cons = []
    for i in range(N):
        for j in range(3):
            ineq_cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: x[3*i + j]
            })
    
    try:
        res = minimize(
            _objective_bary,
            x0,
            method='SLSQP',
            constraints=eq_cons + ineq_cons,
            options={'maxiter': 300, 'ftol': 1e-8}
        )
        bary_opt = _project_to_bary_simplex(res.x)
        score = _min_area_ratio_from_bary(bary_opt)
        return bary_opt, score
    except Exception:
        return bary_init, _min_area_ratio_from_bary(bary_init)

def _local_optimize_lbfgs_polish(bary_init, time_limit=5.0):
    """Polish with L-BFGS-B after SLSQP."""
    x0 = bary_init.flatten()
    # Use bounds [0, 1] for each bary coordinate
    bounds = [(0.0, 1.0)] * (N * 3)
    
    try:
        res = minimize(
            _objective_bary,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        bary_opt = _project_to_bary_simplex(res.x)
        score = _min_area_ratio_from_bary(bary_opt)
        return bary_opt, score
    except Exception:
        return bary_init, _min_area_ratio_from_bary(bary_init)

def _generate_diverse_inits():
    """Generate multiple diverse initial configurations."""
    inits = []
    
    # FPS-based initialization
    inits.append(_structured_init_fps(seed=42))
    
    # Grid-based initialization
    grid_bary = np.zeros((N, 3))
    grid_bary[0] = [1.0, 0.0, 0.0]
    grid_bary[1] = [0.0, 1.0, 0.0]
    grid_bary[2] = [0.0, 0.0, 1.0]
    grid_bary[3] = [0.5, 0.5, 0.0]
    grid_bary[4] = [0.5, 0.0, 0.5]
    grid_bary[5] = [0.0, 0.5, 0.5]
    grid_bary[6] = [1.0/3, 1.0/3, 1.0/3]
    grid_bary[7] = [0.75, 0.125, 0.125]
    grid_bary[8] = [0.125, 0.75, 0.125]
    grid_bary[9] = [0.125, 0.125, 0.75]
    grid_bary[10] = [0.5, 0.25, 0.25]
    inits.append(grid_bary)
    
    # Random initializations with different seeds
    for seed in [1, 7, 13, 21]:
        rng = np.random.RandomState(seed)
        bary = rng.dirichlet([1.0, 1.0, 1.0], size=N)
        inits.append(bary)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_bary = None
    best_score = -float('inf')
    
    # Generate diverse initializations
    inits = _generate_diverse_inits()
    
    # Try each initialization with SLSQP + L-BFGS-B polishing
    for idx, bary_init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # SLSQP optimization
        bary_opt, score = _local_optimize_slsqp(bary_init)
        if score > best_score:
            best_score = score
            best_bary = bary_opt
        
        # Polish with L-BFGS-B
        if time.time() > deadline - 10.0:
            break
        bary_polished, score2 = _local_optimize_lbfgs_polish(bary_opt)
        if score2 > best_score:
            best_score = score2
            best_bary = bary_polished
    
    # If time remains, try more random starts
    if time.time() < deadline - 15.0:
        for seed in [100, 200, 300, 400, 500]:
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(seed)
            bary_rand = rng.dirichlet([2.0, 2.0, 2.0], size=N)
            bary_opt, score = _local_optimize_slsqp(bary_rand)
            if score > best_score:
                best_score = score
                best_bary = bary_opt
    
    # Convert best barycentric to Cartesian
    if best_bary is None:
        # Fallback: vertices + centroid + midpoints
        best_bary = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1/3, 1/3, 1/3],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.75, 0.125, 0.125],
            [0.125, 0.75, 0.125],
            [0.125, 0.125, 0.75],
            [0.5, 0.25, 0.25]
        ])
    
    points = _bary_to_cart(best_bary)
    
    # Final validation: ensure all points are inside the triangle
    # Check barycentric coords are non-negative and sum to 1
    for i in range(N):
        if best_bary[i].min() < -1e-9:
            best_bary[i] = np.maximum(best_bary[i], 0)
            s = best_bary[i].sum()
            if s < 1e-15:
                best_bary[i] = [1/3, 1/3, 1/3]
            else:
                best_bary[i] /= s
    
    points = _bary_to_cart(best_bary)
    return points
# EVOLVE-BLOCK-END
