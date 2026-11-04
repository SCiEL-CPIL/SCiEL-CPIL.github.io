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
    verts = np.array([V0, V1, V2])  # (3, 2)
    return bary @ verts  # (N, 2)

def _all_dets_from_bary(bary):
    """Compute all 165 triangle area ratios from barycentric coordinates.
    
    For points with bary coords (a,b,c), the area ratio of triangle (i,j,k)
    is |det([[1,a_i,b_i,c_i],[1,a_j,b_j,c_j],[1,a_k,b_k,c_k]])| / 2.
    Since a+b+c=1, this simplifies to |det([[a_i,b_i,c_i],[a_j,b_j,c_j],[a_k,b_k,c_k]])| / 2.
    """
    n = len(bary)
    dets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                b1, b2, b3 = bary[i], bary[j], bary[k]
                # 3x3 determinant
                det = (b1[0]*(b2[1]*b3[2] - b2[2]*b3[1]) -
                       b1[1]*(b2[0]*b3[2] - b2[2]*b3[0]) +
                       b1[2]*(b2[0]*b3[1] - b2[1]*b3[0]))
                dets.append(abs(det) / 2.0)
    return np.array(dets)

def _min_area_ratio_from_bary(bary):
    """Compute min triangle area / TRI_AREA from barycentric coordinates (N,3)."""
    dets = _all_dets_from_bary(bary)
    return float(dets.min())

def _smooth_min(dets, beta=50.0):
    """Smooth minimum using log-sum-exp: min(x) ≈ -log(sum(exp(-beta*x)))/beta."""
    # For numerical stability
    d = dets - dets.min()
    return dets.min() - np.log(np.sum(np.exp(-beta * d))) / beta

def _objective_smooth(x, beta=50.0):
    """Smooth objective: negative smooth min area ratio. x is (N*3,) flattened bary."""
    bary = x.reshape(N, 3)
    # Project to simplex
    bary = np.maximum(bary, 0)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-15, 1.0, sums)
    bary = bary / sums
    dets = _all_dets_from_bary(bary)
    sm = _smooth_min(dets, beta)
    return -sm

def _objective_grad(x, beta=50.0):
    """Objective with gradient for L-BFGS-B."""
    bary = x.reshape(N, 3).copy()
    bary = np.maximum(bary, 0)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-15, 1.0, sums)
    bary = bary / sums
    
    dets = _all_dets_from_bary(bary)
    sm = _smooth_min(dets, beta)
    
    # Gradient of smooth min w.r.t. dets
    d = dets - dets.min()
    weights = np.exp(-beta * d) / np.sum(np.exp(-beta * d))
    
    # Gradient of dets w.r.t. bary coordinates
    # det(i,j,k) = 0.5 * |det of 3x3|, but we track signed det for gradient
    # Actually, let's compute gradient numerically for reliability
    grad = np.zeros_like(x)
    eps = 1e-6
    for idx in range(len(x)):
        x_plus = x.copy()
        x_plus[idx] += eps
        b_plus = x_plus.reshape(N, 3)
        b_plus = np.maximum(b_plus, 0)
        s_plus = b_plus.sum(axis=1, keepdims=True)
        s_plus = np.where(s_plus < 1e-15, 1.0, s_plus)
        b_plus = b_plus / s_plus
        dets_plus = _all_dets_from_bary(b_plus)
        sm_plus = _smooth_min(dets_plus, beta)
        grad[idx] = (sm_plus - sm) / eps
    
    return -sm, -grad

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
    bary = np.zeros((N, 3))
    bary[0] = [1.0, 0.0, 0.0]
    bary[1] = [0.0, 1.0, 0.0]
    bary[2] = [0.0, 0.0, 1.0]
    bary[3] = [1.0/3, 1.0/3, 1.0/3]
    
    for i in range(4, N):
        candidates = rng.dirichlet([1.0, 1.0, 1.0], size=300)
        best_idx = -1
        best_min_dist = -1
        for c in candidates:
            dists = np.sqrt(((bary[:i] - c) ** 2).sum(axis=1))
            min_d = dists.min()
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = c
        bary[i] = best_idx
    
    return bary

def _symmetric_init():
    """Create a symmetric initialization exploiting triangle symmetry."""
    bary = np.zeros((N, 3))
    # Vertices
    bary[0] = [1.0, 0.0, 0.0]
    bary[1] = [0.0, 1.0, 0.0]
    bary[2] = [0.0, 0.0, 1.0]
    # Centroid
    bary[3] = [1/3, 1/3, 1/3]
    # Midpoints of edges
    bary[4] = [0.5, 0.5, 0.0]
    bary[5] = [0.5, 0.0, 0.5]
    bary[6] = [0.0, 0.5, 0.5]
    # Three points at 1/4 distance from vertices toward centroid (symmetric)
    r = 0.25
    bary[7] = [r + (1-r)/3, (1-r)/3, (1-r)/3]
    bary[8] = [(1-r)/3, r + (1-r)/3, (1-r)/3]
    bary[9] = [(1-r)/3, (1-r)/3, r + (1-r)/3]
    # One more symmetric point
    bary[10] = [1/3, 1/3, 1/3]  # Will be optimized
    # Actually let's use a different 11th point
    bary[10] = [0.4, 0.3, 0.3]
    return bary

def _local_optimize_lbfgs_smooth(bary_init, beta=50.0, maxiter=200):
    """Optimize using L-BFGS-B with smooth objective."""
    x0 = bary_init.flatten()
    bounds = [(0.0, 1.0)] * (N * 3)
    
    try:
        res = minimize(
            _objective_smooth,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        bary_opt = _project_to_bary_simplex(res.x)
        score = _min_area_ratio_from_bary(bary_opt)
        return bary_opt, score
    except Exception:
        return bary_init, _min_area_ratio_from_bary(bary_init)

def _local_optimize_nm(bary_init, maxiter=1000):
    """Nelder-Mead optimization on projected barycentric coordinates."""
    x0 = bary_init.flatten()
    
    def obj(x):
        bary = x.reshape(N, 3)
        bary = np.maximum(bary, 0)
        sums = bary.sum(axis=1, keepdims=True)
        sums = np.where(sums < 1e-15, 1.0, sums)
        bary = bary / sums
        return -_min_area_ratio_from_bary(bary)
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        bary_opt = _project_to_bary_simplex(res.x)
        score = _min_area_ratio_from_bary(bary_opt)
        return bary_opt, score
    except Exception:
        return bary_init, _min_area_ratio_from_bary(bary_init)

def _local_optimize_slsqp(bary_init, maxiter=500):
    """SLSQP with smooth objective."""
    x0 = bary_init.flatten()
    
    eq_cons = []
    for i in range(N):
        eq_cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: x[3*i] + x[3*i+1] + x[3*i+2] - 1.0
        })
    
    ineq_cons = []
    for i in range(N):
        for j in range(3):
            ineq_cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: x[3*i + j]
            })
    
    try:
        res = minimize(
            _objective_smooth,
            x0,
            method='SLSQP',
            constraints=eq_cons + ineq_cons,
            options={'maxiter': maxiter, 'ftol': 1e-10}
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
    
    # Symmetric initialization
    inits.append(_symmetric_init())
    
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
    for seed in [1, 7, 13, 21, 99, 123]:
        rng = np.random.RandomState(seed)
        bary = rng.dirichlet([1.0, 1.0, 1.0], size=N)
        inits.append(bary)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    best_bary = None
    best_score = -float('inf')
    
    # Generate diverse initializations
    inits = _generate_diverse_inits()
    
    # Phase 1: L-BFGS-B with smooth objective (fast, good gradients)
    for idx, bary_init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        bary_opt, score = _local_optimize_lbfgs_smooth(bary_init, beta=50.0, maxiter=150)
        if score > best_score:
            best_score = score
            best_bary = bary_opt
    
    # Phase 2: Polish best with higher beta (sharper min approximation)
    if best_bary is not None and time.time() < deadline - 20.0:
        bary_polished, score2 = _local_optimize_lbfgs_smooth(best_bary, beta=100.0, maxiter=200)
        if score2 > best_score:
            best_score = score2
            best_bary = bary_polished
    
    # Phase 3: SLSQP for additional polishing
    if best_bary is not None and time.time() < deadline - 15.0:
        bary_slsqp, score3 = _local_optimize_slsqp(best_bary, maxiter=300)
        if score3 > best_score:
            best_score = score3
            best_bary = bary_slsqp
    
    # Phase 4: Nelder-Mead for non-gradient refinement
    if best_bary is not None and time.time() < deadline - 10.0:
        bary_nm, score4 = _local_optimize_nm(best_bary, maxiter=500)
        if score4 > best_score:
            best_score = score4
            best_bary = bary_nm
    
    # Phase 5: More random starts with L-BFGS-B if time remains
    if time.time() < deadline - 25.0:
        for seed in [100, 200, 300, 400, 500, 600, 700, 800]:
            if time.time() > deadline - 15.0:
                break
            rng = np.random.RandomState(seed)
            bary_rand = rng.dirichlet([1.5, 1.5, 1.5], size=N)
            bary_opt, score = _local_optimize_lbfgs_smooth(bary_rand, beta=50.0, maxiter=100)
            if score > best_score:
                best_score = score
                best_bary = bary_opt
    
    # Final polish with very high beta
    if best_bary is not None and time.time() < deadline - 5.0:
        bary_final, score_final = _local_optimize_lbfgs_smooth(best_bary, beta=200.0, maxiter=100)
        if score_final > best_score:
            best_score = score_final
            best_bary = bary_final
    
    # Convert best barycentric to Cartesian
    if best_bary is None:
        # Fallback
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
    
    # Final validation: ensure all points are inside the triangle
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
