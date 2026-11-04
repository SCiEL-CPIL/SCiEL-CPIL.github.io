"""Heilbronn triangles: maximize the smallest triangle area among 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds to reserve

# Vertices of the equilateral triangle in Cartesian
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0  # height

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

# Precompute the 3x3 matrix for determinant calculation
# For points p_i, p_j, p_k in barycentric coords (each sums to 1),
# the area is proportional to |det([p_i, p_j, p_k])| where each p is a 3-vector.
# Actually, the area of triangle with barycentric coords u_i, u_j, u_k (each (a,b,c) summing to 1)
# is |det of matrix with rows u_i, u_j, u_k| * (area of reference triangle)
# Since our reference triangle has area sqrt(3)/4, and we divide by it, the score is just |det|.


def _bary_to_cart(u, v):
    """Convert barycentric coords (u, v) with w=1-u-v to Cartesian."""
    w = 1.0 - u - v
    x = 0.5 * v + 0.5 * w  # 0*u + 1*v + 0.5*w
    y = H * w              # 0*u + 0*v + H*w
    return np.array([x, y])


def _cart_to_bary(x, y):
    """Convert Cartesian to barycentric (u, v) with w=1-u-v."""
    w = y / H
    v = (x - 0.5 * w) / 0.5  # x = 0.5*v + 0.5*w => v = 2*(x - 0.5*w)
    u = 1.0 - v - w
    return u, v


def _clamp_bary(u, v):
    """Clamp barycentric coords (u, v) to the simplex u>=0, v>=0, u+v<=1."""
    w = 1.0 - u - v
    # Project to simplex
    if u < 0:
        u = 0
    if v < 0:
        v = 0
    if u + v > 1:
        s = u + v
        u /= s
        v /= s
    w = 1.0 - u - v
    return u, v


def _all_dets_from_bary(bary_points):
    """
    Compute all |det| values for triples of barycentric points.
    bary_points: (N, 3) array where each row sums to 1.
    Returns array of shape (N_TRIPLES,) with |det| values.
    """
    dets = np.zeros(N_TRIPLES)
    for idx, (i, j, k) in enumerate(_IDX_TRIPLES):
        # det of 3x3 matrix with rows bary[i], bary[j], bary[k]
        ui, vi, wi = bary_points[i]
        uj, vj, wj = bary_points[j]
        uk, vk, wk = bary_points[k]
        det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
        dets[idx] = abs(det)
    return dets


def _score_bary(bary_points):
    """Return the minimum |det| (proportional to min area)."""
    dets = _all_dets_from_bary(bary_points)
    return np.min(dets)


def _softmin_objective(x, beta):
    """
    Smooth approximation of the negative min determinant.
    x: (2N,) array of (u_i, v_i) for i=0..N-1
    Returns -soft_min(|det|)
    """
    bary = np.zeros((N, 3))
    for i in range(N):
        u = x[2*i]
        v = x[2*i+1]
        w = 1.0 - u - v
        # Clamp to simplex
        if u < 0: u = 0
        if v < 0: v = 0
        if u + v > 1:
            s = u + v
            u /= s
            v /= s
        w = 1.0 - u - v
        bary[i] = [u, v, w]
    
    dets = _all_dets_from_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * dets)))
    # Use log-sum-exp trick for stability
    neg_beta_dets = -beta * dets
    max_val = np.max(neg_beta_dets)
    if max_val > 700:  # avoid overflow
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_dets - max_val)))
    else:
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_dets - max_val)))
    soft_min = -log_sum_exp / beta
    
    return -soft_min


def _softmin_gradient(x, beta):
    """
    Gradient of the softmin objective with respect to x.
    This is complex, so we'll use numerical differentiation or skip gradient for SLSQP.
    """
    # For simplicity, we'll let SLSQP use numerical gradients
    return None


def _hard_objective(x, t):
    """
    Epigraph formulation: minimize -t subject to all dets >= t.
    x: (2N,) barycentric coords
    t: scalar lower bound on all determinants
    We'll use a penalty approach instead for simplicity.
    """
    bary = np.zeros((N, 3))
    for i in range(N):
        u = x[2*i]
        v = x[2*i+1]
        w = 1.0 - u - v
        if u < 0: u = 0
        if v < 0: v = 0
        if u + v > 1:
            s = u + v
            u /= s
            v /= s
        w = 1.0 - u - v
        bary[i] = [u, v, w]
    
    dets = _all_dets_from_bary(bary)
    min_det = np.min(dets)
    return -min_det


def _structured_init_bary():
    """Generate a structured initial configuration in barycentric coords (u, v)."""
    points = []
    # Center
    points.append([1/3, 1/3])
    # Near vertices
    eps = 0.08
    points.append([1-2*eps, eps])
    points.append([eps, 1-2*eps])
    points.append([eps, eps])
    # Midpoints of edges
    points.append([0.5, 0.5])
    points.append([0.5, 0.0])
    points.append([0.0, 0.5])
    # Additional points
    rng = np.random.RandomState(42)
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            points.append([u, v])
        else:
            points.append([0.35, 0.35])
    
    return np.array(points[:N])


def _random_init_bary(rng):
    """Generate random initial configuration in barycentric coords."""
    points = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            s = u + v
            u /= s
            v /= s
        points.append([u, v])
    return np.array(points)


def _fps_init_bary():
    """
    Farthest Point Sampling initialization.
    Start with vertices and center, then add points to maximize minimum distance.
    """
    points = []
    # Start with 3 vertices and center
    points.append([0.95, 0.05])  # near A
    points.append([0.05, 0.95])  # near B
    points.append([0.05, 0.05])  # near C
    points.append([1/3, 1/3])    # center
    
    for _ in range(N - 4):
        best_point = None
        best_min_dist = -1
        # Try many random candidates
        rng = np.random.RandomState(int(np.random.randint(0, 1000000)))
        for _ in range(5000):
            u = rng.random()
            v = rng.random()
            if u + v > 1.0:
                s = u + v
                u /= s
                v /= s
            cand = np.array([u, v])
            # Convert to Cartesian for distance
            cand_cart = _bary_to_cart(u, v)
            min_dist = float('inf')
            for p in points:
                p_cart = _bary_to_cart(p[0], p[1])
                dist = np.linalg.norm(cand_cart - p_cart)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_point = cand
        points.append(best_point)
    
    return np.array(points)


def _local_optimize_slsqp(bary_points, beta=100.0, max_iter=200, time_limit=30.0):
    """Optimize using SLSQP on barycentric coords (u, v)."""
    start_time = time.time()
    x0 = np.zeros(2 * N)
    for i in range(N):
        x0[2*i] = bary_points[i, 0]
        x0[2*i+1] = bary_points[i, 1]
    
    bounds = [(0.0, 1.0)] * (2 * N)
    cons = []
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]
        })
    
    def objective_wrapper(x):
        return _softmin_objective(x, beta)
    
    result = minimize(
        objective_wrapper,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )
    
    bary_opt = np.zeros((N, 2))
    for i in range(N):
        u = result.x[2*i]
        v = result.x[2*i+1]
        bary_opt[i] = [u, v]
    
    return bary_opt


def _simulated_annealing(bary_points, start_temp=0.03, end_temp=0.001, steps=5000, rng=None, time_limit=60.0):
    """
    Simulated annealing to explore the landscape.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    start_time = time.time()
    current_bary = bary_points.copy()
    current_score = _score_bary(current_bary)
    best_bary = current_bary.copy()
    best_score = current_score
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        idx = rng.randint(N)
        new_bary = current_bary.copy()
        noise_u = rng.normal(0, temp)
        noise_v = rng.normal(0, temp)
        new_u = new_bary[idx, 0] + noise_u
        new_v = new_bary[idx, 1] + noise_v
        new_bary[idx] = _clamp_bary(new_u, new_v)
        
        new_score = _score_bary(new_bary)
        
        if new_score > current_score:
            current_bary = new_bary
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
        else:
            delta = new_score - current_score
            if delta / temp > -50:  # avoid underflow
                prob = np.exp(delta / temp)
                if rng.random() < prob:
                    current_bary = new_bary
                    current_score = new_score
    
    return best_bary


def _l_bfgs_softmin(bary_points, beta=200.0, max_iter=500, time_limit=30.0):
    """
    Use L-BFGS-B with softmin objective. No explicit constraints, rely on clamping.
    """
    start_time = time.time()
    x0 = np.zeros(2 * N)
    for i in range(N):
        x0[2*i] = bary_points[i, 0]
        x0[2*i+1] = bary_points[i, 1]
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def objective_wrapper(x):
        return _softmin_objective(x, beta)
    
    result = minimize(
        objective_wrapper,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    bary_opt = np.zeros((N, 2))
    for i in range(N):
        u = result.x[2*i]
        v = result.x[2*i+1]
        bary_opt[i] = _clamp_bary(u, v)
    
    return bary_opt


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_structured_init_bary())
    inits.append(_fps_init_bary())
    
    rng = np.random.RandomState(123)
    for _ in range(5):
        inits.append(_random_init_bary(rng))
    
    best_bary = None
    best_score = -float('inf')
    
    for init_bary in inits:
        if time.time() >= deadline:
            break
        
        current_bary = init_bary.copy()
        current_score = _score_bary(current_bary)
        
        # Local optimization with SLSQP
        if time.time() < deadline:
            try:
                opt_bary = _local_optimize_slsqp(current_bary, beta=100.0, max_iter=150, time_limit=20.0)
                score = _score_bary(opt_bary)
                if score > current_score:
                    current_bary = opt_bary
                    current_score = score
            except Exception:
                pass
        
        # L-BFGS-B refinement
        if time.time() < deadline:
            try:
                opt_bary = _l_bfgs_softmin(current_bary, beta=300.0, max_iter=300, time_limit=25.0)
                score = _score_bary(opt_bary)
                if score > current_score:
                    current_bary = opt_bary
                    current_score = score
            except Exception:
                pass
        
        # Simulated annealing
        if time.time() < deadline:
            try:
                sa_bary = _simulated_annealing(current_bary, start_temp=0.02, end_temp=0.0005, steps=3000, rng=rng, time_limit=30.0)
                score = _score_bary(sa_bary)
                if score > current_score:
                    current_bary = sa_bary
                    current_score = score
            except Exception:
                pass
        
        # Final SLSQP polish
        if time.time() < deadline:
            try:
                opt_bary = _local_optimize_slsqp(current_bary, beta=500.0, max_iter=200, time_limit=20.0)
                score = _score_bary(opt_bary)
                if score > current_score:
                    current_bary = opt_bary
                    current_score = score
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_bary = current_bary.copy()
    
    # If no valid result, use structured init
    if best_bary is None:
        best_bary = _structured_init_bary()
    
    # Convert best barycentric points to Cartesian
    cart_points = np.array([_bary_to_cart(best_bary[i, 0], best_bary[i, 1]) for i in range(N)])
    
    # Verify points are inside the triangle
    for i in range(N):
        x, y = cart_points[i]
        u, v = _cart_to_bary(x, y)
        if u < -1e-9 or v < -1e-9 or u + v > 1.0 + 1e-9:
            # Project to triangle
            u_clamped, v_clamped = _clamp_bary(u, v)
            cart_points[i] = _bary_to_cart(u_clamped, v_clamped)
    
    return cart_points
