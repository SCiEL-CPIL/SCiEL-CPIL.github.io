"""Heilbronn triangles: maximize the smallest triangle area among 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 15.0  # seconds to reserve

# Vertices of the equilateral triangle in Cartesian
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0  # height

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)


def _barycentric_coords(pt):
    """Return barycentric coordinates (a, b, c) with a+b+c=1 for point pt in triangle ABC."""
    x, y = pt
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.array([a, b, c])


def _inside_triangle(pt, tol=1e-9):
    """Check if point is inside or on the boundary of the equilateral triangle."""
    bc = _barycentric_coords(pt)
    return np.all(bc >= -tol) and np.all(bc <= 1.0 + tol)


def _project_to_triangle(pt):
    """Project point to the triangle if outside, using barycentric clamping."""
    bc = _barycentric_coords(pt)
    bc_clamped = np.clip(bc, 0.0, 1.0)
    # Normalize to sum to 1
    s = np.sum(bc_clamped)
    if s < 1e-12:
        return A.copy()
    bc_clamped /= s
    return bc_clamped[0] * A + bc_clamped[1] * B + bc_clamped[2] * C


def _bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v, w=1-u-v) to Cartesian."""
    w = 1.0 - u - v
    x = 0.5 - 0.5 * u + 0.5 * v
    y = H * w
    return np.array([x, y])


def _project_to_bary_simplex(u, v, w):
    """Project (u,v,w) to the simplex u+v+w=1, u,v,w >= 0."""
    s = u + v + w
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    u, v, w = u/s, v/s, w/s
    
    if u < 0: u = 0
    if v < 0: v = 0
    if w < 0: w = 0
    
    s = u + v + w
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    u, v, w = u/s, v/s, w/s
    return np.array([u, v, w])


def _min_det_from_bary(bary_points):
    """
    Compute the minimum absolute determinant of triples of barycentric points.
    This is proportional to the triangle area.
    """
    min_det = float('inf')
    for i, j, k in _IDX_TRIPLES:
        ui, vi, wi = bary_points[i]
        uj, vj, wj = bary_points[j]
        uk, vk, wk = bary_points[k]
        
        det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
        abs_det = abs(det)
        if abs_det < min_det:
            min_det = abs_det
    return min_det


def _score_bary(bary_points):
    """Return the minimum area (scaled)."""
    return _min_det_from_bary(bary_points)


def _objective_smooth(x, beta):
    """
    Smooth objective for optimization.
    x is flattened barycentric coordinates (2N values: u, v for each point).
    """
    bary = np.zeros((N, 3))
    for i in range(N):
        u = x[2*i]
        v = x[2*i+1]
        w = 1.0 - u - v
        bary[i] = _project_to_bary_simplex(u, v, w)
    
    dets = np.zeros(N_TRIPLES)
    for idx, (i, j, k) in enumerate(_IDX_TRIPLES):
        ui, vi, wi = bary[i]
        uj, vj, wj = bary[j]
        uk, vk, wk = bary[k]
        det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
        dets[idx] = abs(det)
    
    neg_beta_dets = -beta * dets
    max_val = np.max(neg_beta_dets)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_dets - max_val)))
    soft_min = -log_sum_exp / beta
    
    return -soft_min


def _local_optimize_slsqp(bary_points, beta=50.0, max_iter=100):
    """Optimize using SLSQP on barycentric coords (u, v)."""
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
    
    result = minimize(
        _objective_smooth,
        x0,
        args=(beta,),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-9}
    )
    
    bary_opt = np.zeros((N, 3))
    for i in range(N):
        u = result.x[2*i]
        v = result.x[2*i+1]
        w = 1.0 - u - v
        bary_opt[i] = _project_to_bary_simplex(u, v, w)
    
    return bary_opt


def _simulated_annealing(bary_points, start_temp=0.05, end_temp=0.001, steps=2000, rng=None):
    """
    Simulated annealing to explore the landscape.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_bary = bary_points.copy()
    current_score = _score_bary(current_bary)
    best_bary = current_bary.copy()
    best_score = current_score
    
    for step in range(steps):
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        idx = rng.randint(N)
        new_bary = current_bary.copy()
        noise_u = rng.normal(0, temp)
        noise_v = rng.normal(0, temp)
        new_bary[idx, 0] += noise_u
        new_bary[idx, 1] += noise_v
        new_bary[idx] = _project_to_bary_simplex(new_bary[idx, 0], new_bary[idx, 1], 1.0 - new_bary[idx, 0] - new_bary[idx, 1])
        
        new_score = _score_bary(new_bary)
        
        if new_score > current_score:
            current_bary = new_bary
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
        else:
            delta = new_score - current_score
            prob = np.exp(delta / temp)
            if rng.random() < prob:
                current_bary = new_bary
                current_score = new_score
    
    return best_bary


def _generate_structured_init():
    """Generate a structured initial configuration in barycentric coords."""
    points = []
    points.append([1/3, 1/3, 1/3])
    eps = 0.05
    points.append([1-2*eps, eps, eps])
    points.append([eps, 1-2*eps, eps])
    points.append([eps, eps, 1-2*eps])
    points.append([0.5, 0.5, 0.0])
    points.append([0.5, 0.0, 0.5])
    points.append([0.0, 0.5, 0.5])
    
    rng = np.random.RandomState(1)
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            w = 1.0 - u - v
            points.append([u, v, w])
        else:
            points.append([0.3, 0.3, 0.4])
    
    return np.array(points[:N])


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_bary = _generate_structured_init()
    best_score = _score_bary(best_bary)
    
    rng = np.random.RandomState(7)
    
    # Phase 1: Local optimization from structured start
    if time.time() < deadline:
        try:
            opt_bary = _local_optimize_slsqp(best_bary, beta=50.0, max_iter=200)
            score = _score_bary(opt_bary)
            if score > best_score:
                best_score = score
                best_bary = opt_bary
        except Exception:
            pass
    
    # Phase 2: Simulated Annealing to escape local optima
    if time.time() < deadline:
        try:
            sa_bary = _simulated_annealing(best_bary, start_temp=0.05, end_temp=0.001, steps=3000, rng=rng)
            score = _score_bary(sa_bary)
            if score > best_score:
                best_score = score
                best_bary = sa_bary
        except Exception:
            pass
    
    # Phase 3: Refine with local optimization again
    if time.time() < deadline:
        try:
            opt_bary = _local_optimize_slsqp(best_bary, beta=100.0, max_iter=300)
            score = _score_bary(opt_bary)
            if score > best_score:
                best_score = score
                best_bary = opt_bary
        except Exception:
            pass
    
    # Phase 4: More SA if time permits
    if time.time() < deadline:
        try:
            sa_bary = _simulated_annealing(best_bary, start_temp=0.02, end_temp=0.0005, steps=3000, rng=rng)
            score = _score_bary(sa_bary)
            if score > best_score:
                best_score = score
                best_bary = sa_bary
        except Exception:
            pass
            
    # Phase 5: Final Polish
    if time.time() < deadline:
        try:
            opt_bary = _local_optimize_slsqp(best_bary, beta=200.0, max_iter=500)
            score = _score_bary(opt_bary)
            if score > best_score:
                best_score = score
                best_bary = opt_bary
        except Exception:
            pass

    # Convert best barycentric points to Cartesian
    cart_points = np.array([_bary_to_cart(best_bary[i, 0], best_bary[i, 1]) for i in range(N)])
    
    # Verify points are inside
    for i in range(N):
        if not _inside_triangle(cart_points[i]):
             cart_points[i] = _project_to_triangle(cart_points[i])

    return cart_points
