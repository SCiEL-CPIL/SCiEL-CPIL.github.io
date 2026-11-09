"""Heilbronn triangles: maximize smallest triangle area for 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = np.sqrt(3.0)
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])
AREA_EQ = SQRT3 / 4.0  # area of the equilateral triangle

# All pairs of indices
PAIRS = []
for i in range(N):
    for j in range(i + 1, N):
        PAIRS.append((i, j))
NP = len(PAIRS)  # 55 pairs

# All triples of indices
TRIPLES = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRIPLES.append((i, j, k))
NT = len(TRIPLES)  # 165 triples


def _bary_to_cart(bary):
    """Convert barycentric coordinates (sum=1, each>=0) to Cartesian."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2.0) * bary[:, 2]
    return np.column_stack([x, y])


def _cart_to_bary(cart):
    """Convert Cartesian to barycentric coordinates."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / (SQRT3 / 2.0)
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _triangle_areas_from_cart(pts):
    """Compute areas of all 165 triangles from N Cartesian points."""
    areas = np.empty(NT)
    for idx, (i, j, k) in enumerate(TRIPLES):
        ax, ay = pts[i]
        bx, by = pts[j]
        cx, cy = pts[k]
        areas[idx] = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    return areas


def _min_area_cart(pts):
    """Minimum triangle area for a set of Cartesian points."""
    areas = _triangle_areas_from_cart(pts)
    return np.min(areas)


def _soft_min_loss(params, beta):
    """
    Smooth surrogate: -soft_min of triangle areas.
    params: 2N array (x0,y0,...,xN,yN) Cartesian coordinates.
    Returns loss (negative soft-min area) and gradient.
    """
    pts = params.reshape(N, 2)
    areas = _triangle_areas_from_cart(pts)
    
    # Soft minimum via log-sum-exp: soft_min(a) = -logsumexp(-a/beta) * beta
    # For stability, subtract max
    neg_areas = -areas / beta
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min_val = -beta * log_sum
    loss = -soft_min_val
    
    # Gradient: d(-soft_min)/d(params)
    # soft_min = -beta * logsumexp(-areas/beta)
    # d(soft_min)/d(area_i) = -beta * (1/beta) * exp(-area_i/beta) / sum(exp(-area_j/beta))
    #                        = -exp(-area_i/beta) / sum(exp(-area_j/beta))
    # d(loss)/d(area_i) = -d(soft_min)/d(area_i) = exp(-area_i/beta) / sum(exp(-area_j/beta))
    
    weights = np.exp(neg_areas - max_neg)
    weights /= np.sum(weights)
    
    # Gradient of each area w.r.t. each point
    # Area(i,j,k) = 0.5 * |cross(Pj-Pi, Pk-Pi)|
    # For simplicity and differentiability, use signed area without abs
    # Actually, let's compute gradient of the raw (signed) cross product
    # Since we want to maximize areas, and areas are positive in our domain,
    # we can work with the signed area (which equals the unsigned for valid configs)
    
    grad = np.zeros(2 * N)
    for idx, (i, j, k) in enumerate(TRIPLES):
        w = weights[idx]
        if w < 1e-15:
            continue
        ax, ay = pts[i]
        bx, by = pts[j]
        cx, cy = pts[k]
        # Signed area = 0.5 * ((bx-ax)(cy-ay) - (cx-ax)(by-ay))
        # d(area)/d(ax) = 0.5 * (-(cy-ay) + (by-ay)) = 0.5 * (by - cy)
        # d(area)/d(ay) = 0.5 * (-(bx-ax) + (cx-ax)) = 0.5 * (cx - bx)
        # d(area)/d(bx) = 0.5 * (cy - ay)
        # d(area)/d(by) = 0.5 * (-(cy-ay) + ... ) let me redo
        # area = 0.5*(bx*cy - bx*ay - ax*cy + ax*ay - cx*by + cx*ay + ax*by - ax*ay)
        #      = 0.5*(bx*cy - bx*ay - ax*cy + cx*ay - cx*by + ax*by)
        # d/d(ax) = 0.5*(-cy + by) = 0.5*(by - cy)
        # d/d(ay) = 0.5*(-bx + cx) = 0.5*(cx - bx)
        # d/d(bx) = 0.5*(cy - ay)
        # d/d(by) = 0.5*(-cx + ax) = 0.5*(ax - cx)
        # d/d(cx) = 0.5*(ay - by)
        # d/d(cy) = 0.5*(bx - ax)
        
        grad[2*i]   += w * 0.5 * (by - cy)
        grad[2*i+1] += w * 0.5 * (cx - bx)
        grad[2*j]   += w * 0.5 * (cy - ay)
        grad[2*j+1] += w * 0.5 * (ax - cx)
        grad[2*k]   += w * 0.5 * (ay - by)
        grad[2*k+1] += w * 0.5 * (bx - ax)
    
    return loss, grad


def _check_inside(pts):
    """Check if all points are inside or on the equilateral triangle."""
    # Barycentric coordinates must all be >= 0
    bary = _cart_to_bary(pts)
    return np.all(bary >= -1e-9)


def _penalty(pts):
    """Penalty for points outside the triangle."""
    bary = _cart_to_bary(pts)
    # Penalty for negative barycentric coords
    neg = np.clip(-bary, 0, None)
    return np.sum(neg ** 2)


def _objective(params, beta, penalty_weight=100.0):
    """Combined objective: soft-min loss + boundary penalty."""
    pts = params.reshape(N, 2)
    loss, grad = _soft_min_loss(params, beta)
    pen = _penalty(pts)
    total_loss = loss + penalty_weight * pen
    
    # Gradient of penalty
    # penalty = sum of (max(0, -b_i))^2 where b = barycentric
    # db/dx for each point...
    # b0 = 1 - x + x/2 - y/(sqrt3/2)  => db0/dx = -0.5, db0/dy = -1/(sqrt3/2)
    # b1 = x - x/2 - y/(sqrt3/2)      => db1/dx = 0.5,  db1/dy = -1/(sqrt3/2)
    # b2 = y/(sqrt3/2)                => db2/dx = 0,     db2/dy = 1/(sqrt3/2)
    # 
    # d(pen)/dx = 2*penalty_weight * sum over points of sum over b of clip(-b,0)*(-db/dx)
    
    inv_sqrt3_2 = 2.0 / SQRT3
    pen_grad = np.zeros(2 * N)
    for i in range(N):
        b = bary[i]
        for bi in range(3):
            if b[bi] < 0:
                # d/dx of (-b_bi)^2 = 2*(-b_bi)*(-db_bi/dx)
                if bi == 0:
                    db_dx = -0.5
                    db_dy = -inv_sqrt3_2
                elif bi == 1:
                    db_dx = 0.5
                    db_dy = -inv_sqrt3_2
                else:
                    db_dx = 0.0
                    db_dy = inv_sqrt3_2
                pen_grad[2*i]   += penalty_weight * 2.0 * (-b[bi]) * (-db_dx)
                pen_grad[2*i+1] += penalty_weight * 2.0 * (-b[bi]) * (-db_dy)
    
    return total_loss, grad + pen_grad


def _project_to_triangle(pts):
    """Project points to be inside the triangle using barycentric clipping."""
    bary = _cart_to_bary(pts)
    bary = np.clip(bary, 0, None)
    bary /= bary.sum(axis=1, keepdims=True)
    return _bary_to_cart(bary)


def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    
    # Phase 1: Differential Evolution for global exploration
    # Use barycentric coordinates as parameters to naturally satisfy constraints
    # params: 3N barycentric coords, but they must sum to 1 per point
    # Instead, use 2N free params per point (first 2 barycentric coords), 3rd = 1 - sum
    # Or just use Cartesian with bounds
    
    # Bounds for Cartesian: x in [0,1], y in [0, sqrt(3)/2]
    # But triangle constraint: y <= sqrt(3)*(1-x) for x>=0.5, y <= sqrt(3)*x for x<=0.5
    # Simpler: use barycentric with 2 free params per point
    
    def _bary_params_to_pts(bp):
        """Convert 2N barycentric params to N points."""
        b = np.zeros((N, 3))
        b[:, 0] = 1.0
        b[:, 1] = bp[:N]
        b[:, 2] = bp[N:]
        # Normalize to ensure they sum to 1 and are non-negative
        b = np.clip(b, 0, None)
        s = b.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        b /= s
        return _bary_to_cart(b)
    
    def _de_objective(bp):
        pts = _bary_params_to_pts(bp)
        return -_min_area_cart(pts)
    
    # DE bounds: each barycentric coord in [0, 1]
    de_bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run DE with limited iterations
    try:
        de_result = differential_evolution(
            _de_objective,
            de_bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 0.9),
            recombination=0.8,
            polish=False,
            init='latinhypercube',
            workers=1
        )
        best_pts_de = _bary_params_to_pts(de_result.x)
        best_score = _min_area_cart(best_pts_de)
        best_pts = best_pts_de
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_pts if best_pts is not None else _bary_params_to_pts(np.random.rand(2*N))
    
    # Phase 2: L-BFGS-B refinement from multiple diverse initializations
    # Generate diverse starting points
    rng = np.random.RandomState(123)
    
    initializations = []
    
    # Add the DE result
    if best_pts is not None:
        initializations.append(best_pts.copy())
    
    # Add grid-based initializations
    for seed in range(5):
        pts = _bary_params_to_pts(rng.rand(2 * N) * 0.8 + 0.1)
        initializations.append(pts)
    
    # Add structured initializations (vertices, midpoints, etc.)
    # Place points at vertices, edge midpoints, and centroid
    structured = []
    structured.append(V0.copy())
    structured.append(V1.copy())
    structured.append(V2.copy())
    structured.append((V0 + V1) / 2.0)
    structured.append((V1 + V2) / 2.0)
    structured.append((V2 + V0) / 2.0)
    centroid = (V0 + V1 + V2) / 3.0
    structured.append(centroid)
    # Fill remaining with slight perturbations
    for i in range(N - len(structured)):
        pt = centroid + rng.normal(0, 0.1, 2)
        pt = _project_to_triangle(pt.reshape(1, 2))[0]
        structured.append(pt)
    initializations.append(np.array(structured))
    
    # Add more random
    for seed in range(3):
        pts = _bary_params_to_pts(rng.rand(2 * N) * 0.9 + 0.05)
        initializations.append(pts)
    
    # Refine each initialization with L-BFGS-B using soft-min objective
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - start_time > deadline:
            break
        
        params = init_pts.reshape(-1).copy()
        beta = 0.05  # soft-min temperature
        
        try:
            result = minimize(
                _objective,
                params,
                args=(beta, 50.0),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            refined_pts = _project_to_triangle(result.x.reshape(N, 2))
            score = _min_area_cart(refined_pts)
            
            if score > best_score:
                best_score = score
                best_pts = refined_pts
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with tighter tolerance and lower beta
    if best_pts is not None and time.time() - start_time < deadline - 5:
        params = best_pts.reshape(-1).copy()
        beta = 0.01  # lower temperature for sharper minimum
        
        try:
            result = minimize(
                _objective,
                params,
                args=(beta, 100.0),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            polished_pts = _project_to_triangle(result.x.reshape(N, 2))
            score = _min_area_cart(polished_pts)
            
            if score > best_score:
                best_score = score
                best_pts = polished_pts
        except Exception:
            pass
    
    # Ensure we return valid points
    if best_pts is None:
        best_pts = _bary_params_to_pts(np.random.rand(2 * N) * 0.5 + 0.25)
    
    return best_pts
