"""Heilbronn triangles – barycentric + soft-min L-BFGS-B design."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_S3H = np.sqrt(3.0) / 2.0
_TIME_LIMIT = 120.0
_SOFT_BETA = 8.0  # temperature for soft-min (higher = smoother)


def _min_triangle_area(points):
    """Exact minimum area among all C(n,3) triangles. points: (n,2)."""
    n = points.shape[0]
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                a = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if a < min_a:
                    min_a = a
    return min_a


def _soft_min_area_flat(x, beta=_SOFT_BETA):
    """
    Smooth approximation of -min_area for L-BFGS-B.
    x: flat array of length 3*N (barycentric coords, not normalized internally
       we normalize to be safe).
    Returns (value, grad) where value = -soft_min_area + penalty.
    """
    n = N
    # Normalize barycentric coordinates
    B = x.reshape(n, 3)
    B = np.maximum(B, 1e-12)
    B = B / B.sum(axis=1, keepdims=True)
    
    # Convert to Cartesian: P = b0*V0 + b1*V1 + b2*V2
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, _S3H])
    P = B[:, 0:1] * V0 + B[:, 1:2] * V1 + B[:, 2:3] * V2
    
    # Compute all triangle areas
    areas = []
    idx_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = P[j, 0] - P[i, 0]
                dy1 = P[j, 1] - P[i, 1]
                dx2 = P[k, 0] - P[i, 0]
                dy2 = P[k, 1] - P[i, 1]
                a = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                areas.append(a)
                idx_pairs.append((i, j, k))
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft minimum: -log(sum(exp(-a/beta)))*beta
    # For numerical stability
    m = areas.min()
    soft_min = -beta * np.log(np.sum(np.exp(-(areas - m) / beta))) + m
    # Actually: soft_min = m - beta * log(sum(exp(-(a-m)/beta)))
    # Let me redo: soft_min(a) = -beta * log(sum(exp(-a/beta)))
    # = m - beta * log(sum(exp(-(a-m)/beta)))
    
    # Gradient of soft_min w.r.t. areas:
    # d(soft_min)/d(a_i) = exp(-a_i/beta) / sum(exp(-a_j/beta))
    # But we want d(-soft_min)/d(a_i) = -exp(-a_i/beta) / sum(exp(-a_j/beta))
    
    exp_neg = np.exp(-(areas - m) / beta)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # d(soft_min)/d(a_i)
    
    # We want to maximize soft_min, so minimize -soft_min
    # value = -soft_min + penalty
    # grad_value = -weights * d(areas)/d(x) + grad_penalty
    
    # Compute d(areas)/d(P) for each triangle
    # area = 0.5 * |cross|, but abs makes it non-differentiable at 0
    # Use signed area for gradient, then handle sign
    # Actually for small areas the gradient direction matters
    
    # Let's compute gradient of each area w.r.t. P
    # For triangle (i,j,k): area = 0.5 * |dx1*dy2 - dy1*dx2|
    # = 0.5 * |(Pj-Pi)x(Pk-Pi)|
    # = 0.5 * |Pj_x*Pk_y - Pj_y*Pk_x - Pi_x*Pk_y + Pi_y*Pk_x - Pk_x*Pi_y + Pk_y*Pi_x|
    # = 0.5 * |Pi*(Pj_y-Pk_y) + Pj*(Pk_y-Pi_y) + Pk*(Pi_y-Pj_y)|
    # Actually cross product: (Pj-Pi) x (Pk-Pi) = Pj_x*Pk_y - Pj_y*Pk_x - Pi_x*Pk_y + Pi_y*Pk_x + Pi_x*Pk_y - Pi_y*Pk_x... 
    
    # Let me use: area = 0.5 * |det([Pj-Pi, Pk-Pi])|
    # det = (Pj_x - Pi_x)*(Pk_y - Pi_y) - (Pj_y - Pi_y)*(Pk_x - Pi_x)
    # d(det)/d(Pi) = -(Pk_y - Pi_y), (Pk_x - Pi_x)
    # d(det)/d(Pj) = (Pk_y - Pi_y), -(Pk_x - Pi_x)
    # d(det)/d(Pk) = -(Pj_y - Pi_y), (Pj_x - Pi_x)
    
    # For signed area s = 0.5 * det, area = |s|
    # If s > 0: d(area)/d(P) = d(s)/d(P)
    # If s < 0: d(area)/d(P) = -d(s)/d(P)
    
    # To avoid the abs non-diff, I'll use a smooth approximation:
    # area_smooth = 0.5 * sqrt(det^2 + eps)
    # But this is getting complex. Let me just use finite differences for the gradient
    # of the soft-min objective.
    
    # Actually, let's just return the value and use a numerical gradient via
    # the optimizer. But L-BFGS-B can use analytical gradients.
    
    # For simplicity and correctness, let's compute the gradient numerically
    # within a helper, but that's slow. Instead, let me restructure.
    
    return -soft_min, None  # Will use numerical gradient


def _objective_and_grad(x, beta=_SOFT_BETA):
    """
    Objective function for L-BFGS-B.
    x: flat array of length 3*N (barycentric, unnormalized).
    Returns (value, grad).
    """
    n = N
    B = x.reshape(n, 3)
    B = np.maximum(B, 1e-12)
    B = B / B.sum(axis=1, keepdims=True)
    
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, _S3H])
    P = B[:, 0:1] * V0 + B[:, 1:2] * V1 + B[:, 2:3] * V2
    
    # All triangle areas
    areas = np.empty((n * (n-1) * (n-2)) // 6)
    idx = 0
    idx_list = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = P[j, 0] - P[i, 0]
                dy1 = P[j, 1] - P[i, 1]
                dx2 = P[k, 0] - P[i, 0]
                dy2 = P[k, 1] - P[i, 1]
                det = dx1 * dy2 - dy1 * dx2
                areas[idx] = 0.5 * abs(det)
                idx_list.append((i, j, k, det))
                idx += 1
    
    if len(areas) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft min
    m = areas.min()
    exp_neg = np.exp(-(areas - m) / beta)
    sum_exp = np.sum(exp_neg)
    soft_min = m - beta * np.log(sum_exp)
    
    # weights: d(soft_min)/d(area_i) = exp_neg[i] / sum_exp
    weights = exp_neg / sum_exp
    
    # Gradient of soft_min w.r.t. x:
    # d(soft_min)/dx = sum_i weights[i] * d(area_i)/dx
    # d(area_i)/dx = d(0.5*|det_i|)/dx
    # For det_i > 0: d(area_i)/d(P) = 0.5 * d(det_i)/d(P)
    # For det_i < 0: d(area_i)/d(P) = -0.5 * d(det_i)/d(P)
    
    grad_P = np.zeros_like(P)  # (n, 2)
    
    for w, (i, j, k, det) in zip(weights, idx_list):
        if abs(det) < 1e-15:
            continue
        sign = 1.0 if det > 0 else -1.0
        # d(det)/d(Pi) = (-(Pk_y - Pi_y), (Pk_x - Pi_x))
        # d(det)/d(Pj) = ((Pk_y - Pi_y), -(Pk_x - Pi_x))
        # d(det)/d(Pk) = (-(Pj_y - Pi_y), (Pj_x - Pi_x))
        
        dPi = np.array([-(P[k, 1] - P[i, 1]), (P[k, 0] - P[i, 0])])
        dPj = np.array([(P[k, 1] - P[i, 1]), -(P[k, 0] - P[i, 0])])
        dPk = np.array([-(P[j, 1] - P[i, 1]), (P[j, 0] - P[i, 0])])
        
        grad_P[i] += 0.5 * sign * w * dPi
        grad_P[j] += 0.5 * sign * w * dPj
        grad_P[k] += 0.5 * sign * w * dPk
    
    # Now convert d(soft_min)/d(P) to d(soft_min)/d(x)
    # P = B * V_matrix, where B is normalized barycentric
    # B = x / sum(x)
    # dP/db = V_matrix (each point's barycentric coords map to Cartesian)
    # db/dx: B_i = x_i / S where S = sum(x_i)
    # dB_i/dx_j = delta_ij/S - x_i/S^2
    
    # Let's compute d(soft_min)/d(x) = sum_i d(soft_min)/d(B_i) * dB_i/dx
    # d(soft_min)/d(B_i) = grad_P[i] dot [V1-V0, V2-V0] ... 
    # Actually P_i = B_i0*V0 + B_i1*V1 + B_i2*V2
    # dP_i/dB_i0 = V0, dP_i/dB_i1 = V1, dP_i/dB_i2 = V2
    # So d(soft_min)/dB_i0 = grad_P[i,0]*V0[0] + grad_P[i,1]*V0[1]
    # d(soft_min)/dB_i1 = grad_P[i,0]*V1[0] + grad_P[i,1]*V1[1]
    # d(soft_min)/dB_i2 = grad_P[i,0]*V2[0] + grad_P[i,1]*V2[1]
    
    dB_dx = np.zeros((n, 3, 3))  # dB_i/dx_j
    x_flat = x.reshape(n, 3)
    S = x_flat.sum(axis=1, keepdims=True)
    for i in range(n):
        for j in range(3):
            for k in range(3):
                dB_dx[i, j, k] = (1.0 if j == k else 0.0) / S[i, 0] - x_flat[i, j] / (S[i, 0] ** 2)
    
    # d(soft_min)/d(x_flat[i, j]) = sum_k d(soft_min)/dB_i_k * dB_i_k/dx_i_j
    # But since B_i is normalized, the constraint sum(B_i)=1 means we only need
    # to consider 2 independent directions. The gradient w.r.t. the normalized
    # coordinates projected back.
    
    # Simpler: compute d(soft_min)/d(B_i) for each i, then:
    # d(soft_min)/d(x_i) = d(soft_min)/d(B_i) * (1/S_i) - (d(soft_min)/d(B_i) dot B_i) / S_i^2 * ones
    # This is getting complex. Let me just use a simpler approach:
    # Since B = x/S, and we're optimizing over x with the normalization implicit,
    # I'll use the chain rule more carefully.
    
    # Alternative: just compute the gradient numerically for the soft-min part.
    # Given the complexity, let me use a different strategy:
    # Use L-BFGS-B with numerical gradient (it can do this).
    
    value = -soft_min
    # For numerical gradient, return None for grad
    return value, None


def _penalized_objective(x, beta=_SOFT_BETA):
    """
    Penalized objective: -soft_min_area + penalty for points outside triangle.
    Uses barycentric coordinates, so points are always inside if all coords >= 0.
    We ensure this by clipping.
    """
    n = N
    B = x.reshape(n, 3)
    B = np.maximum(B, 0.0)
    S = B.sum(axis=1, keepdims=True)
    B = B / S
    
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, _S3H])
    P = B[:, 0:1] * V0 + B[:, 1:2] * V1 + B[:, 2:3] * V2
    
    # Compute all areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = P[j, 0] - P[i, 0]
                dy1 = P[j, 1] - P[i, 1]
                dx2 = P[k, 0] - P[i, 0]
                dy2 = P[k, 1] - P[i, 1]
                a = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                areas.append(a)
    
    areas = np.array(areas)
    m = areas.min()
    exp_neg = np.exp(-(areas - m) / beta)
    sum_exp = np.sum(exp_neg)
    soft_min = m - beta * np.log(sum_exp)
    
    return -soft_min


def _local_polish(points, max_iter=200, time_limit=120.0):
    """
    Local optimization using L-BFGS-B on barycentric coordinates.
    points: (n, 2) array.
    Returns improved (n, 2) array.
    """
    n = points.shape[0]
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, _S3H])
    
    # Convert to barycentric
    # P = b0*V0 + b1*V1 + b2*V2
    # b0 + b1 + b2 = 1
    # P = b1*V1 + b2*V2 (since V0 = 0)
    # So b1*1 + b2*0.5 = Px, b2*_S3H = Py
    # b2 = Py / _S3H
    # b1 = Px - 0.5*b2
    # b0 = 1 - b1 - b2
    
    bary = np.zeros((n, 3))
    for i in range(n):
        b2 = points[i, 1] / _S3H
        b1 = points[i, 0] - 0.5 * b2
        b0 = 1.0 - b1 - b2
        bary[i] = [b0, b1, b2]
    
    x0 = bary.flatten()
    
    # Bounds: each barycentric coord in [0, 1]
    bounds = [(0.0, 1.0)] * (n * 3)
    
    # Use L-BFGS-B with numerical gradient
    result = minimize(
        _penalized_objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < _penalized_objective(x0):
        B = result.x.reshape(n, 3)
        B = np.maximum(B, 0.0)
        B = B / B.sum(axis=1, keepdims=True)
        P = B[:, 0:1] * V0 + B[:, 1:2] * V1 + B[:, 2:3] * V2
        return P
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - 5.0  # safety margin
    
    best_points = None
    best_score = 0.0
    
    np.random.seed(42)
    
    # Generate multiple initial configurations and polish each
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random points in triangle
        points = np.zeros((N, 2))
        for i in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            points[i] = [u + 0.5 * v, _S3H * v]
        
        # Add some structure: mix of random and grid-like
        if restart % 3 == 0:
            # More structured initial guess
            for i in range(N):
                u = np.random.random() * 0.8 + 0.1
                v = np.random.random() * 0.8 + 0.1
                if u + v > 1.0:
                    u = 1.0 - u
                    v = 1.0 - v
                points[i] = [u + 0.5 * v, _S3H * v]
        
        # Local polish
        polished = _local_polish(points, max_iter=100)
        
        # Evaluate
        score = _min_triangle_area(polished)
        
        if score > best_score:
            best_score = score
            best_points = polished.copy()
        
        # If we have time, do more iterations from this point
        if time.time() < deadline - 10.0:
            for extra_iter in range(3):
                if time.time() > deadline:
                    break
                polished2 = _local_polish(polished, max_iter=50)
                score2 = _min_triangle_area(polished2)
                if score2 > best_score:
                    best_score = score2
                    best_points = polished2.copy()
                polished = polished2
    
    if best_points is None:
        # Fallback: simple random
        points = np.zeros((N, 2))
        for i in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            points[i] = [u + 0.5 * v, _S3H * v]
        best_points = points
    
    return best_points
# EVOLVE-BLOCK-END
