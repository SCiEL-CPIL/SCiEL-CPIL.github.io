import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import linprog

N = 11
SQRT3 = np.sqrt(3)
TRIG_AREA = 0.5 * SQRT3 / 2.0  # sqrt(3)/4

# Triangle vertices in Cartesian coordinates
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])
CENTROID = (V1 + V2 + V3) / 3.0


def _cart_to_bary(points):
    """Convert N points from Cartesian to barycentric coordinates."""
    n = len(points)
    bary = np.zeros((n, 3))
    bary[:, 0] = (V3[1] * (points[:, 0] - V1[0]) - (V3[0] - V1[0]) * (points[:, 1] - V1[1])) / (
        (V3[1] - V1[1]) * (V2[0] - V1[0]) - (V3[0] - V1[0]) * (V2[1] - V1[1])
    )
    # Simplified: for equilateral triangle with base on x-axis
    # bary[0] = 1 - x/1 - y/(sqrt(3)/2) ... actually let me use the standard formula
    # bary0 = area(P,V2,V3)/area, bary1 = area(V1,P,V3)/area, bary2 = area(V1,V2,P)/area
    # Using determinant:
    # For point (x,y):
    # bary0 (at V1) = (V2[1]*(V3[0]-x) - V3[1]*(V2[0]-x) + (V3[1]-V2[1])*x + ... ) / (2*area)
    # Simpler approach:
    # x = bary1 * 1 + bary2 * 0.5
    # y = bary1 * 0 + bary2 * sqrt(3)/2
    # So: bary2 = y / (sqrt(3)/2) = 2y/sqrt(3)
    #     bary1 = x - 0.5 * bary2 = x - y/sqrt(3)
    #     bary0 = 1 - bary1 - bary2
    
    bary2 = points[:, 1] / (SQRT3 / 2.0)  # = 2*y/sqrt(3)
    bary1 = points[:, 0] - 0.5 * bary2
    bary0 = 1.0 - bary1 - bary2
    bary[:, 0] = bary0
    bary[:, 1] = bary1
    bary[:, 2] = bary2
    return bary


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian."""
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * (SQRT3 / 2.0)
    return np.column_stack((x, y))


def _min_area_bary(bary):
    """
    Compute all C(N,3) triangle areas from barycentric coordinates.
    Area of triangle formed by points with barycentric coords = |det| * TRIG_AREA
    where det is the determinant of the 3x3 matrix of barycentric coords.
    Returns: (min_area, all_areas_array)
    """
    n = len(bary)
    # Precompute all C(n,3) determinants efficiently
    areas = np.empty(165)  # C(11,3) = 165
    idx = 0
    for i in range(n - 2):
        bi = bary[i]
        for j in range(i + 1, n - 1):
            bj = bary[j]
            for k in range(j + 1, n):
                bk = bary[k]
                # det of [[b0i,b1i,b2i],[b0j,b1j,b2j],[b0k,b1k,b2k]]
                d = (bi[0] * (bj[1] * bk[2] - bj[2] * bk[1])
                     - bi[1] * (bj[0] * bk[2] - bj[2] * bk[0])
                     + bi[2] * (bj[0] * bk[1] - bj[1] * bk[0]))
                areas[idx] = abs(d)
                idx += 1
    min_a = np.min(areas)
    return min_a * TRIG_AREA, areas * TRIG_AREA


def _min_area_cart(points):
    """Compute minimum triangle area from Cartesian points."""
    n = len(points)
    min_area = np.inf
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            vj = points[j] - pi
            for k in range(j + 1, n):
                vk = points[k] - pi
                area = 0.5 * abs(vj[0] * vk[1] - vj[1] * vk[0])
                if area < min_area:
                    min_area = area
    return min_area


def _softmin_area_bary(bary, temperature=0.01):
    """
    Compute a smooth approximation of the minimum area using log-sum-exp.
    Returns (softmin_value, gradient_wrt_bary)
    """
    n = len(bary)
    # Compute all areas
    areas = np.empty(165)
    idx = 0
    for i in range(n - 2):
        bi = bary[i]
        for j in range(i + 1, n - 1):
            bj = bary[j]
            for k in range(j + 1, n):
                bk = bary[k]
                d = (bi[0] * (bj[1] * bk[2] - bj[2] * bk[1])
                     - bi[1] * (bj[0] * bk[2] - bj[2] * bk[0])
                     + bi[2] * (bj[0] * bk[1] - bj[1] * bk[0]))
                areas[idx] = abs(d)
                idx += 1
    areas = areas * TRIG_AREA
    
    # Softmin: -T * log(sum(exp(-areas/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    softmin_val = -temperature * log_sum_exp
    
    # Gradient with respect to areas: w_i = exp(-areas_i/T) / sum(exp(-areas_j/T))
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # Now we need gradient of softmin w.r.t. bary
    # d(softmin)/d(bary_m) = sum_i w_i * d(area_i)/d(bary_m)
    # d(area_i)/d(bary_m) = sign(det_i) * TRIG_AREA * d(det_i)/d(bary_m)
    
    # For each triangle (i,j,k), the determinant is:
    # det = b0i*(b1j*b2k - b2j*b1k) - b1i*(b0j*b2k - b2j*b0k) + b2i*(b0j*b1k - b1j*b0k)
    # Gradient w.r.t. bary[i] = (b1j*b2k-b2j*b1k, -(b0j*b2k-b2j*b0k), b0j*b1k-b1j*b0k)
    # Gradient w.r.t. bary[j] = (-b0i*b2k+b2i*b0k, b0i*b1k-b2i*b1k, -b0i*b1j+b1i*b0j)
    # Gradient w.r.t. bary[k] = (b0i*b2j-b2i*b0j, -b0i*b1j+b1i*b0j, b0i*b1j-b1i*b0j) ... need to recompute
    
    # Actually, let me just compute the gradient numerically or use a simpler approach.
    # For now, let me compute the gradient analytically.
    
    grad_bary = np.zeros((n, 3))
    idx = 0
    for i in range(n - 2):
        bi = bary[i]
        for j in range(i + 1, n - 1):
            bj = bary[j]
            for k in range(j + 1, n):
                bk = bary[k]
                d = (bi[0] * (bj[1] * bk[2] - bj[2] * bk[1])
                     - bi[1] * (bj[0] * bk[2] - bj[2] * bk[0])
                     + bi[2] * (bj[0] * bk[1] - bj[1] * bk[0]))
                w = weights[idx] * TRIG_AREA
                idx += 1
                if abs(d) < 1e-15:
                    continue
                sign = np.sign(d)
                # Gradient w.r.t. bary[i]
                gi = np.array([
                    bj[1] * bk[2] - bj[2] * bk[1],
                    -(bj[0] * bk[2] - bj[2] * bk[0]),
                    bj[0] * bk[1] - bj[1] * bk[0]
                ])
                # Gradient w.r.t. bary[j]
                gj = np.array([
                    -(bi[0] * bk[2] - bi[2] * bk[0]),
                    bi[0] * bk[2] - bi[2] * bk[0],  # Wait, need to be more careful
                    # Actually: d/d(bj) of det where det = bi0*(bj1*bk2-bj2*bk1) - bi1*(bj0*bk2-bj2*bk0) + bi2*(bj0*bk1-bj1*bk0)
                    # = -bi0*(bk2-bk0)... let me redo
                ])
                # Let me just use the formula properly:
                # det = |b0i b1i b2i|
                #       |b0j b1j b2j|
                #       |b0k b1k b2k|
                # d(det)/d(b0j) = -bi1*bk2 + bi2*bk1  (cofactor expansion along row 2, col 1 with sign)
                # Actually, the cofactor of b0j is (-1)^(1+1) * det([[b1i,b2i],[b1k,b2k]]) = b1i*b2k - b2i*b1k
                # d(det)/d(b1j) = (-1)^(1+2) * det([[b0i,b2i],[b0k,b2k]]) = -(b0i*b2k - b2i*b0k)
                # d(det)/d(b2j) = (-1)^(1+3) * det([[b0i,b1i],[b0k,b1k]]) = b0i*b1k - b1i*b0k
                # Similarly for k:
                # d(det)/d(b0k) = (-1)^(2+1) * det([[b1i,b2i],[b1j,b2j]]) = -(b1i*b2j - b2i*b1j)
                # d(det)/d(b1k) = (-1)^(2+2) * det([[b0i,b2i],[b0j,b2j]]) = b0i*b2j - b2i*b0j
                # d(det)/d(b2k) = (-1)^(2+3) * det([[b0i,b1i],[b0j,b1j]]) = -(b0i*b1j - b1i*b0j)
                
                grad_i = np.array([bj[1]*bk[2]-bj[2]*bk[1], 
                                   -(bj[0]*bk[2]-bj[2]*bk[0]), 
                                   bj[0]*bk[1]-bj[1]*bk[0]])
                grad_j = np.array([-(bi[1]*bk[2]-bi[2]*bk[1]), 
                                   bi[0]*bk[2]-bi[2]*bk[0], 
                                   bi[0]*bk[1]-bi[1]*bk[0]])
                grad_k = np.array([bi[1]*bj[2]-bi[2]*bj[1], 
                                   -(bi[0]*bj[2]-bi[2]*bj[0]), 
                                   bi[0]*bj[1]-bi[1]*bj[0]])
                
                grad_bary[i] += w * sign * grad_i
                grad_bary[j] += w * sign * grad_j
                grad_bary[k] += w * sign * grad_k
    
    return softmin_val, grad_bary


def _objective_and_grad(x_flat):
    """
    Objective function for L-BFGS-B: minimize negative softmin area.
    x_flat: 22-dimensional array (11 points * 2 coords)
    Returns: (value, gradient)
    """
    points = x_flat.reshape(N, 2)
    bary = _cart_to_bary(points)
    
    # Check if points are inside triangle (with tolerance)
    if np.any(bary < -0.05):
        # Large penalty for being far outside
        penalty = 10.0 * np.sum(np.minimum(bary, 0)**2)
        return penalty, np.zeros(22)
    
    softmin_val, grad_bary = _softmin_area_bary(bary, temperature=0.01)
    
    # Gradient w.r.t. Cartesian: grad_cart = grad_bary * J
    # where J is the Jacobian of barycentric w.r.t. Cartesian
    # bary0 = 1 - x - 2y/sqrt(3)  -> db0/dx = -1, db0/dy = -2/sqrt(3)
    # bary1 = x - y/sqrt(3)       -> db1/dx = 1,  db1/dy = -1/sqrt(3)
    # bary2 = 2y/sqrt(3)          -> db2/dx = 0,  db2/dy = 2/sqrt(3)
    
    grad_cart = np.zeros((N, 2))
    for i in range(N):
        grad_cart[i, 0] = -grad_bary[i, 0] + grad_bary[i, 1]
        grad_cart[i, 1] = (-2.0/SQRT3) * grad_bary[i, 0] + (-1.0/SQRT3) * grad_bary[i, 1] + (2.0/SQRT3) * grad_bary[i, 2]
    
    # We want to maximize softmin area, so minimize -softmin
    value = -softmin_val
    grad = -grad_cart.flatten()
    
    return value, grad


def _generate_inits(rng, n_inits):
    """Generate multiple initial configurations."""
    inits = []
    
    # Init 1: Hexagonal-like packing
    pts = []
    center = CENTROID.copy()
    pts.append(center)
    for k in range(6):
        angle = np.pi / 3 * k + np.pi / 6
        r = 0.28
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        # Project to triangle if outside
        b = _cart_to_bary(np.array([[x, y]]))[0]
        if b[0] < 0 or b[1] < 0 or b[2] < 0:
            # Move towards centroid
            t = max(b[0], b[1], b[2])
            if t < 0:
                x = center[0]
                y = center[1]
            else:
                # Scale towards centroid
                x = center[0] + (x - center[0]) * 0.7
                y = center[1] + (y - center[1]) * 0.7
        pts.append([x, y])
    # Add 4 more points near corners
    for frac in [0.15, 0.35]:
        for corner in [V1, V2, V3]:
            p = corner * frac + CENTROID * (1 - frac)
            if len(pts) < 11:
                pts.append(p)
            else:
                break
        if len(pts) >= 11:
            break
    while len(pts) < 11:
        pts.append(CENTROID + rng.normal(0, 0.1, 2))
    inits.append(np.array(pts[:11]))
    
    # Init 2: Uniform random
    pts = []
    for _ in range(N):
        for _try in range(100):
            p = rng.uniform(0, 1, 2)
            b = _cart_to_bary(np.array([p]))[0]
            if b[0] >= 0 and b[1] >= 0 and b[2] >= 0:
                pts.append(p)
                break
    while len(pts) < N:
        pts.append(CENTROID)
    inits.append(np.array(pts[:11]))
    
    # Init 3: Points along edges and interior
    pts = []
    # 3 corners
    pts.append(V1 + 0.1 * (CENTROID - V1))
    pts.append(V2 + 0.1 * (CENTROID - V2))
    pts.append(V3 + 0.1 * (CENTROID - V3))
    # 3 edge midpoints (slightly inward)
    for edge_pair in [(V1, V2), (V2, V3), (V3, V1)]:
        mid = (edge_pair[0] + edge_pair[1]) / 2
        mid_in = mid + 0.15 * (CENTROID - mid)
        pts.append(mid_in)
    # 3 inner ring
    for k in range(3):
        angle = 2 * np.pi / 3 * k
        r = 0.2
        x = CENTROID[0] + r * np.cos(angle)
        y = CENTROID[1] + r * np.sin(angle)
        pts.append([x, y])
    # 1 center
    pts.append(CENTROID)
    inits.append(np.array(pts[:11]))
    
    # Additional random inits
    for _ in range(n_inits - 3):
        pts = []
        for _ in range(N):
            for _try in range(100):
                p = rng.uniform(0, 1, 2)
                b = _cart_to_bary(np.array([p]))[0]
                if b[0] >= 0 and b[1] >= 0 and b[2] >= 0:
                    pts.append(p)
                    break
        while len(pts) < N:
            pts.append(CENTROID)
        inits.append(np.array(pts[:11]))
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    rng = np.random.default_rng(42)
    
    best_area = 0.0
    best_points = None
    
    def _is_time_up():
        return time.time() > deadline - 10.0
    
    # Generate initial configurations
    inits = _generate_inits(rng, 8)
    
    # Evaluate initial areas
    for init in inits:
        area = _min_area_cart(init)
        if area > best_area:
            best_area = area
            best_points = init.copy()
    
    # L-BFGS-B optimization with softmin objective
    for i_init, init in enumerate(inits):
        if _is_time_up():
            break
        
        x0 = init.flatten()
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            x_opt = res.x
            pts_opt = x_opt.reshape(N, 2)
            
            # Project to triangle (clip barycentric coords)
            bary_opt = _cart_to_bary(pts_opt)
            # Clip to [0, 1] and renormalize
            bary_opt = np.maximum(bary_opt, 0)
            # Normalize each row to sum to 1
            row_sums = bary_opt.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            bary_opt = bary_opt / row_sums
            
            pts_final = _bary_to_cart(bary_opt)
            
            area_final = _min_area_cart(pts_final)
            
            if area_final > best_area:
                best_area = area_final
                best_points = pts_final.copy()
                
        except Exception as e:
            pass
    
    # Final local refinement with coordinate descent on the true objective
    if best_points is not None and not _is_time_up():
        current_points = best_points.copy()
        current_area = best_area
        
        for _pass in range(10):
            if _is_time_up():
                break
            improved = False
            for i in range(N):
                if _is_time_up():
                    break
                # Try to improve point i using Nelder-Mead in 2D
                p_i = current_points[i].copy()
                
                def obj_2d(x):
                    trial = current_points.copy()
                    trial[i] = x
                    b = _cart_to_bary(trial)
                    if np.any(b < -0.01):
                        return 100.0
                    return -_min_area_cart(trial)
                
                x0_2d = p_i
                try:
                    res2 = minimize(
                        obj_2d,
                        x0_2d,
                        method='Nelder-Mead',
                        options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 100}
                    )
                    if res2.x[0] < 1.01 and res2.x[1] < 1.0:
                        new_p = res2.x
                        # Check inside
                        b_new = _cart_to_bary(np.array([new_p]))[0]
                        if b_new[0] >= -1e-8 and b_new[1] >= -1e-8 and b_new[2] >= -1e-8:
                            trial = current_points.copy()
                            trial[i] = new_p
                            new_area = _min_area_cart(trial)
                            if new_area > current_area + 1e-12:
                                current_points[i] = new_p
                                current_area = new_area
                                improved = True
                except Exception:
                    pass
            
            if not improved:
                break
        
        if current_area > best_area:
            best_area = current_area
            best_points = current_points
    
    # Fallback
    if best_points is None:
        bary = np.zeros((N, 3))
        for i in range(N):
            for _try in range(1000):
                b1, b2 = rng.uniform(0, 1, 2)
                if b1 + b2 <= 1:
                    bary[i] = [1 - b1 - b2, b1, b2]
                    break
        best_points = _bary_to_cart(bary)
    
    return best_points
