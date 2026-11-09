"""Rewritten program: SA on smooth soft-min surrogate + exact polish."""
N = 11
import numpy as np
import time
from scipy.optimize import minimize

H = np.sqrt(3) / 2

# Precompute triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)

# Precompute pair indices
_pair_i, _pair_j = [], []
for i in range(N):
    for j in range(i + 1, N):
        _pair_i.append(i)
        _pair_j.append(j)
_pair_i = np.array(_pair_i)
_pair_j = np.array(_pair_j)


def _min_area(points):
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _min_pairwise_dist(points):
    p1 = points[_pair_i]
    p2 = points[_pair_j]
    d = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
    return d.min()


def _is_inside(p):
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _bary_to_cart(u, v):
    """Convert barycentric (u,v) to Cartesian. u,v >= 0, u+v <= 1."""
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return x, y


def _cart_to_bary(x, y):
    """Convert Cartesian to barycentric (u,v)."""
    w = y / H
    s = 1.0 - w  # u + v
    d = 2.0 * (x - 0.5)  # v - u
    u = (s - d) / 2.0
    v = (s + d) / 2.0
    return max(0.0, u), max(0.0, v)


def _params_to_points(params):
    """Convert 2*N barycentric params to Cartesian points."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        if u < 0:
            u = 0.0
        if v < 0:
            v = 0.0
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        x, y = _bary_to_cart(u, v)
        points[i] = np.array([x, y])
    return points


def _points_to_params(points):
    params = np.zeros(2 * N)
    for i in range(N):
        u, v = _cart_to_bary(points[i][0], points[i][1])
        params[2 * i] = u
        params[2 * i + 1] = v
    return params


def _soft_min_area(params, beta):
    """
    Soft-min of triangle areas using log-sum-exp.
    Returns (soft_min_value, gradient w.r.t. params).
    """
    points = _params_to_points(params)
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for soft min: -logsumexp(-areas/beta) * ... 
    # Actually: soft_min = -1/beta * log(sum(exp(-areas/beta)))
    # This is smooth and approximates min as beta -> 0
    neg_areas = -areas / beta
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min_val = -beta * log_sum
    
    # Gradient: d(soft_min)/d(area_k) = -exp(-area_k/beta) / sum(exp(-area_j/beta))
    weights = np.exp(neg_areas - max_neg)
    weights /= np.sum(weights)
    
    # Now compute gradient w.r.t. params via chain rule
    # area_k = 0.5 * |cross_k|
    # d(area_k)/d(cross_k) = 0.5 * sign(cross_k)
    # d(cross_k)/d(points) is a 3x2 matrix for each triangle
    # We need d(params)/d(points) which is the inverse of _params_to_points
    
    grad_params = np.zeros(2 * N)
    
    for k in range(len(areas)):
        w = weights[k]
        if w < 1e-15:
            continue
        i, j, kk = _idx_i[k], _idx_j[k], _idx_k[k]
        pi, pj, pk = points[i], points[j], points[kk]
        
        # cross = (pj-x_i)*(pk-y_i) - (pj-y_i)*(pk-x_i)
        # d(cross)/d(pi) = [-(pk-y_i), pj-x_i]  ... let me be careful
        # cross = (pj0-pi0)*(pk1-pi1) - (pj1-pi1)*(pk0-pi0)
        # d/dpi0 = -(pk1-pi1) - (pj1-pi1)*(-1) = -(pk1-pi1) + (pj1-pi1) = pj1-pk1
        # d/dpi1 = (pj0-pi0)*(-1) - (pj1-pi1)*(-1) = -(pj0-pi0) + (pj1-pi1) = pj1-pi1-pj0+pi0
        # Hmm, let me redo:
        # cross = (pj0-pi0)*(pk1-pi1) - (pj1-pi1)*(pk0-pi0)
        # d/dpi0: -1*(pk1-pi1) - (pj1-pi1)*(-1) = -(pk1-pi1) + (pj1-pi1) = (pj1-pk1)
        # d/dpi1: (pj0-pi0)*(-1) - (pj1-pi1)*(-1) = -(pj0-pi0) + (pj1-pi1) = (pi0-pj0)+(pj1-pi1)
        
        # d/dpj0: 1*(pk1-pi1) - 0 = pk1-pi1
        # d/dpj1: 0 - 1*(pk0-pi0) = -(pk0-pi0) = pi0-pk0
        
        # d/dpk0: 0 - (pj1-pi1)*1 = -(pj1-pi1) = pi1-pj1
        # d/dpk1: (pj0-pi0)*1 - 0 = pj0-pi0
        
        d_cross_d_pi = np.array([pj[1] - pk[1], (pi[0] - pj[0]) + (pj[1] - pi[1])])
        d_cross_d_pj = np.array([pk[1] - pi[1], pi[0] - pk[0]])
        d_cross_d_pk = np.array([pi[1] - pj[1], pj[0] - pi[0]])
        
        sign = np.sign(cross[k])
        if sign == 0:
            sign = 1.0
        
        d_area_d_pi = 0.5 * sign * d_cross_d_pi
        d_area_d_pj = 0.5 * sign * d_cross_d_pj
        d_area_d_pk = 0.5 * sign * d_cross_d_pk
        
        # Now map from points to params
        # For each point m with barycentric (u_m, v_m):
        # x_m = v_m + 0.5*(1-u_m-v_m) = 0.5 + 0.5*v_m - 0.5*u_m
        # y_m = H*(1-u_m-v_m)
        # dx/du = -0.5, dx/dv = 0.5
        # dy/du = -H, dy/dv = -H
        
        # d(area)/du_m = d_area/dx * (-0.5) + d_area/dy * (-H)
        # d(area)/dv_m = d_area/dx * (0.5) + d_area/dy * (-H)
        
        for idx, d_a in [(i, d_area_d_pi), (j, d_area_d_pj), (kk, d_area_d_pk)]:
            du = d_a[0] * (-0.5) + d_a[1] * (-H)
            dv = d_a[0] * (0.5) + d_a[1] * (-H)
            grad_params[2 * idx] += w * du
            grad_params[2 * idx + 1] += w * dv
    
    return soft_min_val, grad_params


def _lattice_initial_points():
    """Generate initial points on a triangular lattice."""
    points = []
    for row in range(4):
        for col in range(4 - row):
            u = row / 3.0
            v = col / 3.0
            if u + v <= 1.0 + 1e-9:
                x, y = _bary_to_cart(u, v)
                cx, cy = 0.5, H / 3
                x = 0.98 * x + 0.02 * cx
                y = 0.98 * y + 0.02 * cy
                points.append([x, y])
    points.append([0.5, H / 3])
    return np.array(points[:N])


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    center = np.array([0.5, H / 3])
    for i in range(N):
        if not _is_inside(points[i]):
            # Move toward center
            t = 0.0
            while not _is_inside(points[i]) and t < 1.0:
                t += 0.05
                points[i] = (1 - t) * points[i] + t * center
            if not _is_inside(points[i]):
                points[i] = center
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    # Phase 1: Lattice init + packing polish
    try:
        lat_pts = _lattice_initial_points()
        lat_params = _points_to_params(lat_pts)
        
        # Quick packing optimization
        def pack_obj(p):
            pts = _params_to_points(p)
            if not all(_is_inside(pts[i]) for i in range(N)):
                return 1e6
            return -_min_pairwise_dist(pts)
        
        res_pack = minimize(
            pack_obj, lat_params, method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        best_params = res_pack.x
        best_score = _min_area(_params_to_points(best_params))
    except Exception:
        best_params = _points_to_params(_lattice_initial_points())
        best_score = _min_area(_params_to_points(best_params))

    # Phase 2: Simulated Annealing on soft-min surrogate
    if time.time() < deadline - 30.0:
        try:
            rng = np.random.RandomState(12345)
            current_params = best_params.copy()
            current_score = _min_area(_params_to_points(current_params))
            
            beta_start = 0.05
            beta_end = 0.005
            n_steps = 2000
            alpha = 0.999  # cooling rate
            
            for step in range(n_steps):
                if time.time() > deadline - 30.0:
                    break
                
                # Compute soft-min and gradient
                beta = beta_start * (beta_end / beta_start) ** (step / max(1, n_steps - 1))
                soft_val, grad = _soft_min_area(current_params, beta)
                
                # Gradient ascent step (we want to maximize min area)
                step_size = 0.01 * (1.0 - 0.5 * step / n_steps)
                new_params = current_params + step_size * grad + rng.randn(2 * N) * 0.005
                
                # Clamp to valid barycentric range
                for i in range(N):
                    new_params[2 * i] = max(0.0, min(1.0, new_params[2 * i]))
                    new_params[2 * i + 1] = max(0.0, min(1.0, new_params[2 * i + 1]))
                    s = new_params[2 * i] + new_params[2 * i + 1]
                    if s > 1.0:
                        new_params[2 * i] /= s
                        new_params[2 * i + 1] /= s
                
                new_pts = _params_to_points(new_params)
                if all(_is_inside(new_pts[i]) for i in range(N)):
                    new_score = _min_area(new_pts)
                    # Accept with Metropolis criterion
                    if new_score >= current_score or rng.random() < np.exp((new_score - current_score) / (beta * 10)):
                        current_params = new_params
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params.copy()
        except Exception:
            pass

    # Phase 3: Exact polish with Nelder-Mead
    if time.time() < deadline - 15.0:
        try:
            def exact_obj(p):
                pts = _params_to_points(p)
                if not all(_is_inside(pts[i]) for i in range(N)):
                    return 1e6
                return -_min_area(pts)
            
            res3 = minimize(
                exact_obj, best_params, method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score3 = -res3.fun
            if score3 > best_score:
                best_params = res3.x
                best_score = score3
        except Exception:
            pass

    # Phase 4: More SA iterations if time remains
    if time.time() < deadline - 15.0:
        try:
            rng2 = np.random.RandomState(999)
            current_params = best_params.copy()
            current_score = _min_area(_params_to_points(current_params))
            
            beta_start = 0.02
            beta_end = 0.002
            n_steps = 3000
            
            for step in range(n_steps):
                if time.time() > deadline - 15.0:
                    break
                beta = beta_start * (beta_end / beta_start) ** (step / max(1, n_steps - 1))
                soft_val, grad = _soft_min_area(current_params, beta)
                step_size = 0.008 * (1.0 - 0.5 * step / n_steps)
                new_params = current_params + step_size * grad + rng2.randn(2 * N) * 0.003
                for i in range(N):
                    new_params[2 * i] = max(0.0, min(1.0, new_params[2 * i]))
                    new_params[2 * i + 1] = max(0.0, min(1.0, new_params[2 * i + 1]))
                    s = new_params[2 * i] + new_params[2 * i + 1]
                    if s > 1.0:
                        new_params[2 * i] /= s
                        new_params[2 * i + 1] /= s
                new_pts = _params_to_points(new_params)
                if all(_is_inside(new_pts[i]) for i in range(N)):
                    new_score = _min_area(new_pts)
                    if new_score >= current_score or rng2.random() < np.exp((new_score - current_score) / (beta * 10)):
                        current_params = new_params
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params.copy()
        except Exception:
            pass

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            def exact_obj2(p):
                pts = _params_to_points(p)
                if not all(_is_inside(pts[i]) for i in range(N)):
                    return 1e6
                return -_min_area(pts)
            
            res_final = minimize(
                exact_obj2, best_params, method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_f = -res_final.fun
            if score_f > best_score:
                best_params = res_final.x
                best_score = score_f
        except Exception:
            pass

    if best_params is not None:
        points = _params_to_points(best_params)
        points = _clamp_to_triangle(points)
        return points
    else:
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.99:
                    x, y = _bary_to_cart(u, v)
                    pts.append([x, y])
                    break
        return np.array(pts)
