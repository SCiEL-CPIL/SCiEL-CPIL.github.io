import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

DEADLINE = 620.0

def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 0.0
    if d_min < 1e-10:
        return 0.0
        
    ratio = d_min / d_max
    return ratio * ratio

def _spherical_to_cartesian(params):
    """
    Convert 28D spherical coordinates (14 points, 2 params each) to 42D cartesian.
    params: (28,) array where params[2i] = theta (polar, 0..pi), params[2i+1] = phi (azimuth, 0..2pi)
    Returns: (14, 3) array of points on unit sphere.
    """
    theta = params[0::2]
    phi = params[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(points):
    """
    Convert (14, 3) cartesian points on unit sphere to (28,) spherical params.
    """
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return np.column_stack([theta, phi]).flatten()

def _score_soft(params, temperature=0.05):
    """
    Smooth surrogate score using soft-min and soft-max.
    params: (28,) spherical coordinates.
    Returns (value, gradient) for minimization of -score.
    """
    points = _spherical_to_cartesian(params)
    
    # Compute all pairwise distances
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    dists_mat = np.sqrt(np.maximum(D2, 1e-30))
    
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((14, 14), dtype=bool), k=1)
    dists = dists_mat[mask]
    
    if len(dists) == 0:
        return 0.0, np.zeros(28)
    
    d_min_raw = np.min(dists)
    d_max_raw = np.max(dists)
    
    if d_max_raw < 1e-10 or d_min_raw < 1e-10:
        return 0.0, np.zeros(28)
    
    scale = 1.0 / temperature
    
    # Soft min: d_min_soft = d_min_raw - T * log(sum(exp(-(d - d_min)/T)))
    exp_neg = np.exp(-(dists - d_min_raw) * scale)
    sum_exp_neg = np.sum(exp_neg)
    soft_min = d_min_raw - temperature * np.log(sum_exp_neg)
    
    # Soft max: d_max_soft = d_max_raw + T * log(sum(exp((d - d_max)/T)))
    exp_pos = np.exp((dists - d_max_raw) * scale)
    sum_exp_pos = np.sum(exp_pos)
    soft_max = d_max_raw + temperature * np.log(sum_exp_pos)
    
    if soft_max < 1e-10 or soft_min < 1e-10:
        return 0.0, np.zeros(28)
    
    score = (soft_min / soft_max) ** 2
    val = -score
    
    # Weights for soft min and soft max
    w_min = exp_neg / sum_exp_neg  # (91,)
    w_max = exp_pos / sum_exp_pos  # (91,)
    
    # d(score)/d(soft_min) = 2 * soft_min / soft_max^2
    # d(score)/d(soft_max) = -2 * soft_min^2 / soft_max^3
    coeff_min = 2.0 * soft_min / (soft_max * soft_max)
    coeff_max = -2.0 * (soft_min * soft_min) / (soft_max * soft_max * soft_max)
    
    # d(soft_min)/d(dists) = -w_min
    # d(soft_max)/d(dists) = w_max
    # d(score)/d(dists) = coeff_min * (-w_min) + coeff_max * w_max
    # = -coeff_min * w_min + coeff_max * w_max
    
    dscore_ddists = -coeff_min * w_min + coeff_max * w_max  # (91,)
    
    # Now compute d(score)/d(params) via chain rule
    # d(score)/d(points[i]) = sum_{j != i} dscore_ddists[idx(i,j)] * (points[i] - points[j]) / d_ij
    # where idx maps (i,j) to flat index in upper triangle
    
    # Build mapping: for each pair (i,j) with i<j, flat index = i*13 - i*(i+1)//2 + j - i - 1
    # Let's vectorize: for each point i, sum over j
    
    grad_points = np.zeros((14, 3))
    
    # For efficiency, precompute
    indices_i = []
    indices_j = []
    for i in range(14):
        for j in range(i+1, 14):
            indices_i.append(i)
            indices_j.append(j)
    indices_i = np.array(indices_i)
    indices_j = np.array(indices_j)
    
    # For each pair, compute contribution to gradient of points[i] and points[j]
    pair_dists = dists_mat[indices_i, indices_j]
    pair_vecs = points[indices_i] - points[indices_j]  # (91, 3)
    unit_vecs = pair_vecs / np.maximum(pair_dists, 1e-15)[:, None]  # (91, 3)
    
    # d(score)/d(points[i]) for pair (i,j): dscore_ddists[k] * unit_vec[k]
    # d(score)/d(points[j]) for pair (i,j): -dscore_ddists[k] * unit_vec[k]
    
    for k in range(91):
        i = indices_i[k]
        j = indices_j[k]
        contrib = dscore_ddists[k] * unit_vecs[k]
        grad_points[i] += contrib
        grad_points[j] -= contrib
    
    # Now d(points)/d(params): 
    # points[i] = [sin(theta_i)*cos(phi_i), sin(theta_i)*sin(phi_i), cos(theta_i)]
    # d(points[i])/d(theta_i) = [cos(theta_i)*cos(phi_i), cos(theta_i)*sin(phi_i), -sin(theta_i)]
    # d(points[i])/d(phi_i) = [-sin(theta_i)*sin(phi_i), sin(theta_i)*cos(phi_i), 0]
    
    theta = params[0::2]
    phi = params[1::2]
    
    grad_params = np.zeros(28)
    for i in range(14):
        # d(score)/d(theta_i) = grad_points[i] dot d(points[i])/d(theta_i)
        dpth = np.array([np.cos(theta[i]) * np.cos(phi[i]),
                         np.cos(theta[i]) * np.sin(phi[i]),
                         -np.sin(theta[i])])
        dphip = np.array([-np.sin(theta[i]) * np.sin(phi[i]),
                          np.sin(theta[i]) * np.cos(phi[i]),
                          0.0])
        
        grad_params[2*i] = np.dot(grad_points[i], dpth)
        grad_params[2*i+1] = np.dot(grad_points[i], dphip)
    
    # We're minimizing -score, so gradient is -grad_score
    grad_flat = -grad_params
    
    return val, grad_flat

def _score_exact(params):
    """
    Exact score for spherical parameters.
    """
    points = _spherical_to_cartesian(params)
    return _raw_score(points)

def _generate_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    points = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    return points

def _generate_fibonacci_sphere(n=14):
    """
    Generate n points on a sphere using the Fibonacci lattice.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        radius = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _optimize_from(params0, deadline, temperature_start=0.1, temperature_end=0.001):
    """
    Optimize from initial spherical parameters using annealing.
    """
    if time.time() > deadline:
        return params0, _score_exact(params0)
    
    params_current = params0.copy()
    best_params = params0.copy()
    best_score = _score_exact(params0)
    
    # Annealing schedule
    num_stages = 5
    temps = np.geomspace(temperature_start, temperature_end, num_stages)
    
    for temp in temps:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                lambda p: _score_soft(p, temp)[0],
                params_current,
                jac=lambda p: _score_soft(p, temp)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            params_current = res.x
            score = _score_exact(params_current)
            if score > best_score:
                best_score = score
                best_params = params_current.copy()
        except Exception:
            pass
    
    # Final polish with very low temperature
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda p: _score_soft(p, 0.001)[0],
                params_current,
                jac=lambda p: _score_soft(p, 0.001)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            params_current = res.x
            score = _score_exact(params_current)
            if score > best_score:
                best_score = score
                best_params = params_current.copy()
        except Exception:
            pass
    
    return best_params, best_score

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_params = np.zeros(28)
    
    rng = np.random.default_rng(42)
    
    # Initial guesses in cartesian, then convert to spherical
    guesses_cart = []
    
    # 1. Icosahedron + 2 points
    ico = _generate_icosahedron()
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms
    
    p1 = rng.normal(size=3)
    p1 = p1 / np.linalg.norm(p1)
    p2 = rng.normal(size=3)
    p2 = p2 / np.linalg.norm(p2)
    
    guess1 = np.vstack([ico, p1, p2])
    guesses_cart.append(guess1)
    
    # 2. Fibonacci sphere
    fib = _generate_fibonacci_sphere(14)
    guesses_cart.append(fib)
    
    # 3. Random points on sphere
    for _ in range(15):
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        guesses_cart.append(pts)
        
    # 4. Cubic points
    cube = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(cube, axis=1, keepdims=True)
    cube = cube / norms
    guesses_cart.append(cube)
    
    # 5. Octahedron + extra points
    octa = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(octa, axis=1, keepdims=True)
    octa = octa / norms
    for _ in range(8):
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p)
        octa = np.vstack([octa, p])
    guesses_cart.append(octa)
    
    # Convert to spherical and optimize
    for guess in guesses_cart:
        if time.time() > deadline - 40:
            break
        
        params0 = _cartesian_to_spherical(guess)
        params_opt, score = _optimize_from(params0, deadline)
        if score > best_score:
            best_score = score
            best_params = params_opt
    
    # If we have time, try more random starts
    while time.time() < deadline - 60:
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        params0 = _cartesian_to_spherical(pts)
        params_opt, score = _optimize_from(params0, deadline, temperature_start=0.05, temperature_end=0.001)
        if score > best_score:
            best_score = score
            best_params = params_opt
    
    # Final polish on best
    if time.time() < deadline - 10 and best_score > 0:
        try:
            params_final, score_final = _optimize_from(
                best_params, deadline, temperature_start=0.01, temperature_end=0.0005
            )
            if score_final > best_score:
                best_score = score_final
                best_params = params_final
        except Exception:
            pass
    
    if best_score == 0.0:
        # Fallback
        best_points = _generate_icosahedron()[:12]
        p1 = rng.normal(size=3)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=3)
        p2 = p2 / np.linalg.norm(p2)
        best_points = np.vstack([best_points, p1, p2])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
        return best_points
    
    return _spherical_to_cartesian(best_params)

if __name__ == "__main__":
    pts = min_max_dist_dim3_14()
    print(_raw_score(pts))
