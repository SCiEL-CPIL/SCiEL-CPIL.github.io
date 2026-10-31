"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 145.0  # safety margin below 150s kill

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
VERTS = np.array([V0, V1, V2])


def bary_to_cart(bary):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    # bary[i] = [a, b, c] with a+b+c=1, point = a*V0 + b*V1 + c*V2
    return bary[:, 1] * V1 + bary[:, 2] * V2


def cart_to_bary(pts):
    """Convert Cartesian (n, 2) to barycentric (n, 3)."""
    n = pts.shape[0]
    bary = np.zeros((n, 3))
    # Using area coordinates
    # a = area of (p, V1, V2) / total_area
    # b = area of (V0, p, V2) / total_area
    # c = area of (V0, V1, p) / total_area
    total_area = np.sqrt(3) / 4.0
    for i in range(n):
        p = pts[i]
        # c = y / h where h = sqrt(3)/2
        c = p[1] / (np.sqrt(3) / 2.0)
        # b = (x - c/2) / (1 - c/2) ... let's use the standard formula
        # Actually: x = b + c/2, y = c*sqrt(3)/2
        # So c = 2y/sqrt(3), b = x - c/2, a = 1 - b - c
        c_val = 2.0 * p[1] / np.sqrt(3.0)
        b_val = p[0] - 0.5 * c_val
        a_val = 1.0 - b_val - c_val
        bary[i] = [a_val, b_val, c_val]
    return bary


def triangle_areas_cart(pts):
    """Compute all C(n,3) triangle areas from Cartesian points."""
    n = pts.shape[0]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[k] - pts[i]
                b = pts[j] - pts[i]
                area = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
                areas.append(area)
    return np.array(areas)


def min_area_cart(pts):
    """Minimum triangle area among all triples."""
    return np.min(triangle_areas_cart(pts))


def objective_bary(bary_flat):
    """Objective: negative of minimum triangle area (to minimize)."""
    bary = bary_flat.reshape(-1, 3)
    # Normalize to ensure they sum to 1 (handle numerical drift)
    bary = bary / bary.sum(axis=1, keepdims=True)
    pts = bary_to_cart(bary)
    return -min_area_cart(pts)


def objective_bary_smooth(bary_flat, beta=50.0):
    """Smooth surrogate using log-sum-exp for the min."""
    bary = bary_flat.reshape(-1, 3)
    bary = bary / bary.sum(axis=1, keepdims=True)
    pts = bary_to_cart(bary)
    areas = triangle_areas_cart(pts)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # For stability
    min_val = np.min(areas)
    shifted = -beta * (areas - min_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_val - log_sum / beta
    return -soft_min


def random_bary_init(rng, n=N):
    """Generate random barycentric points inside the triangle."""
    bary = np.zeros((n, 3))
    for i in range(n):
        # Uniform in barycentric simplex
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        bary[i] = [1.0 - u - v, u, v]
    return bary


def polish_cartesian(pts, max_iter=1000, tol=1e-10):
    """Final high-precision polish using gradient descent on Cartesian coords."""
    # Use a smooth objective for better gradients
    beta = 100.0

    def obj(x):
        p = x.reshape(-1, 2)
        areas = triangle_areas_cart(p)
        min_val = np.min(areas)
        shifted = -beta * (areas - min_val)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min = min_val - log_sum / beta
        return -soft_min

    def grad(x):
        p = x.reshape(-1, 2)
        n = p.shape[0]
        areas = triangle_areas_cart(p)
        min_val = np.min(areas)
        shifted = -beta * (areas - min_val)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        # d(soft_min)/d(areas_i) = -exp_vals[i] / sum_exp
        # d(soft_min)/d(p) = sum over triangles of d(soft_min)/d(area_i) * d(area_i)/d(p)
        grad_p = np.zeros((n, 2))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # area = 0.5 * |(p_k - p_i) x (p_j - p_i)|
                    a = p[k] - p[i]
                    b = p[j] - p[i]
                    cross = a[0] * b[1] - a[1] * b[0]
                    # For smooth version, we use the signed area approach
                    # d(area)/d(p_i) = 0.5 * (-(b[1]) - (-b[0]) ... ) 
                    # Let's use: area = 0.5 * |cross|
                    # For gradient of soft min, we need d(area)/d(p)
                    # Using the absolute value, gradient is sign(cross) * d(cross)/d(p)
                    sign = 1.0 if cross > 0 else -1.0
                    # d(cross)/d(p_i): cross = (p_k - p_i)[0]*(p_j - p_i)[1] - (p_k - p_i)[1]*(p_j - p_i)[0]
                    # = p_k0*p_j1 - p_k0*p_i1 - p_i0*p_j1 + p_i0*p_i1 - p_k1*p_j0 + p_k1*p_i0 + p_i1*p_j0 - p_i1*p_i0
                    # d/d(p_i0) = -p_j1 + p_i1 + p_k1 - p_i1 = -p_j1 + p_k1
                    # d/d(p_i1) = -p_k0 + p_i0 + p_j0 - p_i0 = -p_k0 + p_j0
                    # d/d(p_j0) = -p_k1 + p_i1
                    # d/d(p_j1) = p_k0 - p_i0
                    # d/d(p_k0) = p_j1 - p_i1
                    # d/d(p_k1) = -p_j0 + p_i0
                    d_pi0 = -p[j][1] + p[k][1]
                    d_pi1 = -p[k][0] + p[j][0]
                    d_pj0 = -p[k][1] + p[i][1]
                    d_pj1 = p[k][0] - p[i][0]
                    d_pk0 = p[j][1] - p[i][1]
                    d_pk1 = -p[j][0] + p[i][0]

                    coeff = -sign * 0.5 * exp_vals[idx] / sum_exp  # d(-soft_min)/d(area) * sign
                    # Actually: d(soft_min)/d(area_i) = -exp_vals[i]/sum_exp
                    # d(-soft_min)/d(area_i) = exp_vals[i]/sum_exp
                    # d(area_i)/d(p) = 0.5 * sign * d(cross)/d(p)
                    # So d(-soft_min)/d(p) = (exp_vals[i]/sum_exp) * 0.5 * sign * d(cross)/d(p)
                    coeff = 0.5 * sign * exp_vals[idx] / sum_exp

                    grad_p[i][0] += coeff * d_pi0
                    grad_p[i][1] += coeff * d_pi1
                    grad_p[j][0] += coeff * d_pj0
                    grad_p[j][1] += coeff * d_pj1
                    grad_p[k][0] += coeff * d_pk0
                    grad_p[k][1] += coeff * d_pk1
                    idx += 1

        return grad_p.flatten()

    x0 = pts.flatten()
    result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return result.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)

    best_pts = None
    best_score = -np.inf

    # Phase 1: Multiple random restarts with barycentric optimization
    n_restarts = 200

    for restart in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 30:
            break

        # Random initialization in barycentric coords
        bary0 = random_bary_init(rng)
        x0 = bary0.flatten()

        # Optimize using L-BFGS-B on barycentric coordinates
        # Constraints: each bary[i] >= 0 (sum is 1, so <= 1 is implied if others >= 0)
        bounds = [(0.0, 1.0)] * (N * 3)

        # Use smooth objective for better optimization
        result = minimize(objective_bary_smooth, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})

        bary_opt = result.x.reshape(N, 3)
        bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
        pts = bary_to_cart(bary_opt)

        # Check validity
        score = min_area_cart(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

        # Try polynomial restarts with different seeds
        if restart % 50 == 49:
            rng = np.random.RandomState(rng.randint(0, 2**31))

    # Phase 2: Polish the best result in Cartesian coordinates
    if best_pts is not None and time.time() - start_time < TIME_LIMIT - 15:
        polished = polish_cartesian(best_pts, max_iter=500, tol=1e-12)
        polished_score = min_area_cart(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score

    # Final validation: ensure points are inside the triangle
    if best_pts is not None:
        bary_check = cart_to_bary(best_pts)
        # Clamp small negative values to 0
        bary_check = np.clip(bary_check, 0, 1)
        bary_check = bary_check / bary_check.sum(axis=1, keepdims=True)
        best_pts = bary_to_cart(bary_check)

    return best_pts
