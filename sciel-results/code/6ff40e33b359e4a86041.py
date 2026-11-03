import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi, np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.pi, np.array([]), np.array([])
        
    # Extract vertices for all facets
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_radius = np.pi

    # Initializations
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Create a grid of test points Y for the soft-max objective
    # Use a Fibonacci lattice for Y as well, with higher resolution
    M = 3000
    j = np.arange(M) + 0.5
    zy = 1.0 - 2.0 * j / M
    phiy = np.pi * (1.0 + np.sqrt(5.0)) * j
    ry = np.sqrt(1.0 - zy * zy)
    Y = np.column_stack([ry * np.cos(phiy), ry * np.sin(phiy), zy])

    # Main Optimization Loop: Soft-Max Gradient Descent
    # We will run this for most of the time, then switch to Convex Hull polish
    
    points = init_fib.copy()
    
    # Parameters
    sigma = 0.15  # Soft-min temperature
    rho = 0.15    # Soft-max temperature
    lr = 0.005
    decay = 0.98
    
    # Precompute Y norms (should be 1)
    
    while time.time() - start_time < time_limit * 0.7:
        # Compute distances D (M x N)
        C = Y @ points.T  # Cosines of angles
        C = np.clip(C, -1.0, 1.0)
        D = np.arccos(C)
        
        # Soft-min for each y: D_soft[y] = -sigma * log(sum_i exp(-D[y,i]/sigma))
        # To avoid underflow/overflow, we can shift, but D is in [0, pi], so exp(-D/sigma) is small.
        # Let's use log-sum-exp trick
        # log(sum exp(x)) = max(x) + log(sum exp(x - max(x)))
        # Here x = -D/sigma. max(x) = -min(D)/sigma.
        
        # We need gradients, so let's compute weights explicitly.
        # Weights w[y,i] = exp(-D[y,i]/sigma) / sum_j exp(-D[y,j]/sigma)
        
        # For numerical stability in exp, subtract min D for each row
        D_min = np.min(D, axis=1, keepdims=True)
        exp_term = np.exp(-(D - D_min) / sigma)
        sum_exp = np.sum(exp_term, axis=1, keepdims=True)
        W = exp_term / sum_exp  # (M, N)
        
        # D_soft
        D_soft = D_min.squeeze(1) - sigma * np.log(sum_exp.squeeze(1))
        
        # Soft-max for y: J = rho * log(sum_y exp(D_soft[y]/rho))
        # Weights v[y] = exp(D_soft[y]/rho) / sum_z exp(D_soft[z]/rho)
        
        D_soft_max = np.max(D_soft)
        exp_term_y = np.exp((D_soft - D_soft_max) / rho)
        sum_exp_y = np.sum(exp_term_y)
        V = exp_term_y / sum_exp_y  # (M,)
        
        # Gradient of J w.r.t points
        # J = rho * log(sum_y exp(D_soft[y]/rho))
        # dJ/dX_i = sum_y V[y] * (dD_soft[y]/dX_i)
        
        # dD_soft[y]/dX_i = sum_j W[y,j] * (dD[y,j]/dX_i)
        # dD[y,j]/dX_i = 0 if i != j
        # dD[y,j]/dX_j = -Y[y] / sqrt(1 - (Y[y] . X[j])^2)
        
        # So dD_soft[y]/dX_i = W[y,i] * (-Y[y] / sqrt(1 - C[y,i]^2))
        
        # We need to compute G[y,i] = W[y,i] * (-Y[y] / sqrt(1 - C[y,i]^2))
        # And then grad[i] = sum_y V[y] * G[y,i]
        
        # Term: 1 / sqrt(1 - C^2)
        # Clip C to avoid 1 - C^2 <= 0
        C_safe = np.clip(C, -1.0 + 1e-6, 1.0 - 1e-6)
        denom = np.sqrt(1.0 - C_safe * C_safe)
        
        # G is (M, N, 3)
        # Y is (M, 3) -> Y[:, None, :]
        # W is (M, N) -> W[:, :, None]
        # denom is (M, N) -> denom[:, :, None]
        
        # Vector part: -Y[y]
        Y_exp = Y[:, np.newaxis, :]  # (M, 1, 3)
        W_exp = W[:, :, np.newaxis]  # (M, N, 1)
        denom_exp = denom[:, :, np.newaxis]  # (M, N, 1)
        
        # G = W * (-Y / denom)
        G = W_exp * (-Y_exp / denom_exp)  # (M, N, 3)
        
        # grad[i] = sum_y V[y] * G[y, i, :]
        # V is (M,) -> V[:, None, None]
        V_exp = V[:, np.newaxis, np.newaxis]  # (M, 1, 1)
        
        grad = np.sum(V_exp * G, axis=0)  # (N, 3)
        
        # Update points
        points -= lr * grad
        
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check improvement
        # We can't easily check true covering radius without hull, 
        # but we can check the soft objective or just trust the process.
        # Let's check the soft max value occasionally? No, too slow.
        # Just update best if we do a hull check every 100 steps?
        
        # Anneal
        sigma *= decay
        rho *= decay
        
        if sigma < 0.01:
            sigma = 0.01
        if rho < 0.01:
            rho = 0.01

    # Polish Phase: Convex Hull Optimization
    # Use the current points as start for the hull-based optimizer
    # This ensures we minimize the *exact* covering radius
    
    alpha = 0.05
    no_improve = 0
    
    while time.time() - start_time < time_limit - 5.0:
        max_radius, radii, simplices = compute_covering_radius_and_facets(points)
        
        if len(simplices) == 0:
            break
            
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                alpha *= 0.9
                if alpha < 1e-4:
                    break

        # Find largest facet
        max_idx = np.argmax(radii)
        simplex = simplices[max_idx]
        i_idx, j_idx, k_idx = simplex
        
        a = points[i_idx]
        b = points[j_idx]
        c = points[k_idx]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        u = n / norm_n
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
        
        # Move vertices towards u
        step_size = alpha * radii[max_idx]
        
        points[i_idx] += step_size * (u - a)
        points[j_idx] += step_size * (u - b)
        points[k_idx] += step_size * (u - c)
        
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    if best_points is None:
        best_points = points

    return best_points
