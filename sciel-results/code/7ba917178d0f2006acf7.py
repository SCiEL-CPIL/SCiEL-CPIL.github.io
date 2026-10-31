import numpy as np
import scipy.optimize as so
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def get_areas(points):
    # points: (N, 2)
    # returns array of all triangle areas
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def smooth_min(areas, beta=20.0):
    # Soft min approximation
    # min(a) ~ -1/beta * log(sum(exp(-beta * a)))
    # To avoid overflow, shift by min
    min_a = np.min(areas)
    shifted = areas - min_a
    # exp(-beta * shifted) is <= 1
    sum_exp = np.sum(np.exp(-beta * shifted))
    # -1/beta * log(sum_exp) + min_a
    return min_a - (1.0/beta) * np.log(sum_exp)

def objective(x, beta=20.0):
    points = x.reshape(N, 2)
    areas = get_areas(points)
    smin = smooth_min(areas, beta)
    return -smin

def objective_grad(x, beta=20.0):
    points = x.reshape(N, 2)
    # Calculate areas and weights
    # We need sum over k involving i of w_k * grad_A_k
    # w_k = exp(-beta * A_k) / Z
    # Z = sum exp(-beta * A_k)
    
    # Precompute areas and cross products signs
    n_tri = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                n_tri += 1
    weights = np.zeros(n_tri)
    grad_sum = np.zeros((N, 2))
    
    # First pass: calculate areas and exp terms
    areas = np.zeros(n_tri)
    signs = np.zeros(n_tri)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                area = 0.5 * abs(cross)
                areas[idx] = area
                signs[idx] = 1.0 if cross > 0 else -1.0
                idx += 1
                
    min_a = np.min(areas)
    shifted = areas - min_a
    exp_terms = np.exp(-beta * shifted)
    Z = np.sum(exp_terms)
    weights = exp_terms / Z
    
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                w = weights[idx]
                s = signs[idx]
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                
                # grad A_k wrt p1
                # 0.5 * s * (y_j - y_k, x_k - x_j)
                g1 = 0.5 * s * np.array([p2[1]-p3[1], p3[0]-p2[0]])
                # grad A_k wrt p2
                # 0.5 * s * (y_k - y_i, x_i - x_k)
                g2 = 0.5 * s * np.array([p3[1]-p1[1], p1[0]-p3[0]])
                # grad A_k wrt p3
                # 0.5 * s * (y_i - y_j, x_j - x_i)
                g3 = 0.5 * s * np.array([p1[1]-p2[1], p2[0]-p1[0]])
                
                grad_sum[i] += w * g1
                grad_sum[j] += w * g2
                grad_sum[k] += w * g3
                idx += 1
                
    # d(-S)/dx = - dS/dx
    # dS/dx = (1/Z) sum w_k_unnorm * grad_A_k  ... wait
    # S = min_a - 1/beta log(Z)
    # dS/da_m = 1/beta * (1/Z) * (-beta * exp(-beta(a_m - min_a)) * (1 if a_m == min_a else 0 ...))
    # Actually, the derivative of the soft min w.r.t a_m is:
    # p_m = exp(-beta a_m) / sum(exp(-beta a_k))
    # dS/da_m = -p_m
    # So dS/dx_i = sum_m p_m * dA_m/dx_i
    # And d(-S)/dx_i = - sum_m p_m * dA_m/dx_i
    
    # My weights `weights` are p_m.
    # So grad_sum is sum p_m * grad_A_m.
    # So objective grad is -grad_sum.
    
    return -grad_sum.flatten()

def constraints(x):
    points = x.reshape(N, 2)
    cons = []
    for i in range(N):
        x_i, y_i = points[i]
        # y >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # sqrt(3)*x + y <= sqrt(3)  => sqrt(3) - sqrt(3)*x - y >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: np.sqrt(3) - np.sqrt(3)*x[2*i] - x[2*i+1]})
        # y <= sqrt(3)*x => sqrt(3)*x - y >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: np.sqrt(3)*x[2*i] - x[2*i+1]})
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initialization: Jittered Grid
    # Generate a grid of points in barycentric coords
    # For N=11, let's try a 4x4 grid (16 pts) and pick 11?
    # Or just a random initialization with a good seed.
    # Let's use a few restarts.
    
    best_x = None
    best_val = -np.inf
    
    # Restart 1: Grid-like
    # Barycentric grid
    M = 4
    pts = []
    for u in np.linspace(0.1, 0.8, M):
        for v in np.linspace(0.1, 0.8, M):
            w = 1.0 - u - v
            if w > 0.05:
                p = u*V0 + v*V1 + w*V2
                pts.append(p)
    # If we have more than N, subsample?
    # If less, add random?
    # For M=4, we have 16 points.
    # Let's just take the first 11.
    if len(pts) >= N:
        x0 = np.array(pts[:N]).flatten()
    else:
        x0 = np.random.rand(2*N)
        
    # Run optimization
    try:
        res = so.minimize(objective, x0, method='SLSQP', jac=objective_grad, 
                          constraints=constraints(x0), 
                          options={'maxiter': 500, 'ftol': 1e-10})
        if res.success:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x
    except Exception:
        pass
        
    if time.time() > deadline:
        if best_x is not None:
            return best_x.reshape(N, 2)
        return np.random.rand(N, 2) # Fallback

    # Restart 2: Random
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        x0 = np.random.rand(2*N)
        # Scale to be inside triangle roughly?
        # Just let SLSQP handle it, but start inside to be safe.
        # Generate random points in triangle
        pts_r = []
        for _ in range(N):
            u, v = np.random.rand(), np.random.rand()
            if u+v > 1:
                u, v = 1-u, 1-v
            p = u*V0 + v*V1 + (1-u-v)*V2
            pts_r.append(p)
        x0 = np.array(pts_r).flatten()
        
        try:
            res = so.minimize(objective, x0, method='SLSQP', jac=objective_grad, 
                              constraints=constraints(x0), 
                              options={'maxiter': 300, 'ftol': 1e-10})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x
        except Exception:
            pass
            
    if best_x is not None:
        return best_x.reshape(N, 2)
    else:
        return np.array(pts[:N]) # Fallback to grid

