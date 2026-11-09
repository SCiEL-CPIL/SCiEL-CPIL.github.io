import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(cx, cy, theta, scale=1.0):
    angles = theta + np.arange(6) * (np.pi / 3)
    x = cx + scale * np.cos(angles)
    y = cy + scale * np.sin(angles)
    return np.column_stack((x, y))

def min_dist_poly(v1, v2):
    """
    Minimum distance between two convex polygons.
    Returns 0 if intersecting.
    """
    def get_normals(vertices):
        normals = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            # Normal
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                n /= norm
            else:
                n = np.array([1.0, 0.0])
            normals.append(n)
        return normals

    normals = get_normals(v1) + get_normals(v2)
    
    min_gap = float('inf')
    
    for n in normals:
        proj1 = v1.dot(n)
        proj2 = v2.dot(n)
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for separation
        if max1 < min2:
            gap = min2 - max1
            if gap < min_gap:
                min_gap = gap
        elif max2 < min1:
            gap = min1 - max2
            if gap < min_gap:
                min_gap = gap
                
    if min_gap < float('inf'):
        return min_gap
    else:
        return 0.0

def softplus(x, beta=10.0):
    x = np.clip(x, -50, 50)
    return np.log1p(np.exp(beta * x)) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def run_optimization(x0):
        def objective(x):
            inner_params = x[:36].reshape(12, 3)
            xc, yc, phi = x[36], x[37], x[38]
            L = x[39]
            
            if L < 3.0:
                return 1000.0 + (3.0 - L) * 100
            
            outer_normals = []
            for k in range(6):
                angle = phi + k * (np.pi / 3) + np.pi / 2
                n = np.array([np.cos(angle), np.sin(angle)])
                outer_normals.append(n)
            outer_normals = np.array(outer_normals)
            
            verts = []
            for i in range(12):
                xi, yi, thi = inner_params[i]
                v = get_hex_vertices(xi, yi, thi)
                verts.append(v)
            
            penalty_cont = 0.0
            for i in range(12):
                v = verts[i]
                rel_v = v - np.array([xc, yc])
                projs = rel_v.dot(outer_normals.T)
                d_i = np.max(projs)
                penalty_cont += softplus(d_i - L)
                
            penalty_overlap = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    dist = min_dist_poly(verts[i], verts[j])
                    penalty_overlap += softplus(-dist)
                    
            lambda_p = 10.0
            return L + lambda_p * (penalty_cont + penalty_overlap)

        bounds = []
        for i in range(12):
            bounds.append((-10, 10))
            bounds.append((-10, 10))
            bounds.append((0, 2 * np.pi))
            
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, np.pi / 3))
        bounds.append((3.0, 20.0))
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8})
        
        return result.x, result.fun

    best_L = float('inf')
    best_result = None
    
    restarts = 0
    max_restarts = 15
    
    while time.time() < deadline and restarts < max_restarts:
        x0 = np.zeros(40)
        
        if restarts == 0:
            for i in range(12):
                row = i // 4
                col = i % 4
                x_base = (col - 1.5) * 1.8
                y_base = (row - 1.0) * 1.6
                x0[i * 3] = x_base
                x0[i * 3 + 1] = y_base
                x0[i * 3 + 2] = 0.0
            x0[36] = 0.0
            x0[37] = 0.0
            x0[38] = 0.0
            x0[39] = 8.0
        else:
            for i in range(12):
                x0[i * 3] = np.random.uniform(-5, 5)
                x0[i * 3 + 1] = np.random.uniform(-5, 5)
                x0[i * 3 + 2] = np.random.uniform(0, 2 * np.pi)
            x0[36] = 0.0
            x0[37] = 0.0
            x0[38] = np.random.uniform(0, np.pi/3)
            x0[39] = np.random.uniform(7, 12)

        x_res, fun = run_optimization(x0)
        
        L_res = x_res[39]
        
        if L_res < best_L:
            best_L = L_res
            best_result = x_res
            
        restarts += 1

    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_params = best_result[:36].reshape(12, 3)
    xc, yc, phi = best_result[36], best_result[37], best_result[38]
    L = best_result[39]
    
    inner = []
    for i in range(12):
        xi, yi, thi = inner_params[i]
        angle_deg = np.degrees(thi) % 360
        inner.append([xi, yi, angle_deg])
        
    return inner, (xc, yc), L, np.degrees(phi)
