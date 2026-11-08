"""Heilbronn triangles: maximize the minimum area of any triangle formed by 11 points in an equilateral triangle."""
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

def barycentric_to_cartesian(u, v):
    """
    Convert barycentric-like parameters (u, v) to Cartesian coordinates.
    Mapping: P = u * V0 + v * V1 + (1-u-v) * V2
    V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x = 0.5 + 0.5v - 0.5u
    y = sqrt(3)/2 * (1 - u - v)
    """
    x = 0.5 + 0.5 * v - 0.5 * u
    y = (SQRT3 / 2.0) * (1.0 - u - v)
    return np.column_stack((x, y))

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing Parameters
    T_start = 0.01
    T_end = 1e-6
    iterations = 20000
    alpha = (T_end / T_start) ** (1.0 / iterations)
    
    # Initial state: Random points inside triangle
    uv = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                uv[i] = [u, v]
                break
    
    points = barycentric_to_cartesian(uv[:, 0], uv[:, 1])
    current_area = min_triangle_area(points)
    current_energy = -current_area
    best_energy = current_energy
    best_uv = uv.copy()
    
    T = T_start
    
    for i in range(iterations):
        idx = rng.integers(0, N)
        
        step_size = 0.1 * (T / T_start) + 0.001
        du = rng.normal(0, step_size)
        dv = rng.normal(0, step_size)
        
        new_u = uv[idx, 0] + du
        new_v = uv[idx, 1] + dv
        
        reflected_u = new_u
        reflected_v = new_v
        
        if reflected_u < 0:
            reflected_u = -reflected_u
        if reflected_v < 0:
            reflected_v = -reflected_v
        if reflected_u + reflected_v > 1:
            excess = reflected_u + reflected_v - 1
            reflected_u -= excess
            reflected_v -= excess
            if reflected_u < 0: reflected_u = 0
            if reflected_v < 0: reflected_v = 0
            if reflected_u + reflected_v > 1:
                s = 1.0 / (reflected_u + reflected_v)
                reflected_u *= s
                reflected_v *= s
        
        old_u = uv[idx, 0]
        old_v = uv[idx, 1]
        
        uv[idx, 0] = reflected_u
        uv[idx, 1] = reflected_v
        
        points = barycentric_to_cartesian(uv[:, 0], uv[:, 1])
        new_area = min_triangle_area(points)
        new_energy = -new_area
        
        delta_e = new_energy - current_energy
        if delta_e < 0:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_uv = uv.copy()
        else:
            prob = np.exp(-delta_e / T)
            if rng.random() < prob:
                current_energy = new_energy
            else:
                uv[idx, 0] = old_u
                uv[idx, 1] = old_v
        
        T *= alpha
    
    # Polish with Nelder-Mead on the best configuration found by SA
    best_points_uv = best_uv.copy()
    x0 = best_points_uv.flatten()
    
    def penalized_objective(x):
        uv_opt = x.reshape((N, 2))
        penalty = 0.0
        for i in range(N):
            u, v = uv_opt[i]
            if u < 0 or v < 0 or u + v > 1:
                penalty += 10.0 * (max(0, -u)**2 + max(0, -v)**2 + max(0, u+v-1)**2)
        
        if penalty > 1e-6:
            return penalty - min_triangle_area(barycentric_to_cartesian(uv_opt[:, 0], uv_opt[:, 1]))
        
        points = barycentric_to_cartesian(uv_opt[:, 0], uv_opt[:, 1])
        return -min_triangle_area(points)

    try:
        result = minimize(
            penalized_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        uv_res = result.x.reshape((N, 2))
        valid = True
        for i in range(N):
            u, v = uv_res[i]
            if u < -1e-9 or v < -1e-9 or u + v > 1 + 1e-9:
                valid = False
                break
        
        if valid:
            for i in range(N):
                uv_res[i, 0] = np.clip(uv_res[i, 0], 0, 1)
                uv_res[i, 1] = np.clip(uv_res[i, 1], 0, 1)
                if uv_res[i, 0] + uv_res[i, 1] > 1:
                    s = 1.0 / (uv_res[i, 0] + uv_res[i, 1])
                    uv_res[i, 0] *= s
                    uv_res[i, 1] *= s
            
            final_points = barycentric_to_cartesian(uv_res[:, 0], uv_res[:, 1])
            final_area = min_triangle_area(final_points)
            
            if final_area > -best_energy:
                best_uv = uv_res
    except Exception:
        pass
    
    final_points = barycentric_to_cartesian(best_uv[:, 0], best_uv[:, 1])
    return final_points
