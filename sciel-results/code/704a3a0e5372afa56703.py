import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_cluster_points():
    pts = [(0.0, 0.0)]
    for i in range(6):
        angle = i * np.pi / 3
        pts.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
    for i in range(5):
        angle = (2*i + 1) * np.pi / 6
        pts.append((3.0 * np.cos(angle), 3.0 * np.sin(angle)))
    return np.array(pts)

CLUSTER_PTS = get_cluster_points()

def construct():
    start_time = time.time()
    deadline = 140.0
    
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.5]
    
    def objective(params):
        x_c, y_c, theta_in, x_o, y_o, theta_out, S = params
        
        cos_in = np.cos(theta_in)
        sin_in = np.sin(theta_in)
        rot_in = np.array([[cos_in, -sin_in], [sin_in, cos_in]])
        centers = (CLUSTER_PTS @ rot_in.T) + np.array([x_c, y_c])
        
        angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + theta_out
        normals = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
        d_side = S * np.sqrt(3) / 2
        
        penalty = 0.0
        for i in range(N):
            cx, cy = centers[i]
            angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False) + theta_in
            verts = np.column_stack((np.cos(angles_in), np.sin(angles_in)))
            verts += np.array([cx, cy])
            
            for vx, vy in verts:
                dx = vx - x_o
                dy = vy - y_o
                dots = normals @ np.array([dx, dy])
                viol = np.maximum(0, dots - d_side)
                penalty += np.sum(viol**2)
                
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < np.sqrt(3):
                    penalty += (np.sqrt(3) - dist)**2 * 100.0
        
        return S + penalty

    best_params = x0
    best_score = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass

    if time.time() < deadline - 30:
        x0_2 = [0.0, 0.0, np.pi/6, 0.0, 0.0, np.pi/6, 5.0]
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        except Exception:
            pass

    x_c, y_c, theta_in, x_o, y_o, theta_out, S = best_params
    
    cos_in = np.cos(theta_in)
    sin_in = np.sin(theta_in)
    rot_in = np.array([[cos_in, -sin_in], [sin_in, cos_in]])
    centers = (CLUSTER_PTS @ rot_in.T) + np.array([x_c, y_c])
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(np.degrees(theta_in))])
        
    return inner, (float(x_o), float(y_o)), float(S), float(np.degrees(theta_out))
