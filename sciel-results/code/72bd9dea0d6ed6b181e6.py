import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # 10 second margin
    
    # Initial configurations
    configs = []
    
    # Config 1: 7 cluster + 4 on sides
    pts1 = [(0.0, 0.0)]
    for k in range(6):
        angle = np.pi / 3 * k
        pts1.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
    for k in range(4):
        angle = np.pi / 3 * (2 * k + 1)
        pts1.append((2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)))
    configs.append(np.array(pts1))
    
    # Config 2: 3-4-3-1 grid?
    # Just use random perturbations of Config 1
    for _ in range(4):
        pts = configs[0].copy()
        pts += np.random.normal(0, 0.5, pts.shape)
        configs.append(pts)
        
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for pts in configs:
        if time.time() > deadline:
            break
            
        x0 = np.concatenate([pts.flatten(), [0.0]])
        
        def objective(x):
            p = x[:22].reshape(-1, 2)
            theta = x[22]
            angles = theta + np.pi / 3 * np.arange(6)
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            proj = np.max(np.dot(p, normals.T))
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < np.sqrt(3):
                        penalty += (np.sqrt(3) - d) ** 2
            return proj + 10.0 * penalty
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.success:
                p = res.x[:22].reshape(-1, 2)
                theta = res.x[22]
                angles = theta + np.pi / 3 * np.arange(6)
                normals = np.column_stack((np.cos(angles), np.sin(angles)))
                proj = np.max(np.dot(p, normals.T))
                S = 2.0 / np.sqrt(3) * proj
                
                # Check validity (no overlap)
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if np.linalg.norm(p[i] - p[j]) < np.sqrt(3) - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                            
                if valid and S < best_S:
                    best_S = S
                    best_inner = [[p[i][0], p[i][1], 0.0] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = np.degrees(theta)
        except Exception:
            pass
            
    if best_inner is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
