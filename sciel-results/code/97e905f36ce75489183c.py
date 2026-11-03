import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s kill
    
    # Precompute unit hexagon vertices (side 1, center 0,0, angle 0)
    unit_vertices = np.array([
        [math.cos(2 * math.pi * k / 6), math.sin(2 * math.pi * k / 6)] for k in range(6)
    ])
    
    def get_min_side_length(centers, phi):
        max_proj = 0.0
        for i in range(N):
            verts = unit_vertices + centers[i]
            for k in range(3):
                angle = phi + k * math.pi / 3
                n = np.array([math.cos(angle), math.sin(angle)])
                projs = np.dot(verts, n)
                max_abs_proj = np.max(np.abs(projs))
                if max_abs_proj > max_proj:
                    max_proj = max_abs_proj
        return (2.0 / math.sqrt(3.0)) * max_proj

    def objective(params):
        centers = params[:2*N].reshape(N, 2)
        phi = params[2*N]
        
        S = get_min_side_length(centers, phi)
        
        penalty = 0.0
        dist_thresh = math.sqrt(3.0)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < dist_thresh:
                    penalty += (dist_thresh - d) ** 2
        
        return S + 10.0 * penalty

    v1 = np.array([math.sqrt(3), 0])
    v2 = np.array([math.sqrt(3)/2, 1.5])
    
    def initial_guess():
        centers = np.zeros((N, 2))
        centers[0] = [0,0]
        for k in range(6):
            angle = k * math.pi / 3
            centers[k+1] = [math.sqrt(3)*math.cos(angle), math.sqrt(3)*math.sin(angle)]
        
        # 4 more: Symmetric placement
        # (1,1), (-1,1), (0,-2), (2,0) was asymmetric.
        # Let's try (1,1), (-1,2), (-2,1), (0,-2) ?
        # Actually, just place them at the "corners" of the next ring
        # (2,0), (1,1), (-1,1), (-2,0) ?
        # Let's just use a random valid perturbation of a dense cluster
        # We'll add 4 more at distance sqrt(3) from some of the ring 1 hexagons
        # To keep it simple, we just use the previous asymmetric guess and rely on optimization
        ring2_candidates = [
            (1, 1), (-1, 1), (0, -2), (2, 0) 
        ]
        for k in range(4):
            i, j = ring2_candidates[k]
            centers[k+7] = i*v1 + j*v2
            
        centers += np.random.uniform(-0.2, 0.2, centers.shape)
        phi = np.random.uniform(0, math.pi/3)
        
        params = np.concatenate([centers.flatten(), [phi]])
        return params

    best_score = float('inf')
    best_params = None
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        x0 = initial_guess()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_score:
                centers = res.x[:2*N].reshape(N, 2)
                phi = res.x[2*N]
                
                valid = True
                dist_thresh = math.sqrt(3.0)
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < dist_thresh - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S = get_min_side_length(centers, phi)
                    if S < best_score:
                        best_score = S
                        best_params = res.x.copy()
                        
        except Exception:
            pass

    if best_params is None:
        x0 = initial_guess()
        best_params = x0
        best_score = objective(x0)

    centers = best_params[:2*N].reshape(N, 2)
    phi = best_params[2*N]
    
    inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
    angle_deg = math.degrees(phi)
    
    return inner, (0.0, 0.0), best_score, angle_deg
# EVOLVE-BLOCK-END
