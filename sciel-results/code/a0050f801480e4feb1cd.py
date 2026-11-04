import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0 # Safety margin

# EVOLVE-BLOCK-START
def construct():
    # Initial guesses
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2.0)
        return x, y

    # Guess 1: 3x4 grid minus 1
    coords1 = []
    for r in range(-1, 3):
        for q in range(-1, 2):
            coords1.append((q, r))
    # Remove one to make 11
    coords1 = coords1[:11]
    
    # Guess 2: 7-cluster + 4
    coords2 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (2,0), (2,-1), (1,-2), (-1,2)]
    
    guesses = []
    for c in [coords1, coords2]:
        x0 = []
        for q, r in c:
            x, y = axial_to_xy(q, r)
            x0.extend([x, y])
        x0.append(3.5)
        guesses.append(x0)
        
    best_L = 40.0
    best_coords = None
    
    angles = np.pi/6 + np.arange(6) * np.pi/3
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def objective(v):
        L = v[-1]
        if L < 1: return 100.0
        coords = v[:-1].reshape(-1, 2)
        
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                d2 = np.sum((coords[i] - coords[j])**2)
                if d2 < 3.0:
                    penalty += (3.0 - d2) ** 2
        
        for i in range(N):
            for k in range(6):
                proj = np.dot(coords[i], normals[k])
                limit = np.sqrt(3)/2 * (L - 1)
                if proj > limit:
                    penalty += (proj - limit) ** 2
                    
        return L + 10 * penalty

    for x0 in guesses:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            L_val = res.x[-1]
            if L_val < best_L:
                best_L = L_val
                best_coords = res.x[:-1].reshape(-1, 2)
        except Exception:
            pass
            
    if best_coords is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return [[x, y, 0.0] for x, y in best_coords], (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
