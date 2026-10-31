"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Perimeter = 4 => w + h = 2
    # Maximize sum of radii for N=21 circles.
    
    best_sum = -1
    best_result = None
    
    # Try a range of aspect ratios
    aspect_ratios = np.linspace(0.2, 5.0, 15)
    
    for ratio in aspect_ratios:
        w = 2 * ratio / (1 + ratio)
        h = 2 / (1 + ratio)
        
        # Initialization: Grid
        grid_res = 10
        xs = np.linspace(0, w, grid_res)
        ys = np.linspace(0, h, grid_res)
        grid_points = np.array([[x, y] for y in ys for x in xs])
        
        # Greedy selection for max-min distance
        positions = np.zeros((N, 2))
        selected_indices = []
        
        for _ in range(N):
            best_idx = -1
            best_score = -1
            
            for idx, pt in enumerate(grid_points):
                if idx in selected_indices:
                    continue
                    
                db = min(pt[0], w - pt[0], pt[1], h - pt[1])
                
                if selected_indices:
                    sel_pts = grid_points[selected_indices]
                    dists = np.linalg.norm(sel_pts - pt, axis=1)
                    ds = np.min(dists)
                else:
                    ds = np.inf
                    
                score = min(db, ds)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            positions[len(selected_indices)-1] = grid_points[best_idx]
        
        if len(selected_indices) < N:
            continue
            
        # Objective function: Minimize -sum(radii)
        def objective(x):
            pos = x.reshape(N, 2)
            # Enforce boundaries
            pos[:, 0] = np.clip(pos[:, 0], 0, w)
            pos[:, 1] = np.clip(pos[:, 1], 0, h)
            
            radii = np.zeros(N)
            for i in range(N):
                min_dist = min(pos[i][0], w - pos[i][0], pos[i][1], h - pos[i][1])
                for j in range(N):
                    if i != j:
                        d = np.linalg.norm(pos[i] - pos[j])
                        min_dist = min(min_dist, d)
                radii[i] = min_dist / 2.0
            return -np.sum(radii)
        
        # Use Nelder-Mead for optimization
        x0 = positions.flatten()
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        pos = res.x.reshape(N, 2)
        pos[:, 0] = np.clip(pos[:, 0], 0, w)
        pos[:, 1] = np.clip(pos[:, 1], 0, h)
        
        # Calculate final radii
        radii = np.zeros(N)
        for i in range(N):
            min_dist = min(pos[i][0], w - pos[i][0], pos[i][1], h - pos[i][1])
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist = min(min_dist, d)
            radii[i] = min_dist / 2.0
            
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            result = np.zeros((N, 3))
            result[:, 0] = pos[:, 0]
            result[:, 1] = pos[:, 1]
            result[:, 2] = radii
            best_result = result
            
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
