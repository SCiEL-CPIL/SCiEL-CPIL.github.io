"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def construct():
    # Perimeter = 4 => w + h = 2
    # We will try different aspect ratios
    
    best_value = -1
    best_result = None
    
    # Aspect ratios to try (w from 0.5 to 1.5)
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # 1. Generate a grid of candidate positions
        # We want enough candidates to choose from
        grid_n = 10
        xs = np.linspace(w / (2 * grid_n), w - w / (2 * grid_n), grid_n)
        ys = np.linspace(h / (2 * grid_n), h - h / (2 * grid_n), grid_n)
        
        # Create all grid points
        grid_points = np.array([(x, y) for x in xs for y in ys])
        M = len(grid_points)
        
        if M < N:
            continue
            
        # 2. "Matching" step: Assign N circles to N grid points
        # We want to maximize the potential radius. 
        # Heuristic: Maximize the minimum boundary distance? 
        # Or maximize sum of boundary distances?
        # Let's maximize the sum of boundary distances as a proxy for "good" positions.
        # This is a simple assignment problem.
        
        # Cost matrix: -boundary_distance (minimize cost = maximize distance)
        boundary_dists = np.min(grid_points, axis=1) # min(x, y) is not correct for rectangle
        # Correct boundary distance for point (x,y) in [0,w]x[0,h]:
        # min(x, w-x, y, h-y)
        bd = np.min(np.column_stack([grid_points[:, 0], w - grid_points[:, 0], 
                                     grid_points[:, 1], h - grid_points[:, 1]]), axis=1)
        
        # We need to assign N circles to M points.
        # Since circles are identical, we just want the N best points.
        # However, "best" depends on interactions.
        # For initialization, let's just pick the N points with largest boundary distance.
        # To use Hungarian as per directive:
        # Create a cost matrix of shape (N, M).
        # Cost for circle i to point j is -bd[j].
        # This will assign each circle to a distinct point with max bd.
        
        cost_matrix = -np.tile(bd, (N, 1))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # The selected points are grid_points[col_ind]
        # Note: linear_sum_assignment returns indices for min cost.
        # We need to ensure we get N distinct points.
        
        positions = grid_points[col_ind].copy()
        
        # 3. Local Optimization
        # Objective: Maximize sum of radii.
        # Radii are determined by LP given positions.
        
        def get_radii(pos):
            # Calculate pairwise distances
            # d[i,j] is distance between i and j
            d = cdist(pos, pos)
            # Boundary distances
            b = np.min(np.column_stack([pos[:, 0], w - pos[:, 0], 
                                        pos[:, 1], h - pos[:, 1]]), axis=1)
            
            # LP: Maximize sum(r)
            # s.t. r_i + r_j <= d_ij
            #      r_i <= b_i
            #      r_i >= 0
            
            c = -np.ones(N) # Minimize -sum(r)
            
            A_ub = []
            b_ub = []
            
            # Pairwise constraints
            for i in range(N):
                for j in range(i + 1, N):
                    # r_i + r_j <= d_ij
                    row = np.zeros(N)
                    row[i] = 1
                    row[j] = 1
                    A_ub.append(row)
                    b_ub.append(d[i, j])
            
            # Boundary constraints
            # r_i <= b_i
            A_ub.append(np.eye(N))
            b_ub.extend(b)
            
            # Bounds: r_i >= 0
            bounds = [(0, None)] * N
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    return -res.fun, -res.x # return value and radii
                else:
                    return -1e9, np.zeros(N)
            except:
                return -1e9, np.zeros(N)
        
        # Initial evaluation
        val, radii = get_radii(positions)
        
        # Optimization using Nelder-Mead
        # We minimize -val
        def objective(pos_flat):
            pos = pos_flat.reshape(N, 2)
            # Keep points inside rectangle with small margin to avoid huge penalties
            # If a point goes outside, the LP might fail or give 0.
            # We should penalize out-of-bounds.
            # Actually, the LP handles boundary via r_i <= b_i. If pos is outside, b_i < 0, so r_i <= negative, which with r_i>=0 is infeasible.
            # So LP will fail if any point is outside.
            # We need to ensure points stay inside.
            # We can add a penalty to the objective if points go outside.
            
            # Check bounds
            margin = 1e-5
            if np.any(pos[:, 0] < margin) or np.any(pos[:, 0] > w - margin) or \
               np.any(pos[:, 1] < margin) or np.any(pos[:, 1] > h - margin):
                # Calculate penalty: sum of squared violations
                violations = 0
                for i in range(N):
                    if pos[i, 0] < margin: violations += (margin - pos[i, 0])**2
                    if pos[i, 0] > w - margin: violations += (pos[i, 0] - (w - margin))**2
                    if pos[i, 1] < margin: violations += (margin - pos[i, 1])**2
                    if pos[i, 1] > h - margin: violations += (pos[i, 1] - (h - margin))**2
                return 1e9 + violations # High penalty
            
            v, _ = get_radii(pos)
            return -v
        
        # Run minimization
        # Limit iterations to ensure speed
        try:
            res_opt = minimize(objective, positions.flatten(), method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res_opt.success or (res_opt.fun < -val): # If it improved
                final_pos = res_opt.x.reshape(N, 2)
                final_val, final_radii = get_radii(final_pos)
                
                if final_val > best_value:
                    best_value = final_val
                    # Construct result array
                    result = np.zeros((N, 3))
                    result[:, 0] = final_pos[:, 0]
                    result[:, 1] = final_pos[:, 1]
                    result[:, 2] = final_radii
                    best_result = result
                    
        except Exception as e:
            pass

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
