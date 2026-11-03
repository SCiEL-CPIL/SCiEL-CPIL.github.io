import numpy as np
import time
from scipy.optimize import linprog

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # We will try several aspect ratios
    # W + H = 2
    # W in [0.5, 1.5]
    aspect_ratios = np.linspace(0.5, 1.5, 15)
    
    # For each aspect ratio, we run the SCR loop
    for W_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        H_init = 2.0 - W_init
        
        # Initialize with a grid
        # 5x5 grid for 21 circles (last row has 1)
        cols = 5
        rows = 5
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            xs[i] = (col + 0.5) * (W_init / cols)
            ys[i] = (row + 0.5) * (H_init / rows)
            # Small initial radius
            rs[i] = 0.01
            
        # Variable vector for LP:
        # [x_1, ..., x_N, y_1, ..., y_N, r_1, ..., r_N, W]
        # Size: 3N + 1
        
        # LP Objective: Minimize -sum(r_i)
        # c vector: 0 for x, 0 for y, -1 for r, 0 for W
        c = np.zeros(3 * N + 1)
        c[2 * N : 3 * N] = -1.0
        
        # Bounds for variables
        # x_i in [0, 2], y_i in [0, 2], r_i in [0, 1], W in [0.1, 1.9]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((0.0, 1.0)) # r
        bounds.append((0.1, 1.9)) # W
        
        # We will iterate SCR
        current_x = xs.copy()
        current_y = ys.copy()
        current_r = rs.copy()
        current_W = W_init
        
        max_iter = 20
        prev_score = -np.inf
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Build A_ub and b_ub for linear constraints
            # We want A_ub @ x <= b_ub
            # 
            # Wall Constraints:
            # x_i - r_i >= 0  =>  -x_i + r_i <= 0
            # -x_i - r_i + W >= 0 => x_i + r_i - W <= 0
            # y_i - r_i >= 0  =>  -y_i + r_i <= 0
            # -y_i - r_i + (2-W) >= 0 => y_i + r_i + W <= 2
            
            # Collision Constraints (Linearized):
            # n_ij @ (p_i - p_j) >= r_i + r_j
            # n_ij_x * (x_i - x_j) + n_ij_y * (y_i - y_j) - r_i - r_j >= 0
            # => -n_ij_x * x_i + n_ij_x * x_j - n_ij_y * y_i + n_ij_y * y_j + r_i + r_j <= 0
            
            # Let's assemble the constraint matrix.
            # Number of wall constraints: 4 * N
            # Number of collision constraints: N * (N-1) / 2 = 210
            
            n_wall = 4 * N
            n_coll = N * (N - 1) // 2
            total_constraints = n_wall + n_coll
            
            A_ub = np.zeros((total_constraints, 3 * N + 1))
            b_ub = np.zeros(total_constraints)
            
            # Fill Wall Constraints
            row = 0
            for i in range(N):
                idx_x = i
                idx_y = N + i
                idx_r = 2 * N + i
                idx_W = 3 * N
                
                # -x_i + r_i <= 0
                A_ub[row, idx_x] = -1.0
                A_ub[row, idx_r] = 1.0
                b_ub[row] = 0.0
                row += 1
                
                # x_i + r_i - W <= 0
                A_ub[row, idx_x] = 1.0
                A_ub[row, idx_r] = 1.0
                A_ub[row, idx_W] = -1.0
                b_ub[row] = 0.0
                row += 1
                
                # -y_i + r_i <= 0
                A_ub[row, idx_y] = -1.0
                A_ub[row, idx_r] = 1.0
                b_ub[row] = 0.0
                row += 1
                
                # y_i + r_i + W <= 2
                A_ub[row, idx_y] = 1.0
                A_ub[row, idx_r] = 1.0
                A_ub[row, idx_W] = 1.0
                b_ub[row] = 2.0
                row += 1
            
            # Fill Collision Constraints
            for i in range(N):
                for j in range(i + 1, N):
                    # Compute normal from current positions
                    dx = current_x[i] - current_x[j]
                    dy = current_y[i] - current_y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < 1e-6:
                        # If centers coincide, pick a random normal
                        angle = np.random.uniform(0, 2 * np.pi)
                        nx = np.cos(angle)
                        ny = np.sin(angle)
                    else:
                        nx = dx / dist
                        ny = dy / dist
                    
                    idx_xi = i
                    idx_xj = j
                    idx_yi = N + i
                    idx_yj = N + j
                    idx_ri = 2 * N + i
                    idx_rj = 2 * N + j
                    
                    # -nx * x_i + nx * x_j - ny * y_i + ny * y_j + r_i + r_j <= 0
                    A_ub[row, idx_xi] = -nx
                    A_ub[row, idx_xj] = nx
                    A_ub[row, idx_yi] = -ny
                    A_ub[row, idx_yj] = ny
                    A_ub[row, idx_ri] = 1.0
                    A_ub[row, idx_rj] = 1.0
                    b_ub[row] = 0.0
                    row += 1
            
            # Solve LP
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            except Exception:
                break
                
            if not res.success:
                break
                
            x_sol = res.x
            
            # Extract solution
            new_x = x_sol[0:N]
            new_y = x_sol[N:2*N]
            new_r = x_sol[2*N:3*N]
            new_W = x_sol[3*N]
            
            # Check if this is a valid packing (non-linear check)
            valid = True
            H_sol = 2.0 - new_W
            
            if new_W <= 0 or H_sol <= 0:
                valid = False
            else:
                # Wall check
                for i in range(N):
                    if new_x[i] - new_r[i] < -1e-6 or new_x[i] + new_r[i] > new_W + 1e-6:
                        valid = False
                        break
                    if new_y[i] - new_r[i] < -1e-6 or new_y[i] + new_r[i] > H_sol + 1e-6:
                        valid = False
                        break
                if valid:
                    # Collision check
                    for i in range(N):
                        for j in range(i+1, N):
                            dist_sq = (new_x[i] - new_x[j])**2 + (new_y[i] - new_y[j])**2
                            rad_sum = new_r[i] + new_r[j]
                            if dist_sq < rad_sum**2 - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                score = np.sum(new_r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([new_x, new_y, new_r])
                    
                # Update current points for next linearization
                current_x = new_x
                current_y = new_y
                current_r = new_r
                current_W = new_W
                
                # Convergence check
                if abs(score - prev_score) < 1e-6:
                    break
                prev_score = score
            else:
                # If invalid, the linearization might be too aggressive or we hit a local optimum
                # We can try to shrink radii slightly and continue, or break
                # Let's break to save time for other aspect ratios
                break
                
    # Fallback
    if best_result is None:
        # Simple grid in 1x1
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) < N:
                    x = (i + 0.5) / 5.0
                    y = (j + 0.5) / 5.0
                    r = 0.5 / 5.0 - 1e-3
                    cells.append([x, y, r])
        best_result = np.array(cells[:N])
        
    return best_result
