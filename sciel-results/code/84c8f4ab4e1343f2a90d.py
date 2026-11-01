"""Circle packing in unit square n=32 using differentiable soft contact graph optimization."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 32

def construct():
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    xs = np.array([(i + 0.5) / side for i in range(side) for j in range(side)])
    ys = np.array([(j + 0.5) / side for j in range(side) for i in range(side)])
    rs = np.full(N, 0.5 / side - 1e-3)
    
    # Build initial positions
    x0 = np.concatenate([xs, ys, rs])
    
    # Number of variables: 3*N
    n_vars = 3 * N
    
    def objective(params):
        """Negative sum of radii (we minimize negative to maximize)"""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        return -np.sum(r)
    
    def constraints(params):
        """Return constraint values where >= 0 means feasible"""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Boundary constraints: circles must be inside [0,1]^2
        # x_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        bounds = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])
        
        # Pairwise distance constraints: dist(i,j) >= r_i + r_j
        # We use a soft constraint approach: penalize violations smoothly
        # For differentiability, we use max(0, r_i + r_j - dist)^2 style but return raw gap
        # Actually for SLSQP, we need g(params) >= 0
        # So we compute: dist(i,j) - r_i - r_j >= 0
        gaps = np.zeros(N*(N-1)//2)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                gaps[idx] = dist - r[i] - r[j]
                idx += 1
        
        return np.concatenate([bounds, gaps])
    
    # Define bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds_list = []
    for i in range(N):
        bounds_list.append((0, 1))  # x_i
        bounds_list.append((0, 1))  # y_i
        bounds_list.append((1e-6, 0.5))  # r_i
    
    # Use SLSQP with penalty-based approach for better performance
    # Actually, let's use a more robust approach with multiple restarts
    
    best_result = None
    best_value = -np.inf
    
    # Multiple random restarts
    np.random.seed(42)
    
    for restart in range(10):
        if restart == 0:
            x_start = xs.copy()
            y_start = ys.copy()
            r_start = rs.copy()
        else:
            # Perturb initial grid
            x_start = xs + np.random.uniform(-0.02, 0.02, N)
            y_start = ys + np.random.uniform(-0.02, 0.02, N)
            x_start = np.clip(x_start, 0.05, 0.95)
            y_start = np.clip(y_start, 0.05, 0.95)
            r_start = rs * (0.9 + 0.2 * np.random.random(N))
        
        x0_start = np.concatenate([x_start, y_start, r_start])
        
        try:
            res = minimize(
                objective,
                x0_start,
                method='SLSQP',
                bounds=bounds_list,
                constraints=[{'type': 'ineq', 'fun': constraints}],
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_value:
                # Verify constraints are satisfied
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                
                # Check if all constraints are satisfied
                valid = True
                # Boundary check
                if np.any(x_final - r_final < -1e-6) or np.any(1 - x_final - r_final < -1e-6):
                    valid = False
                if np.any(y_final - r_final < -1e-6) or np.any(1 - y_final - r_final < -1e-6):
                    valid = False
                
                # Pairwise check
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_final[i] - x_final[j]
                        dy = y_final[i] - y_final[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r_final[i] + r_final[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    value = np.sum(r_final)
                    if value > best_value:
                        best_value = value
                        best_result = res.x.copy()
                        
        except Exception as e:
            continue
    
    if best_result is not None:
        x = best_result[:N]
        y = best_result[N:2*N]
        r = best_result[2*N:3*N]
        # Clamp to ensure validity
        r = np.clip(r, 1e-6, 0.5)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # Final verification and fix
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-9:
                    # Reduce radii slightly to fix
                    overlap = min_dist - dist
                    reduction = overlap / 2 + 1e-9
                    r[i] = max(r[i] - reduction, 1e-6)
                    r[j] = max(r[j] - reduction, 1e-6)
        
        return np.column_stack([x, y, r])
    else:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
