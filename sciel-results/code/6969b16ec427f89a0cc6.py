"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Variables: w (width), then for each circle: x, y, r
    # h = 2 - w
    # Total variables: 1 + 3*N
    
    def penalty_and_objective(v):
        w = v[0]
        h = 2.0 - w
        
        # Constraints on w, h
        if w <= 0 or h <= 0:
            return 1e10
            
        xs = v[1::3]
        ys = v[2::3]
        rs = v[3::3]
        
        # Radii must be positive
        if np.any(rs <= 0):
            return 1e10
            
        penalty = 0.0
        
        # Boundary constraints
        # Circle i must be inside [0, w] x [0, h]
        for i in range(N):
            # x bounds
            if xs[i] < rs[i]:
                penalty += (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]:
                penalty += (xs[i] - (w - rs[i])) ** 2
            # y bounds
            if ys[i] < rs[i]:
                penalty += (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]:
                penalty += (ys[i] - (h - rs[i])) ** 2
        
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist:
                    # Penalty for violation
                    # Using squared difference of distances is smoother than sqrt
                    dist = np.sqrt(dist_sq)
                    penalty += (min_dist - dist) ** 2
        
        # Objective: Maximize sum(rs) -> Minimize -sum(rs) + penalty
        # Scale penalty to ensure it dominates violations
        return -np.sum(rs) + 1000.0 * penalty

    best_val = -1.0
    best_sol = None
    start_time = time.time()
    
    # Try multiple random restarts with different initial aspect ratios
    # We will sample aspect ratios and use Nelder-Mead which is good for this kind of problem
    # despite being slower per step, it handles the non-smooth penalty landscape better
    # when combined with restarts.
    
    np.random.seed(42)
    
    # Aspect ratios to try: w/h. Since w+h=2, w = 2*ar/(1+ar), h = 2/(1+ar)
    # Let's try a range of aspect ratios
    aspect_ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() - start_time > 100:
            break
            
        h_init = 2.0 / (1.0 + ar)
        w_init = 2.0 * ar / (1.0 + ar)
        
        # Initialize circles in a grid within the rectangle
        # Approximate grid size
        # Area per circle ~ 1/N. 
        # Let's just place them in a grid of 5x5 or similar, clipped to N
        cols = int(np.ceil(np.sqrt(N * (w_init/h_init))))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        x0 = np.zeros(3*N + 1)
        x0[0] = w_init
        
        # Grid initialization
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                # Center of cell
                x = (j + 0.5) * w_init / cols
                y = (i + 0.5) * h_init / rows
                # Initial radius: small fraction of cell size
                cell_w = w_init / cols
                cell_h = h_init / rows
                r = 0.4 * min(cell_w, cell_h)
                
                x0[1 + 3*idx] = x
                x0[2 + 3*idx] = y
                x0[3 + 3*idx] = r
                idx += 1
        
        # Fill any remaining slots (if N > rows*cols due to ceiling logic mismatch, though unlikely)
        # Actually rows*cols >= N is guaranteed by the loop above if we break correctly.
        # But we need to make sure we filled exactly N.
        # If idx < N, we missed some? No, rows*cols >= N.
        # If idx == N, we are good.
        
        # Add some jitter to help escape symmetry
        # Jitter positions slightly
        for i in range(1, 3*N+1, 3):
             x0[i] += np.random.uniform(-0.01, 0.01)
             x0[i+1] += np.random.uniform(-0.01, 0.01)
        
        # Nelder-Mead optimization
        # We use Nelder-Mead because it doesn't require gradients and can handle the
        # non-differentiable points of the penalty function better in some cases,
        # especially with the box constraints implicitly handled by the penalty.
        # However, Nelder-Mead can be slow. We limit iterations.
        
        res = minimize(penalty_and_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < 0: # Valid solution has negative objective (since -sum(rs) + penalty < 0 implies sum(rs) > penalty)
            # Check validity explicitly to be safe
            w = res.x[0]
            h = 2.0 - w
            xs = res.x[1::3]
            ys = res.x[2::3]
            rs = res.x[3::3]
            
            valid = True
            if w <= 0 or h <= 0:
                valid = False
            else:
                for i in range(N):
                    if rs[i] <= 0:
                        valid = False
                        break
                    if xs[i] < rs[i] - 1e-6 or xs[i] > w - rs[i] + 1e-6:
                        valid = False
                        break
                    if ys[i] < rs[i] - 1e-6 or ys[i] > h - rs[i] + 1e-6:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if dist < rs[i] + rs[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_sol = np.column_stack([xs, ys, rs])
    
    # Fallback if no valid solution found (should not happen with penalty method usually)
    if best_sol is None:
        # Simple grid fallback
        w = 1.0
        h = 1.0
        side = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                x = (j + 0.5) / 5.0
                y = (i + 0.5) / 5.0
                r = 0.4 / 5.0
                cells.append([x, y, r])
        return np.array(cells)
        
    return best_sol
# EVOLVE-BLOCK-END
