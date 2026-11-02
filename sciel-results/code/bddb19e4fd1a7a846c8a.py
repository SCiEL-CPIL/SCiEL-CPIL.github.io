"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_x = None
    best_score = 0.0

    # Helper to check validity and score
    def _compute_score(x):
        if x is None:
            return 0.0
        # Check boundary constraints
        for i in range(N):
            x_i, y_i, r_i = x[i]
            if x_i - r_i < -1e-9 or y_i - r_i < -1e-9 or x_i + r_i > 1 + 1e-9 or y_i + r_i > 1 + 1e-9:
                return 0.0
        # Check pairwise distance constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = x[i, 2] + x[j, 2]
                if dist < min_dist - 1e-9:
                    return 0.0
        return np.sum(x[:, 2])

    def _relax(x_init, max_iters=500, time_limit=None):
        """
        Iterative relaxation to find a locally optimal packing.
        Strategy: 
        1. Push apart overlapping circles.
        2. Maximize radii given current positions.
        """
        x = x_init.copy()
        n = N
        
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            
            # Step 1: Resolve overlaps by pushing apart
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j, 0] - x[i, 0]
                    dy = x[j, 1] - x[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = x[i, 2] + x[j, 2]
                    
                    if dist < min_dist:
                        if dist < 1e-9:
                            # Random perturbation
                            x[i, 0] += random.uniform(-0.01, 0.01)
                            x[i, 1] += random.uniform(-0.01, 0.01)
                            x[j, 0] += random.uniform(-0.01, 0.01)
                            x[j, 1] += random.uniform(-0.01, 0.01)
                        else:
                            # Push apart along the vector
                            dx_unit = dx / dist
                            dy_unit = dy / dist
                            overlap = min_dist - dist
                            # Push each by half the overlap
                            push = overlap / 2
                            x[i, 0] -= dx_unit * push
                            x[i, 1] -= dy_unit * push
                            x[j, 0] += dx_unit * push
                            x[j, 1] += dy_unit * push
            
            # Step 2: Enforce boundary constraints by adjusting positions and radii
            for i in range(n):
                # Clip positions to stay inside [0,1] with margin for radius
                # First, ensure position is within [0,1]
                x[i, 0] = np.clip(x[i, 0], 0.0, 1.0)
                x[i, 1] = np.clip(x[i, 1], 0.0, 1.0)
                
                # Calculate max possible radius based on boundaries
                max_r_bound = min(x[i, 0], x[i, 1], 1.0 - x[i, 0], 1.0 - x[i, 1])
                
                # Calculate max possible radius based on neighbors
                max_r_neigh = max_r_bound
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i, 0] - x[j, 0]
                    dy = x[i, 1] - x[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    # r_i + r_j <= dist => r_i <= dist - r_j
                    max_r_neigh = min(max_r_neigh, dist - x[j, 2])
                
                # Update radius to be as large as possible, but keep it positive
                x[i, 2] = max(0.0, max_r_neigh)
            
            # Step 3: Optional - Small random perturbation to escape local minima
            if it % 50 == 0 and it > 0:
                # Only perturb if we seem stuck? For now, small perturbation
                # Actually, perturbing positions can break valid packings.
                # Let's rely on the multi-start instead.
                pass

        return x

    def _polish(x_init, time_limit=None):
        """
        Use SLSQP to polish the solution.
        """
        if time_limit and time.time() > time_limit:
            return x_init
            
        x_flat = x_init.flatten()
        
        # Objective: Maximize sum of radii
        def objective(x):
            return -np.sum(x[2*N:3*N])
            
        # Constraints
        # We define constraints as functions returning g(x) >= 0
        # Boundary constraints
        cons = []
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
            # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i] - x[2*N+i]})
            # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[N+i] - x[2*N+i]})
            
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[N+i]-x[N+j])**2) - x[2*N+i] - x[2*N+j]
                })
                
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        try:
            # SLSQP is good for small problems with many constraints
            # L-BFGS-B is faster but doesn't handle constraints as well for this structure
            result = minimize(
                objective,
                x_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if result.success:
                x_opt = result.x.reshape(N, 3)
                # Final clipping for safety
                x_opt[:, :2] = np.clip(x_opt[:, :2], 0.0, 1.0)
                x_opt[:, 2] = np.clip(x_opt[:, 2], 0.0, 0.5)
                return x_opt
            else:
                return x_init
        except Exception:
            return x_init

    # Generate initial configurations
    def get_grid_init():
        side = 5
        x = np.zeros((N, 3))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                x[idx, 0] = (i + 0.5) / side
                x[idx, 1] = (j + 0.5) / side
                x[idx, 2] = 0.5 / side - 0.01
                idx += 1
        # If N is not a perfect square, adjust the last one or use a different grid
        if N % side != 0:
            # Simple fix: just place the rest randomly or adjust
            pass
        return x

    def get_random_init():
        x = np.zeros((N, 3))
        x[:, 0] = np.random.uniform(0.1, 0.9, N)
        x[:, 1] = np.random.uniform(0.1, 0.9, N)
        x[:, 2] = 0.05
        return x

    def get_hex_init():
        # Approximate hexagonal packing
        x = np.zeros((N, 3))
        rows = 5
        cols = 5
        idx = 0
        for r in range(rows):
            offset = 0 if r % 2 == 0 else 0.5
            for c in range(cols):
                if idx >= N: break
                x[idx, 0] = (c + offset + 0.5) / cols
                x[idx, 1] = (r + 0.5) / rows
                x[idx, 2] = 0.5 / cols - 0.01
                idx += 1
        return x

    initializers = [get_grid_init, get_random_init, get_hex_init]
    
    # Run Multi-Start
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 30: # Leave time for polishing
            break
            
        if i < len(initializers):
            x_init = initializers[i]()
        else:
            # Use random or perturbed grid
            x_init = get_random_init()
            # Or perturb the best found so far?
            if best_x is not None:
                x_init = best_x.copy()
                x_init[:, :2] += np.random.normal(0, 0.05, (N, 2))
                x_init[:, :2] = np.clip(x_init[:, :2], 0.0, 1.0)
                x_init[:, 2] = 0.05 # Reset radii to small to allow re-packing

        # Phase 1: Relaxation
        x_relaxed = _relax(x_init, max_iters=100, time_limit=deadline - 20)
        
        # Phase 2: Polish
        x_polished = _polish(x_relaxed, time_limit=deadline - 5)
        
        score = _compute_score(x_polished)
        if score > best_score:
            best_score = score
            best_x = x_polished
            # print(f"Start {i}: Score {score:.4f}") # Disabled for eval

    # Final check
    if best_x is None:
        # Fallback to simple grid
        best_x = get_grid_init()
        
    return best_x

# EVOLVE-BLOCK-END
