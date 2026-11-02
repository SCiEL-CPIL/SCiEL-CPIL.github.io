import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # We will try multiple random initializations
    # 26 circles -> 78 variables (x, y, r for each)
    
    def objective(params):
        # Maximize sum of radii -> Minimize negative sum
        r = params[2::3]
        return -np.sum(r)

    def constraints(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        cons = []
        
        # Boundary constraints:
        # x_i >= r_i  => x_i - r_i >= 0
        # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        # y_i >= r_i => y_i - r_i >= 0
        # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
        
        # We can vectorize these for efficiency in the constraint function if needed,
        # but SLSQP handles arrays well if we return a dictionary or list.
        # Let's return a list of dictionaries for clarity, or a single array if we can.
        # SLSQP accepts a list of dicts with 'type': 'ineq' and 'fun': lambda.
        
        # To be efficient, let's define the constraint functions separately or use a single vectorized function?
        # SLSQP 'fun' for constraints can return an array.
        # Let's use a single constraint function that returns an array of all inequality constraints >= 0.
        
        # 1. x_i - r_i >= 0
        c_x_min = x - r
        # 2. 1 - x_i - r_i >= 0
        c_x_max = 1.0 - x - r
        # 3. y_i - r_i >= 0
        c_y_min = y - r
        # 4. 1 - y_i - r_i >= 0
        c_y_max = 1.0 - y - r
        
        # 5. Pairwise non-overlap:
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        # This is non-linear.
        
        # We can compute all pairwise constraints at once.
        # Distance matrix
        # x shape (N,), y shape (N,)
        # dx = x[:, None] - x[None, :]
        # dy = y[:, None] - y[None, :]
        # dist_sq = dx**2 + dy**2
        # r_sum = r[:, None] + r[None, :]
        # overlap_constr = dist_sq - r_sum**2
        # We need the upper triangle (i < j) to avoid duplicates and self-interaction (which is 0).
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        r_sum = r[:, None] + r[None, :]
        overlap_constr = dist_sq - r_sum**2
        
        # Get indices for i < j
        iu = np.triu_indices(N, k=1)
        pair_constr = overlap_constr[iu]
        
        # Combine all constraints
        # All must be >= 0
        all_constr = np.concatenate([c_x_min, c_x_max, c_y_min, c_y_max, pair_constr])
        
        return all_constr

    # Note: SLSQP expects constraints as a list of dictionaries or a single dictionary.
    # If we return a single array from 'fun', we must specify 'type': 'ineq' and it implies all elements >= 0.
    # However, SLSQP's 'fun' for constraints typically expects a scalar or a vector.
    # If we use a vector, it treats each element as a separate constraint.
    
    cons = ({
        'type': 'ineq',
        'fun': constraints
    },)
    
    # Bounds: r_i > 0, x,y in [0,1]
    # x,y bounds are handled by constraints, but we can also add them to bounds for stability.
    # x in [0,1], y in [0,1], r in [0, 0.5] (max radius in unit square is 0.5)
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Multi-start strategy
    # We'll generate several random feasible initial guesses.
    # A simple feasible guess: place circles in a grid with small radii.
    
    num_starts = 20 # Max number of random starts to try
    
    for k in range(num_starts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        # Grid layout
        side = int(np.ceil(np.sqrt(N)))
        # Create grid points
        xs = []
        ys = []
        for i in range(side):
            for j in range(side):
                if len(xs) < N:
                    xs.append((i + 0.5) / side)
                    ys.append((j + 0.5) / side)
        
        # Add random jitter
        jitter = 0.05
        x0 = np.array(xs) + np.random.uniform(-jitter, jitter, N)
        y0 = np.array(ys) + np.random.uniform(-jitter, jitter, N)
        
        # Clip to [0,1]
        x0 = np.clip(x0, 0, 1)
        y0 = np.clip(y0, 0, 1)
        
        # Initial radii: small enough to not overlap in the grid
        # Distance between grid points is 1/side.
        # Max radius for no overlap is 0.5/side.
        # Let's start with 0.4/side to be safe.
        r0 = np.full(N, 0.4 / side)
        
        # Flatten to params: [x0, y0, r0, x1, y1, r1, ...]
        params0 = np.zeros(3 * N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        # Check if initial guess is feasible (it should be, but just in case)
        # If not, we might need to shrink radii further.
        
        # Run SLSQP
        # maxiter needs to be sufficient but not too high to avoid long hangs.
        # 200 iterations per start * 20 starts = 4000 iterations.
        # Each iteration involves solving a QP.
        # Let's try maxiter=100 first to see speed.
        
        try:
            res = minimize(
                objective, 
                params0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.success or res.status == 0:
                # Check validity of result
                # SLSQP might still have slight violations due to numerical tolerance.
                # We should verify and potentially repair or discard.
                # For now, let's assume SLSQP is robust enough.
                
                score = -res.fun
                if score > best_score:
                    best_score = score
                    # Reconstruct array
                    x_res = res.x[0::3]
                    y_res = res.x[1::3]
                    r_res = res.x[2::3]
                    
                    # Verify constraints strictly to avoid invalid submission
                    # If any constraint is violated by more than a tiny epsilon, 
                    # we might need to shrink radii slightly.
                    # Let's do a quick check.
                    valid = True
                    # Boundaries
                    if np.any(x_res < r_res - 1e-6) or np.any(x_res > 1 - r_res + 1e-6):
                        valid = False
                    if np.any(y_res < r_res - 1e-6) or np.any(y_res > 1 - r_res + 1e-6):
                        valid = False
                    # Overlaps
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = x_res[i] - x_res[j]
                                dy = y_res[i] - y_res[j]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < r_res[i] + r_res[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        best_result = np.column_stack((x_res, y_res, r_res))
                    else:
                        # If invalid, try to repair by scaling down radii?
                        # Or just discard. Discarding is safer.
                        pass
                        
        except Exception as e:
            # Skip this start if it fails
            continue
            
    if best_result is not None:
        return best_result
    else:
        # Fallback to a known valid configuration (e.g., the grid seed)
        # This ensures we always return something valid, even if suboptimal.
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-4
                    cells.append((x, y, r))
        return np.array(cells)

# EVOLVE-BLOCK-END
