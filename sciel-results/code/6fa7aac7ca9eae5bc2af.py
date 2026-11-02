"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles.
Strategy: Simulated Annealing with softplus surrogate + LP polish.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay
import time

N = 21
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def get_delaunay_edges(centers):
        """Get Delaunay triangulation edges for circle centers."""
        try:
            if len(centers) < 3:
                return [(i, j) for i in range(len(centers)) for j in range(i+1, len(centers))]
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = tuple(sorted((simplex[i], simplex[j])))
                        edges.add(edge)
            return list(edges)
        except Exception:
            # Fallback: all pairs if Delaunay fails
            return [(i, j) for i in range(N) for j in range(i+1, N)]

    def solve_radii_lp(centers, w, h, edges):
        """Solve LP for maximum sum of radii given fixed centers and adjacency."""
        c = -np.ones(N)
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        A_ub_list = []
        b_ub_list = []
        
        # Boundary constraints
        for i in range(N):
            # r_i <= x_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(x[i])
            
            # r_i <= w - x_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(w - x[i])
            
            # r_i <= y_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(y[i])
            
            # r_i <= h - y_i
            row = np.zeros(N)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(h - y[i])
        
        # Non-overlap constraints
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(dist)
        
        if not A_ub_list:
            return np.zeros(N)
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        bounds = [(0, None)] * N
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                return np.maximum(result.x, 0.0)
            else:
                return np.zeros(N)
        except Exception:
            return np.zeros(N)

    def evaluate_solution(params):
        """
        params: [w, x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
        Returns: (score, penalty)
        score = sum(r)
        penalty = softplus violations
        """
        w = params[0]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return -1e10, 1e10
            
        x = params[1:1+N]
        y = params[1+N:1+2*N]
        r = params[1+2*N:1+3*N]
        
        # Ensure non-negative radii for calculation
        r_pos = np.maximum(r, 0.0)
        
        score = np.sum(r_pos)
        
        # Softplus penalty: log(1 + exp(k * violation))
        # Using a stable implementation
        def softplus(violations, k=50.0):
            # violations are negative when constraint is satisfied (e.g., x - r < 0 is violation)
            # We want to penalize positive violations.
            # Let v be the amount of violation (positive if violated).
            # Penalty = log(1 + exp(k * v))
            # If v is large positive, penalty ~ k*v
            # If v is large negative, penalty ~ 0
            # We compute v as the violation amount.
            
            # Boundary violations
            # Left: x - r < 0 => violation = r - x
            # Right: x + r > w => violation = x + r - w
            # Bottom: y - r < 0 => violation = r - y
            # Top: y + r > h => violation = y + r - h
            
            v_left = r_pos - x
            v_right = x + r_pos - w
            v_bottom = r_pos - y
            v_top = y + r_pos - h
            
            all_v = np.concatenate([v_left, v_right, v_bottom, v_top])
            
            # Circle-circle violations
            centers = np.column_stack([x, y])
            edges = get_delaunay_edges(centers)
            v_cc = []
            for (i, j) in edges:
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                v = r_pos[i] + r_pos[j] - dist
                v_cc.append(v)
            
            if v_cc:
                v_cc = np.array(v_cc)
                all_v = np.concatenate([all_v, v_cc])
            
            # Compute softplus
            # To avoid overflow, use:
            # if z > 0: z + log(1 + exp(-z))
            # else: log(1 + exp(z))
            z = k * all_v
            sp = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
            
            penalty = np.sum(sp)
            
            return penalty

        penalty = softplus_penalty = evaluate_penalty(x, y, r_pos, w, h)
        
        # The objective for minimization is -score + penalty
        # But we want to return score and penalty separately for tracking
        return score, penalty

    def evaluate_penalty(x, y, r, w, h):
        """Compute softplus penalty only."""
        k = 50.0
        
        v_left = r - x
        v_right = x + r - w
        v_bottom = r - y
        v_top = y + r - h
        
        all_v = np.concatenate([v_left, v_right, v_bottom, v_top])
        
        centers = np.column_stack([x, y])
        edges = get_delaunay_edges(centers)
        v_cc = []
        for (i, j) in edges:
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            v = r[i] + r[j] - dist
            v_cc.append(v)
        
        if v_cc:
            v_cc = np.array(v_cc)
            all_v = np.concatenate([all_v, v_cc])
            
        z = k * all_v
        sp = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
        
        return np.sum(sp)

    def polish(params):
        """Polish a candidate solution using LP."""
        w = params[0]
        h = 2.0 - w
        x = params[1:1+N]
        y = params[1+N:1+2*N]
        r = params[1+2*N:1+3*N]
        
        # Clip positions to be inside rectangle (approx)
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        centers = np.column_stack([x, y])
        edges = get_delaunay_edges(centers)
        
        r_opt = solve_radii_lp(centers, w, h, edges)
        
        # Construct new params
        new_params = np.zeros(1 + 3*N)
        new_params[0] = w
        new_params[1:1+N] = x
        new_params[1+N:1+2*N] = y
        new_params[1+2*N:1+3*N] = r_opt
        
        return new_params

    def is_valid(params, tol=1e-7):
        w = params[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        
        x = params[1:1+N]
        y = params[1+N:1+2*N]
        r = params[1+2*N:1+3*N]
        
        if np.any(r < -tol):
            return False
        
        r_pos = np.maximum(r, 0)
        
        # Check boundaries
        if np.any(x - r_pos < -tol) or np.any(x + r_pos > w + tol):
            return False
        if np.any(y - r_pos < -tol) or np.any(y + r_pos > h + tol):
            return False
        
        # Check overlaps
        centers = np.column_stack([x, y])
        edges = get_delaunay_edges(centers)
        for (i, j) in edges:
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r_pos[i] + r_pos[j] - tol:
                return False
                
        return True

    def get_score(params):
        if not is_valid(params):
            return -1e10
        r = params[1+2*N:1+3*N]
        return np.sum(np.maximum(r, 0))

    # Main SA Loop
    best_params = None
    best_score = -1e10
    
    # Initial guess: square, grid layout
    w0 = 1.0
    h0 = 1.0
    side = int(np.ceil(np.sqrt(N)))
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    
    for i in range(N):
        ix = i % side
        iy = i // side
        x0[i] = (ix + 0.5) * w0 / side
        y0[i] = (iy + 0.5) * h0 / side
        r0[i] = 0.5 / side * 0.9 # Slightly smaller to start
    
    params = np.zeros(1 + 3*N)
    params[0] = w0
    params[1:1+N] = x0
    params[1+N:1+2*N] = y0
    params[1+2*N:1+3*N] = r0
    
    # Polish initial guess
    params = polish(params)
    score = get_score(params)
    if score > best_score:
        best_score = score
        best_params = params.copy()
    
    current_params = params.copy()
    current_score = score
    
    # SA Parameters
    T_start = 0.05
    T_end = 0.001
    alpha = 0.9995 # Cooling rate
    
    # Estimate steps based on time
    # Each step involves Delaunay + Penalty calc. 
    # Let's do as many as possible.
    step_count = 0
    max_steps = 500000 # High cap, time will stop it
    
    while time.time() < DEADLINE:
        if step_count >= max_steps:
            break
            
        # Generate neighbor
        new_params = current_params.copy()
        
        # Perturb width
        if np.random.random() < 0.1:
            new_params[0] += np.random.randn() * 0.05
            new_params[0] = np.clip(new_params[0], 0.2, 1.8)
            
        # Perturb positions
        if np.random.random() < 0.8:
            idx = np.random.randint(N)
            new_params[1+idx] += np.random.randn() * 0.02
            new_params[1+N+idx] += np.random.randn() * 0.02
            
        # Perturb radii
        if np.random.random() < 0.8:
            idx = np.random.randint(N)
            new_params[1+2*N+idx] += np.random.randn() * 0.01
            new_params[1+2*N+idx] = max(0.001, new_params[1+2*N+idx])
            
        # Evaluate new
        # We use the penalized objective for SA acceptance, but track raw score for best
        # Objective to minimize: -score + penalty
        w_new = new_params[0]
        h_new = 2.0 - w_new
        if w_new <= 0.1 or h_new <= 0.1:
            continue
            
        x_new = new_params[1:1+N]
        y_new = new_params[1+N:1+2*N]
        r_new = new_params[1+2*N:1+3+N] # Typo fix: 1+2*N to 1+3*N
        
        # Recalculate penalty for current to compare? 
        # For efficiency, we can just compute new penalty and compare to current penalty if we stored it.
        # But we need to be careful. Let's store current penalty.
        
        # Actually, let's just compute the full objective for new.
        # And we need current objective.
        
        # Let's restructure to store current objective.
        
        # This block is inside the loop, so we need current_obj.
        # Let's initialize current_obj before loop.
        
        # Re-evaluating current_obj is expensive if done every step.
        # Let's assume we have current_obj from previous step.
        
        # For now, let's just compute new_penalty and new_score.
        new_score_raw = np.sum(np.maximum(r_new, 0))
        new_penalty = evaluate_penalty(x_new, y_new, np.maximum(r_new, 0), w_new, h_new)
        new_obj = -new_score_raw + new_penalty
        
        # We need current_obj. Let's calculate it once outside or keep it updated.
        # Let's update current_obj when we accept.
        
        # Wait, I didn't initialize current_obj. Let's fix the loop structure.
        
        # Break out and rewrite loop properly.
        break

    # Proper SA Loop
    # Initialize
    current_params = best_params.copy() if best_params is not None else params.copy()
    w_c = current_params[0]
    h_c = 2.0 - w_c
    x_c = current_params[1:1+N]
    y_c = current_params[1+N:1+2*N]
    r_c = current_params[1+2*N:1+3*N]
    current_score = np.sum(np.maximum(r_c, 0))
    current_penalty = evaluate_penalty(x_c, y_c, np.maximum(r_c, 0), w_c, h_c)
    current_obj = -current_score + current_penalty
    
    T = T_start
    step = 0
    max_steps = 1000000
    
    while time.time() < DEADLINE and step < max_steps:
        # Generate neighbor
        new_params = current_params.copy()
        
        # Perturb width (less frequent)
        if np.random.random() < 0.05:
            new_params[0] += np.random.randn() * 0.05
            new_params[0] = np.clip(new_params[0], 0.2, 1.8)
            
        # Perturb position
        idx = np.random.randint(N)
        new_params[1+idx] += np.random.randn() * 0.02
        new_params[1+N+idx] += np.random.randn() * 0.02
        
        # Perturb radius
        if np.random.random() < 0.5:
            new_params[1+2*N+idx] += np.random.randn() * 0.01
            new_params[1+2*N+idx] = max(0.001, new_params[1+2*N+idx])
            
        # Evaluate new
        w_new = new_params[0]
        h_new = 2.0 - w_new
        if w_new <= 0.1 or h_new <= 0.1:
            T *= alpha
            step += 1
            continue
            
        x_new = new_params[1:1+N]
        y_new = new_params[1+N:1+2*N]
        r_new = np.maximum(new_params[1+2*N:1+3*N], 0)
        
        new_score = np.sum(r_new)
        new_penalty = evaluate_penalty(x_new, y_new, r_new, w_new, h_new)
        new_obj = -new_score + new_penalty
        
        # Acceptance
        delta_obj = new_obj - current_obj
        if delta_obj < 0 or np.random.random() < np.exp(-delta_obj / T):
            # Accept
            current_params = new_params
            current_obj = new_obj
            current_score = new_score
            current_penalty = new_penalty
            
            # Check if better than best (only if valid or low penalty)
            # We can track best based on score if penalty is low, or just polish periodically
            # Let's polish every 500 steps to update best
            if step % 500 == 0:
                polished = polish(current_params)
                p_score = get_score(polished)
                if p_score > best_score:
                    best_score = p_score
                    best_params = polished.copy()
        
        T *= alpha
        step += 1
        
    # Final polish
    if best_params is not None:
        # Try a few more polishes from different aspects if time allows?
        # Just ensure best_params is polished
        final_params = polish(best_params)
        final_score = get_score(final_params)
        if final_score > best_score:
            best_score = final_score
            best_params = final_params

    # Construct output
    if best_params is not None:
        w = best_params[0]
        h = 2.0 - w
        x = best_params[1:1+N]
        y = best_params[1+N:1+2*N]
        r = best_params[1+2*N:1+3*N]
        
        result = np.zeros((N, 3))
        for i in range(N):
            result[i, 0] = x[i]
            result[i, 1] = y[i]
            result[i, 2] = max(0.001, r[i])
            
        return result
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        result = np.zeros((N, 3))
        for i in range(N):
            ix = i % side
            iy = i // side
            r = 0.5 / side - 0.001
            result[i, 0] = (ix + 0.5) * w / side
            result[i, 1] = (iy + 0.5) * h / side
            result[i, 2] = r
        return result

