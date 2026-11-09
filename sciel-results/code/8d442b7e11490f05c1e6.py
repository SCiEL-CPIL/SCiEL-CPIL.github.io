import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using Multi-start L-BFGS-B Local Optimization.
    
    Strategy:
    1. Define a smooth penalty function for overlaps and boundary violations.
    2. Use L-BFGS-B with analytic gradients to efficiently find local optima.
    3. Perform multiple restarts from diverse initial configurations.
    4. Track the best feasible solution found.
    """
    n = 10
    start_time = time.time()
    deadline = 600.0
    # Safety margin: stop 15 seconds before the hard kill to allow for cleanup and return
    hard_deadline = start_time + deadline - 15.0
    
    # Penalty weight for the optimization objective
    # We want to maximize sum(sides) subject to constraints.
    # Objective = -sum(sides) + Lambda * Penalty
    # Lambda should be large enough to prioritize feasibility, but not so large 
    # that it dominates the search for large side lengths once feasible.
    # However, since we track feasible solutions separately, we can use a moderate 
    # penalty to guide the search towards feasibility.
    LAMBDA = 100.0
    
    def smooth_max(x, eps=1e-6):
        """
        Smooth approximation of max(0, x).
        f(x) = 0.5 * (x + sqrt(x^2 + eps^2))
        f'(x) = 0.5 * (1 + x / sqrt(x^2 + eps^2))
        """
        sqrt_term = np.sqrt(x**2 + eps**2)
        val = 0.5 * (x + sqrt_term)
        # Gradient is needed for L-BFGS-B
        # We will compute gradients manually or use finite differences if not provided.
        # L-BFGS-B can use finite differences, but providing gradients is faster.
        # Let's define the objective and gradient together.
        return val

    def objective_and_grad(params):
        """
        params: array of shape (3n,)
        [cx0, cy0, s0, cx1, cy1, s1, ..., cx9, cy9, s9]
        
        Returns: (objective_value, gradient)
        Objective = -sum(sides) + LAMBDA * (Boundary_Penalty + Overlap_Penalty)
        """
        # Unpack
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        # 1. Boundary Penalty
        # x_min = cx - s/2 >= 0  => cx - s/2 >= 0
        # x_max = cx + s/2 <= 1  => cx + s/2 <= 1
        # y_min = cy - s/2 >= 0
        # y_max = cy + s/2 <= 1
        
        # Violation variables (positive if violated)
        vx_min = -(cx - s/2)
        vx_max = (cx + s/2) - 1
        vy_min = -(cy - s/2)
        vy_max = (cy + s/2) - 1
        
        # Smooth max of these violations
        # Sum of smooth_max(v)
        # grad w.r.t cx: 
        # d/dcx smooth_max(-(cx-s/2)) = -smooth_max_grad(vx_min)
        # d/dcx smooth_max(cx+s/2-1) = smooth_max_grad(vx_max)
        
        def sm_grad(v, eps=1e-6):
            sqrt_term = np.sqrt(v**2 + eps**2)
            return 0.5 * (1 + v / sqrt_term)
            
        g_vx_min = sm_grad(vx_min)
        g_vx_max = sm_grad(vx_max)
        g_vy_min = sm_grad(vy_min)
        g_vy_max = sm_grad(vy_max)
        
        boundary_pen = np.sum(smooth_max(vx_min)) + np.sum(smooth_max(vx_max)) + \
                       np.sum(smooth_max(vy_min)) + np.sum(smooth_max(vy_max))
        
        # Gradient of boundary penalty w.r.t params
        grad_cx = -LAMBDA * g_vx_min + LAMBDA * g_vx_max
        grad_cy = -LAMBDA * g_vy_min + LAMBDA * g_vy_max
        grad_s = LAMBDA * 0.5 * g_vx_min - LAMBDA * 0.5 * g_vx_max + \
                 LAMBDA * 0.5 * g_vy_min - LAMBDA * 0.5 * g_vy_max
        
        # 2. Overlap Penalty
        # For each pair (i, j), i < j:
        # dx = cx_i - cx_j
        # dy = cy_i - cy_j
        # Overlap in x: max(0, (s_i + s_j)/2 - |dx|)
        # Overlap in y: max(0, (s_i + s_j)/2 - |dy|)
        # Penalty = sum(overlap_x * overlap_y)
        
        # This is tricky to differentiate directly with smooth_max because of the product.
        # Alternative: Use a penalty based on the "minimum separation required" violation.
        # If squares overlap, the distance between centers is less than (s_i+s_j)/2.
        # Let's use a simpler penalty: Sum of squared overlap areas?
        # Or just sum of (overlap_x * overlap_y).
        
        # Let's implement the gradient for P_ij = smooth_max((s_i+s_j)/2 - |dx|) * smooth_max((s_i+s_j)/2 - |dy|)
        # This is computationally heavy for 45 pairs but n=10 is small.
        
        total_overlap_pen = 0.0
        grad_cx_overlap = np.zeros(n)
        grad_cy_overlap = np.zeros(n)
        grad_s_overlap = np.zeros(n)
        
        eps = 1e-6
        
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                s_sum_half = (s[i] + s[j]) / 2.0
                
                # Overlap width in x
                ox_arg = s_sum_half - np.abs(dx)
                oy_arg = s_sum_half - np.abs(dy)
                
                ox = smooth_max(ox_arg, eps)
                oy = smooth_max(oy_arg, eps)
                
                p = ox * oy
                total_overlap_pen += p
                
                # Gradients
                # d/dx smooth_max(s_sum - |x|)
                # Let u = s_sum - |x|
                # d/dx smooth_max(u) = sm_grad(u) * du/dx
                # du/dx = -sign(x) (if x != 0, else 0)
                # Actually, derivative of -|x| is -sign(x).
                
                g_ox = sm_grad(ox_arg, eps) * (-np.sign(dx))
                g_oy = sm_grad(oy_arg, eps) * (-np.sign(dy))
                
                # Product rule: d(ox*oy)/dx = (d ox/dx)*oy + ox*(d oy/dx)
                # d ox/dx involves cx[i] and cx[j]
                # d cx[i]/dx = 1, d cx[j]/dx = -1 (if we consider dx = cx[i]-cx[j])
                # Wait, we are differentiating w.r.t cx[i] and cx[j] separately.
                
                # d p / d cx[i]:
                # d ox / d cx[i] = g_ox * d(-|cx[i]-cx[j]|)/d cx[i] = g_ox * (-sign(cx[i]-cx[j]))
                # d oy / d cx[i] = 0 (since oy depends on cy)
                # So d p / d cx[i] = (g_ox * -sign(dx)) * oy + ox * 0
                
                # Let's be precise.
                # ox depends on dx = cx[i] - cx[j].
                # d ox / d cx[i] = sm_grad(ox_arg) * d(ox_arg)/d cx[i]
                # ox_arg = s_sum - |cx[i]-cx[j]|
                # d(ox_arg)/d cx[i] = -sign(cx[i]-cx[j])
                # So d ox / d cx[i] = -sign(dx) * sm_grad(ox_arg)
                
                # Similarly d ox / d cx[j] = -sign(cx[j]-cx[i]) * sm_grad(ox_arg) = sign(dx) * sm_grad(ox_arg)
                
                # oy depends on dy = cy[i] - cy[j].
                # d oy / d cy[i] = -sign(dy) * sm_grad(oy_arg)
                # d oy / d cy[j] = sign(dy) * sm_grad(oy_arg)
                
                # s_sum depends on s[i] and s[j].
                # d ox / d s[i] = sm_grad(ox_arg) * 0.5
                # d ox / d s[j] = sm_grad(ox_arg) * 0.5
                # d oy / d s[i] = sm_grad(oy_arg) * 0.5
                # d oy / d s[j] = sm_grad(oy_arg) * 0.5
                
                g_ox_val = sm_grad(ox_arg, eps)
                g_oy_val = sm_grad(oy_arg, eps)
                
                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                
                # Gradient w.r.t cx[i]
                d_p_dcxi = (-sign_dx * g_ox_val) * oy
                grad_cx_overlap[i] += LAMBDA * d_p_dcxi
                
                # Gradient w.r.t cx[j]
                d_p_dcj = (sign_dx * g_ox_val) * oy
                grad_cx_overlap[j] += LAMBDA * d_p_dcj
                
                # Gradient w.r.t cy[i]
                d_p_dcyi = (-sign_dy * g_oy_val) * ox
                grad_cy_overlap[i] += LAMBDA * d_p_dcyi
                
                # Gradient w.r.t cy[j]
                d_p_dcyj = (sign_dy * g_oy_val) * ox
                grad_cy_overlap[j] += LAMBDA * d_p_dcyj
                
                # Gradient w.r.t s[i]
                d_p_ds_i = (0.5 * g_ox_val) * oy + ox * (0.5 * g_oy_val)
                grad_s_overlap[i] += LAMBDA * d_p_ds_i
                
                # Gradient w.r.t s[j]
                d_p_ds_j = (0.5 * g_ox_val) * oy + ox * (0.5 * g_oy_val)
                grad_s_overlap[j] += LAMBDA * d_p_ds_j
        
        # Total Objective
        obj = -np.sum(s) + LAMBDA * (boundary_pen + total_overlap_pen)
        
        # Total Gradient
        grad_cx_total = grad_cx + grad_cx_overlap
        grad_cy_total = grad_cy + grad_cy_overlap
        grad_s_total = grad_s + grad_s_overlap - 1.0  # -1 from -sum(s)
        
        # Interleave gradients
        grad = np.empty(3*n)
        grad[0::3] = grad_cx_total
        grad[1::3] = grad_cy_total
        grad[2::3] = grad_s_total
        
        return obj, grad

    def is_feasible(centers, sides, tol=1e-7):
        """
        Checks if the configuration is strictly valid according to the task contract.
        """
        x_min = centers[:, 0] - sides / 2
        x_max = centers[:, 0] + sides / 2
        y_min = centers[:, 1] - sides / 2
        y_max = centers[:, 1] + sides / 2
        
        if np.any(x_min < -tol) or np.any(x_max > 1 + tol) or np.any(y_min < -tol) or np.any(y_max > 1 + tol):
            return False
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = np.abs(centers[i_idx, 0] - centers[j_idx, 0])
        dy = np.abs(centers[i_idx, 1] - centers[j_idx, 1])
        
        t = (sides[i_idx] + sides[j_idx]) / 2.0 - tol
        
        # Violation if dx < t AND dy < t
        violations = (dx < t) & (dy < t)
        
        return not np.any(violations)

    def run_optimization(initial_params, max_iter=500):
        """
        Runs L-BFGS-B from initial_params.
        Returns best feasible params and sum of sides found during the run.
        """
        best_params = initial_params.copy()
        best_sum = -np.inf
        
        # Callback to check feasibility and update best
        def callback(xk):
            nonlocal best_params, best_sum
            if time.time() > hard_deadline:
                return True # Signal to stop
            
            cx = xk[0::3]
            cy = xk[1::3]
            s = xk[2::3]
            
            centers = np.column_stack((cx, cy))
            
            # Quick feasibility check (might be slightly expensive but n=10 is small)
            if is_feasible(centers, s):
                current_sum = np.sum(s)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_params = xk.copy()
            return False
        
        # Bounds: 
        # cx, cy in [0, 1] (loose bounds, penalty handles strictness)
        # s in [0.01, 1.0]
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
            bounds.append((0.01, 1.0)) # s
        
        try:
            res = minimize(
                objective_and_grad,
                initial_params,
                method='L-BFGS-B',
                jac=True, # We return (f, g)
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            # Check final result
            cx = res.x[0::3]
            cy = res.x[1::3]
            s = res.x[2::3]
            centers = np.column_stack((cx, cy))
            if is_feasible(centers, s):
                current_sum = np.sum(s)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_params = res.x.copy()
        except Exception as e:
            pass
            
        return best_params, best_sum

    # Main Loop: Multiple Restarts
    best_global_params = None
    best_global_sum = -np.inf
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial configurations
    # 1. Grid-like
    # 2. Random clusters
    # 3. Random uniform
    
    restart_id = 0
    while time.time() < hard_deadline:
        # Create initial params
        # Strategy: Start with a feasible-ish configuration to help convergence
        # e.g., 4x4 grid with small squares, then let optimizer grow them
        
        init_params = np.zeros(3*n)
        
        # Randomize initial positions and sizes
        # Sizes: Small to start, e.g., 0.1
        # Positions: Random
        
        # Option 1: Random
        if restart_id % 3 == 0:
            s = rng.uniform(0.05, 0.3, n)
            cx = rng.uniform(s/2, 1 - s/2, n)
            cy = rng.uniform(s/2, 1 - s/2, n)
        # Option 2: Grid
        elif restart_id % 3 == 1:
            k = 4
            side = 0.2
            idx = np.arange(10)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * (1.0/k) + rng.uniform(-0.05, 0.05, n)
            cy = (rows + 0.5) * (1.0/k) + rng.uniform(-0.05, 0.05, n)
            s = np.full(n, side)
            # Clip to bounds
            cx = np.clip(cx, 0.01, 0.99)
            cy = np.clip(cy, 0.01, 0.99)
        # Option 3: Clusters
        else:
            # 2 clusters of 5
            s = rng.uniform(0.1, 0.4, n)
            cx = np.concatenate([
                rng.uniform(0.1, 0.4, 5),
                rng.uniform(0.6, 0.9, 5)
            ])
            cy = np.concatenate([
                rng.uniform(0.1, 0.4, 5),
                rng.uniform(0.6, 0.9, 5)
            ])
            cx = np.clip(cx, s/2, 1 - s/2)
            cy = np.clip(cy, s/2, 1 - s/2)
            
        init_params[0::3] = cx
        init_params[1::3] = cy
        init_params[2::3] = s
        
        # Run optimization
        p_opt, sum_opt = run_optimization(init_params, max_iter=200)
        
        if sum_opt > best_global_sum:
            best_global_sum = sum_opt
            best_global_params = p_opt
            
        restart_id += 1
        
        # If we've run for a while, break to return
        # L-BFGS-B is fast, so we can do many restarts
        if restart_id > 100:
            break

    # Fallback: If no feasible solution found (unlikely), return a known valid one
    if best_global_params is None:
        # Simple 4x4 grid
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.column_stack((cx, cy))
        sides = np.full(10, side)
        return centers, sides
    
    # Extract centers and sides from best_params
    centers = np.column_stack((best_global_params[0::3], best_global_params[1::3]))
    sides = best_global_params[2::3]
    
    return centers, sides

# EVOLVE-BLOCK-END
