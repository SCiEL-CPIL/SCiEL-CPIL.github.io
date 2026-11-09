"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-aware optimizer for Rosenbrock-like functions.
    Strategy:
    1. Estimate gradient using finite differences.
    2. Identify "valley direction" (perpendicular to gradient in the subspace of motion).
    3. Perform line search along the valley direction to move fast along the valley.
    4. Use a small perpendicular step to track the valley center.
    5. Adapt step sizes based on success rate.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # State
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1
        x = best_x.copy()
        fx = best_f

        # Adaptive parameters
        step_size = 1.0
        min_step = 1e-6
        max_step = 5.0
        success_count = 0
        total_trials = 0
        stagnation = 0

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            fx_val = func(np.clip(x, -5, 5))
            calls_used += 1
            return fx_val

        # Initial exploration: Random restarts to find the basin
        # Rosenbrock has a global min at (1,...,1) but rotated/shifted here.
        # We start at 0 and do a few random jumps to escape bad local minima if any.
        
        # Phase 1: Coarse Search / Basin Finding
        # Since the function is unimodal in the valley but has a narrow valley,
        # we first try to get close to the valley.
        
        # Let's use a simple random walk for the first 10% of budget to find a good region
        coarse_budget = int(budget * 0.1)
        for _ in range(coarse_budget):
            if check_deadline() or calls_used >= budget:
                break
            
            # Random perturbation
            dx = rng.standard_normal(n) * 2.0
            y = x + dx
            fy = do_call(y)
            
            if fy is None:
                break
                
            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Keep current x, maybe shrink perturbation if stuck? 
                # For now, just move on. The valley walker will handle refinement.
                pass

        # Phase 2: Valley Walking (Structure-Aware)
        # We estimate the gradient to find the direction of the valley.
        # For Rosenbrock, the valley direction is roughly orthogonal to the gradient 
        # in the local frame, but more precisely, we want to move along the 
        # direction of least curvature.
        
        # A simpler, robust approach for Rosenbrock:
        # The gradient points "up" the valley walls. 
        # The valley direction is perpendicular to the gradient.
        # However, in high dimensions, the valley is a 1D manifold (roughly).
        # We can use the gradient to project out the "wall" component.
        
        # Let's implement a directional derivative based step.
        # 1. Estimate gradient g at x.
        # 2. Choose a direction d. If we move along d, we want d to be roughly 
        #    perpendicular to g (to stay in valley) but also have a component 
        #    towards the minimum.
        
        # Actually, a standard trick for Rosenbrock is to use the gradient 
        # to identify the "bad" direction and step along the "good" one.
        
        # Let's try a different structural approach: 
        # **Nelder-Mead is often good for Rosenbrock if initialized well.**
        # But the prompt asks for a DIFFERENT strategy.
        
        # Let's try **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) 
        # implemented manually? No, that's complex.
        
        # Let's try **Line Search along the Gradient** (Steepest Descent) with 
        # **Backtracking**. This is structurally different from ES.
        # Steepest Descent is slow on Rosenbrock (zig-zagging).
        
        # How about **Conjugate Gradient** (non-linear)? 
        # It uses the gradient to build a direction that is conjugate to previous ones, 
        # avoiding zig-zagging. This is a very different family from ES/NM.
        
        # Implementation of Non-linear Conjugate Gradient (Polak-Ribiere)
        
        # Reset for CG
        x = best_x.copy()
        fx = best_f
        calls_used = max(calls_used, 1) # Ensure at least 1 call
        
        if calls_used >= budget or check_deadline():
            return best_x, best_f

        # Compute initial gradient
        def compute_gradient(x, f):
            grad = np.zeros(n)
            eps = 1e-5
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = do_call(x_plus)
                if f_plus is None:
                    return None
                grad[i] = (f_plus - f) / eps
            return grad

        # We need to be careful with budget. CG requires gradient calls (n calls) 
        # plus line search calls. 
        # n=5, so gradient costs 5 calls. 
        # Budget is 500. 
        # 500 / 5 = 100 gradient evaluations. 
        # This is feasible.
        
        g = compute_gradient(x, fx)
        if g is None:
            return best_x, best_f
            
        d = -g  # Initial direction
        calls_used_check = calls_used
        
        for k in range(200): # Max iterations
            if check_deadline() or calls_used >= budget - 10:
                break
                
            # Line search along d
            # Use Backtracking Line Search (Armijo)
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            
            # Ensure we have budget for line search
            while calls_used < budget - 5 and not check_deadline():
                x_new = x + alpha * d
                x_new = np.clip(x_new, -5, 5)
                f_new = do_call(x_new)
                
                if f_new is None:
                    break
                    
                # Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * g.T * d
                # Note: g.T * d is negative since d = -g initially.
                slope = np.dot(g, d)
                if f_new <= fx + c1 * alpha * slope:
                    # Accept step
                    x = x_new
                    fx = f_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    break
                else:
                    alpha *= rho
                    if alpha < 1e-8:
                        # Line search failed, try to reset direction
                        break
            
            if alpha < 1e-8:
                # Reset direction to steepest descent
                g = compute_gradient(x, fx)
                if g is None: break
                d = -g
                continue

            # Compute new gradient
            g_new = compute_gradient(x, fx)
            if g_new is None:
                break
                
            # Check convergence
            if np.linalg.norm(g_new) < 1e-6:
                break
                
            # Polak-Ribiere update
            # beta = max(0, (g_new.T * (g_new - g)) / (g.T * g))
            num = np.dot(g_new, g_new - g)
            den = np.dot(g, g)
            if den > 1e-12:
                beta = max(0.0, num / den)
            else:
                beta = 0.0
                
            d = -g_new + beta * d
            g = g_new

        return best_x, best_f
# EVOLVE-BLOCK-END
