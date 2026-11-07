"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 330s wall clock limit. Keep a safety margin of 5s.
        deadline = start_time + 325.0
        
        best_x = None
        best_f = np.inf
        call_count = 0
        n = self.dim
        
        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            call_count += 1
            return fx

        # 1. Initial Probe
        # Start at origin. It's a safe, central point.
        x0 = np.zeros(n)
        f0 = make_call(x0)
        if f0 is not None and np.isfinite(f0):
            best_f = f0
            best_x = x0.copy()
        else:
            # Fallback if origin fails (unlikely but safe)
            return None, np.inf

        # 2. Strategy: Nelder-Mead Simplex with Restart
        # Nelder-Mead is a derivative-free method that works well on ill-conditioned
        # problems by adapting the simplex shape. It is structurally different from
        # stochastic search (ES/SA) as it uses geometric operations (reflection,
        # expansion, contraction) based on the ordering of function values.
        
        # Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrink
        
        # We will run multiple Nelder-Mead instances.
        # Each instance starts from a random point (or the current best) and runs
        # until it converges or uses a portion of the budget.
        
        # Budget allocation:
        # We have 500 calls.
        # Let's allow up to 5 restarts.
        # Each restart gets roughly budget/5 = 100 calls.
        # However, we should stop a run early if it converges (simplex shrinks).
        
        max_restarts = 5
        calls_per_restart = self.budget // max_restarts
        
        # First run starts from best_x (origin)
        current_start = best_x.copy()
        
        for restart_idx in range(max_restarts):
            # If we have enough budget left, start a new NM run
            if call_count >= self.budget or time.time() > deadline:
                break
            
            # For the first restart, use best_x. For subsequent, use random points
            # to escape local minima, unless we are already very close to the bound.
            if restart_idx == 0:
                x_start = current_start
            else:
                # Random start in the box, biased towards the center slightly?
                # Or just uniform.
                x_start = np.random.uniform(-5, 5, n)
            
            # Initialize Simplex
            # The simplex consists of n+1 points.
            # Point 0 is the start point.
            # Points 1..n are start point + small perturbation in each dimension.
            
            # Choose an initial step size for the simplex.
            # If we are at the origin, a step of 0.5 is reasonable.
            # If we are restarting from a random point, maybe 0.5 is also fine.
            # For Schaffer F7, the features can be small, so we might want to start
            # with a slightly larger simplex to capture the basin, then shrink.
            simplex_step = 0.5
            
            simplex = [x_start.copy()]
            for i in range(n):
                point = x_start.copy()
                # If x_start[i] is close to boundary, adjust step direction
                # to stay within bounds if possible, but clipping handles it.
                point[i] += simplex_step
                simplex.append(point)
            
            # Evaluate initial simplex
            simplex_f = []
            for i in range(n + 1):
                fx = make_call(simplex[i])
                if fx is None:
                    # Out of budget or time
                    return best_x, best_f
                if not np.isfinite(fx):
                    fx = np.inf
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()
            
            # Main NM loop
            # We limit the number of iterations per restart to avoid getting stuck
            # in a long, non-productive tail.
            max_nm_iter = calls_per_restart
            
            for _ in range(max_nm_iter):
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Check convergence: if the range of function values is very small
                # or the simplex is very small, stop this restart.
                if abs(simplex_f[-1] - simplex_f[0]) < 1e-8:
                    break
                # Also check simplex size
                simplex_size = np.max(np.linalg.norm(simplex[i] - simplex[0] for i in range(n+1)))
                if simplex_size < 1e-6:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                
                # Reflection
                x_ref = centroid + alpha * (centroid - worst_point)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = make_call(x_ref)
                if f_ref is None:
                    break
                if not np.isfinite(f_ref):
                    f_ref = np.inf
                
                if f_ref < simplex_f[0]:
                    # Reflection is better than best. Try expansion.
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = make_call(x_exp)
                    if f_exp is None:
                        break
                    if not np.isfinite(f_exp):
                        f_exp = np.inf
                    
                    if f_exp < f_ref:
                        # Expansion is better
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        # Reflection is better
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Reflection is better than second worst, but not best. Accept.
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Reflection is not good. Try contraction.
                    # Outside contraction (if f_ref < worst) or Inside contraction
                    # Standard NM usually does inside contraction if reflection fails.
                    x_con = centroid + rho * (worst_point - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = make_call(x_con)
                    if f_con is None:
                        break
                    if not np.isfinite(f_con):
                        f_con = np.inf
                    
                    if f_con < simplex_f[-1]:
                        # Contraction is better than worst
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        # Shrink all points except the best towards the best
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fx = make_call(simplex[i])
                            if fx is None:
                                return best_x, best_f
                            if not np.isfinite(fx):
                                fx = np.inf
                            simplex_f[i] = fx
                            if fx < best_f:
                                best_f = fx
                                best_x = simplex[i].copy()
                        
                        # After shrink, we might need to re-evaluate or just continue
                        # The loop will re-sort at the top.
                
                # Update global best
                min_f_in_simplex = min(simplex_f)
                if min_f_in_simplex < best_f:
                    best_f = min_f_in_simplex
                    # Find the index of the min
                    idx_min = np.argmin(simplex_f)
                    best_x = simplex[idx_min].copy()

            # If we finished the NM loop without breaking due to budget/time,
            # we are ready for the next restart.
            # The next restart will start from a random point to explore elsewhere.
            # We keep the global best_x for the final return.

        return best_x, best_f
# EVOLVE-BLOCK-END
