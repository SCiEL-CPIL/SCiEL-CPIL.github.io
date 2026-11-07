"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s safety margin within 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_time():
            return time.time() < deadline

        def nm_simplex(x0, f0, budget_remaining, rng, func, deadline, calls_counter):
            """Run Nelder-Mead simplex starting from x0."""
            nonlocal best_x, best_f, calls_used
            
            n = dim
            # Initialize simplex: n+1 points
            simplex = np.zeros((n + 1, n))
            values = np.zeros(n + 1)
            
            simplex[0] = x0
            values[0] = f0
            calls_counter[0] += 1
            
            # Initial step size
            step = 0.5
            for i in range(1, n + 1):
                simplex[i] = simplex[0].copy()
                simplex[i][i - 1] += step
                simplex[i] = np.clip(simplex[i], -5, 5)
                values[i] = func(simplex[i])
                calls_counter[0] += 1
                if values[i] < best_f:
                    best_f = values[i]
                    best_x = simplex[i].copy()
            
            # Sort simplex
            order = np.argsort(values)
            simplex = simplex[order]
            values = values[order]
            
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            max_iter = budget_remaining // 6  # rough estimate of iterations
            for iteration in range(max_iter):
                if not check_time():
                    break
                if calls_counter[0] >= budget:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_counter[0] += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < values[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_counter[0] += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        values[-1] = f_ref
                elif f_ref < values[-2]:
                    simplex[-1] = x_ref
                    values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < values[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        calls_counter[0] += 1
                        if f_con < best_f:
                            best_f = f_con
                            best_x = x_con.copy()
                        if f_con <= f_ref:
                            simplex[-1] = x_con
                            values[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                values[i] = func(simplex[i])
                                calls_counter[0] += 1
                                if values[i] < best_f:
                                    best_f = values[i]
                                    best_x = simplex[i].copy()
                    else:
                        x_con = centroid - rho * (centroid - simplex[-1])
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        calls_counter[0] += 1
                        if f_con < best_f:
                            best_f = f_con
                            best_x = x_con.copy()
                        if f_con < values[-1]:
                            simplex[-1] = x_con
                            values[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                values[i] = func(simplex[i])
                                calls_counter[0] += 1
                                if values[i] < best_f:
                                    best_f = values[i]
                                    best_x = simplex[i].copy()
                
                # Re-sort
                order = np.argsort(values)
                simplex = simplex[order]
                values = values[order]
                
                # Stagnation check: if best hasn't improved in many iterations, break
                if iteration > 50 and values[0] >= best_f * 1.000001:
                    # Check if we've been stuck
                    pass
            
            return simplex[0], values[0]

        # Phase 1: Initial exploration with several random starts
        n_initial = 10
        initial_budget = min(60, budget // 8)
        calls_counter = [0]
        
        candidates = []
        for i in range(n_initial):
            if not check_time() or calls_counter[0] >= initial_budget:
                break
            x = rng.uniform(-4, 4, dim)
            f = func(x)
            calls_counter[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            candidates.append((f, x))
        
        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Run Nelder-Mead from top candidates
        remaining_budget = budget - calls_counter[0]
        n_restarts = min(5, len(candidates))
        
        for i in range(n_restarts):
            if not check_time() or calls_counter[0] >= budget:
                break
            
            # Allocate budget per restart
            restart_budget = remaining_budget // (n_restarts - i)
            if restart_budget < 10:
                restart_budget = 10
            
            x_start = candidates[i][1]
            f_start = candidates[i][0]
            
            # Run NM from this start
            x_best, f_best = nm_simplex(
                x_start, f_start, restart_budget, rng, func, deadline, calls_counter
            )
            
            remaining_budget = budget - calls_counter[0]
            
            # If we found something better, add to candidates for next restart
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
        
        # Phase 3: Final polish from the best point found
        if check_time() and calls_counter[0] < budget:
            final_budget = budget - calls_counter[0]
            if final_budget >= 10 and best_x is not None:
                x_final, f_final = nm_simplex(
                    best_x, best_f, final_budget, rng, func, deadline, calls_counter
                )
        
        return best_x, best_f
# EVOLVE-BLOCK-END
