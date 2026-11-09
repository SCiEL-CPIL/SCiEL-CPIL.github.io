"""Rewritten optimizer for BBOB f20 (Schwefel) with robust (1+1)-ES and deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with step-size adaptation, multiple restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        total_calls = 0

        # Use a fixed seed for reproducibility across instances
        rng = np.random.default_rng(42)

        # Strategy: Multi-start Nelder-Mead with dynamic budget allocation
        # Phase 1: Initial sampling to find diverse starting points
        n_init = 20
        init_points = rng.uniform(-5, 5, (n_init, self.dim))
        init_vals = []
        for i in range(n_init):
            x = init_points[i]
            fx = func(x)
            total_calls += 1
            init_vals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead from top diverse points
        # Sort by function value and select top diverse points
        sorted_indices = np.argsort(init_vals)
        
        # Select up to 8 diverse starting points (spread across the landscape)
        selected_indices = []
        for idx in sorted_indices:
            if len(selected_indices) >= 8:
                break
            # Check diversity: ensure not too close to already selected
            is_diverse = True
            for sel_idx in selected_indices:
                if np.linalg.norm(init_points[idx] - init_points[sel_idx]) < 1.5:
                    is_diverse = False
                    break
            if is_diverse:
                selected_indices.append(idx)
        
        # If not enough diverse points, fill with best remaining
        for idx in sorted_indices:
            if len(selected_indices) >= 8:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)

        # Allocate budget dynamically among Nelder-Mead runs
        # Each run gets a portion of remaining budget
        remaining_budget = self.budget - total_calls
        n_runs = len(selected_indices)
        
        # Track convergence for each run
        run_states = []
        for idx in selected_indices:
            x0 = init_points[idx].copy()
            f0 = init_vals[idx]
            run_states.append({
                'x': x0,
                'f': f0,
                'budget_used': 0,
                'converged': False,
                'last_improvement': 0
            })

        # Dynamic budget allocation: runs that improve faster get more budget
        max_iter_per_run = 150  # Cap iterations per run to ensure time safety
        
        for run_idx, state in enumerate(run_states):
            if time.time() > deadline:
                break
            
            if state['converged']:
                continue
            
            # Allocate budget based on current performance
            # Better runs get more budget
            run_budget = min(max_iter_per_run, remaining_budget // (n_runs - run_idx))
            run_budget = max(run_budget, 10)  # Minimum 10 iterations
            
            # Run Nelder-Mead from this starting point
            try:
                # Custom Nelder-Mead with budget tracking
                x = state['x'].copy()
                f = state['f']
                
                # Initialize simplex
                simplex = [x.copy()]
                simplex_f = [f]
                
                # Create initial simplex vertices
                for d in range(self.dim):
                    v = x.copy()
                    v[d] += 0.5 if x[d] < 4.5 else -0.5
                    v = np.clip(v, -5, 5)
                    fv = func(v)
                    total_calls += 1
                    run_budget -= 1
                    simplex.append(v)
                    simplex_f.append(fv)
                
                # Nelder-Mead iterations
                for iter_i in range(run_budget):
                    if time.time() > deadline:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Check convergence
                    if simplex_f[0] - simplex_f[-1] < 1e-8:
                        state['converged'] = True
                        break
                    
                    # Compute centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    alpha = 1.0
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = func(x_ref)
                    total_calls += 1
                    run_budget -= 1
                    
                    if f_ref < simplex_f[0]:
                        # Expansion
                        gamma = 2.0
                        x_exp = centroid + gamma * (centroid - simplex[-1])
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = func(x_exp)
                        total_calls += 1
                        run_budget -= 1
                        
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        rho = 0.5
                        x_con = centroid + rho * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        total_calls += 1
                        run_budget -= 1
                        
                        if f_con < simplex_f[-1]:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, self.dim + 1):
                                simplex[i] = (simplex[0] + simplex[i]) / 2
                                f_shrink = func(simplex[i])
                                total_calls += 1
                                run_budget -= 1
                                simplex_f[i] = f_shrink
                    
                    # Update best
                    if simplex_f[0] < best_f:
                        best_f = simplex_f[0]
                        best_x = simplex[0].copy()
                        state['f'] = best_f
                        state['x'] = best_x.copy()
                        state['last_improvement'] = iter_i
                
                state['converged'] = True
            except Exception:
                state['converged'] = True
        
        # Phase 3: Final polish with (1+1)-ES from best point
        if time.time() < deadline and total_calls < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            success_window = []
            remaining = self.budget - total_calls
            
            for i in range(remaining):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    success_window.append(1)
                    sigma *= 1.05
                else:
                    success_window.append(0)
                    sigma *= 0.95
                
                if len(success_window) > 10:
                    success_window.pop(0)
                
                if len(success_window) == 10:
                    success_rate = sum(success_window) / 10.0
                    if success_rate > 0.3:
                        sigma *= 1.02
                    elif success_rate < 0.2:
                        sigma *= 0.98
                
                sigma = max(1e-10, min(sigma, 5.0))

        return best_x, best_f
