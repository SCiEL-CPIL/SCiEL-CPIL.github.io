"""Rewritten optimizer: Multi-start Nelder-Mead with adaptive simplex for Sharp Ridge."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using Multi-start Nelder-Mead with adaptive simplex scaling.
    Nelder-Mead is effective on sharp ridge functions because it can follow
    narrow valleys by maintaining directional information in the simplex.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Multi-start Initialization ----------
        # Use random sampling to find the basin of the sharp ridge.
        # The minimizer is in [-4, 4]^5, so sample in that range.
        n_starts = 30
        for _ in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: Nelder-Mead Refinement ----------
        # Nelder-Mead is very effective on ridges in low dimensions.
        # Key: use a large initial simplex to capture the ridge direction.
        
        x0 = best_x.copy()
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink

        def run_nelder_mead(x0, remaining_budget):
            """Run Nelder-Mead from x0 with given remaining budget."""
            nonlocal evaluated, best_x, best_f
            
            if evaluated >= budget or not elapsed_ok():
                return
            
            # Initialize simplex with larger delta for faster initial convergence
            # delta=3 works well for AOCC scoring as noted in findings
            delta = 3.0
            
            simplex = [x0.copy()]
            for i in range(dim):
                x_pert = x0.copy()
                # Perturb in direction i, ensuring we stay in bounds
                step = delta if x0[i] < 5.0 - delta else -delta
                x_pert[i] += step
                simplex.append(clip(x_pert))
            
            # Evaluate initial simplex
            simplex_f = []
            for x in simplex:
                if evaluated >= budget or not elapsed_ok():
                    return
                simplex_f.append(eval_point(x))
            
            if len(simplex_f) < dim + 1:
                return
            
            # Nelder-Mead main loop
            max_iters = 300
            no_improve_count = 0
            prev_best = best_f
            
            for _ in range(max_iters):
                if evaluated >= budget or not elapsed_ok():
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check convergence
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-10:
                    break
                
                # Track improvement
                if best_f < prev_best - 1e-12:
                    no_improve_count = 0
                    prev_best = best_f
                else:
                    no_improve_count += 1
                
                # If no improvement for many iterations, shrink the simplex
                if no_improve_count > 10:
                    # Shrink all points toward the best
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i])
                        if evaluated >= budget or not elapsed_ok():
                            return
                        simplex_f[i] = eval_point(simplex[i])
                    no_improve_count = 0
                    prev_best = best_f
                    continue
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = clip(x_ref)
                f_ref = eval_point(x_ref)
                
                if evaluated >= budget or not elapsed_ok():
                    break
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = clip(x_exp)
                    f_exp = eval_point(x_exp)
                    
                    if evaluated >= budget or not elapsed_ok():
                        break
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    
                    x_con = clip(x_con)
                    f_con = eval_point(x_con)
                    
                    if evaluated >= budget or not elapsed_ok():
                        break
                    
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = clip(simplex[i])
                            if evaluated >= budget or not elapsed_ok():
                                return
                            simplex_f[i] = eval_point(simplex[i])
            
            # After Nelder-Mead, do a final local search with smaller steps
            if evaluated < budget and elapsed_ok():
                # Fine-tuning with very small perturbations
                for _ in range(20):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    x_try = best_x + rng.normal(0, 0.01, size=dim)
                    eval_point(x_try)
        
        # Run Nelder-Mead from the best point
        run_nelder_mead(x0, budget - evaluated)
        
        # If we still have budget, try another Nelder-Mead from a slightly different point
        if evaluated < budget - 10 and elapsed_ok():
            # Try from a point slightly offset from best
            x0b = best_x + rng.normal(0, 0.5, size=dim)
            x0b = clip(x0b)
            f0b = eval_point(x0b)
            if f0b <= best_f + 1.0:  # Only if not much worse
                run_nelder_mead(x0b, budget - evaluated)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
