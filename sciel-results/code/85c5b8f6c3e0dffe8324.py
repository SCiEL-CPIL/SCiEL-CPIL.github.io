"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (μ+λ)-ES with adaptive step, multi-restart,
    and Rastrigin-aware refinement for f15 (rotated Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Rastrigin-aware local refinement ---
        def rastrigin_local_refine(x, fx, max_rounds=10):
            """Rastrigin-aware local refinement using coordinate descent
            with periodic structure exploitation."""
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return x, fx
            
            for _ in range(max_rounds):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for axis in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x0 = x[axis]
                    
                    # Try snapping to nearest 2*pi multiple (Rastrigin minima)
                    nearest = 2.0 * np.pi * np.round(x0 / (2.0 * np.pi))
                    candidates = [x0]
                    
                    # Add periodic candidates
                    for k in [-1, 0, 1]:
                        cand = nearest + k * 2 * np.pi
                        if -5 <= cand <= 5:
                            candidates.append(cand)
                    
                    # Add fine-grained perturbations around current and nearest
                    for base in [x0, nearest]:
                        for d in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                            cand = base + d
                            if -5 <= cand <= 5:
                                candidates.append(cand)
                    
                    # Remove duplicates
                    candidates = list(set([float(c) for c in candidates]))
                    
                    best_cand = x0
                    best_cand_f = fx
                    for c in candidates:
                        if call_count >= budget or time.time() > deadline:
                            break
                        if abs(c - x0) < 1e-10:
                            continue
                        y = x.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    
                    if best_cand != x0:
                        x[axis] = best_cand
                        fx = best_cand_f
                        improved = True
                
                if not improved:
                    break
            
            return x, fx

        # --- Nelder-Mead style local refinement for near-optimum ---
        def nelder_mead_refine(x, fx, max_iter=50):
            """Simple Nelder-Mead for fine-tuning near the optimum."""
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return x, fx
            
            # Initialize simplex
            simplex = [x.copy()]
            simplex_f = [fx]
            step = 0.1
            for i in range(dim):
                new_x = x.copy()
                new_x[i] += step
                new_x = np.clip(new_x, -5, 5)
                new_f = eval_func(new_x)
                simplex.append(new_x)
                simplex_f.append(new_f)
            
            for _ in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_func(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_func(x_exp)
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
                    f_con = eval_func(x_con)
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = eval_func(simplex[i])
            
            # Return best point from simplex
            best_idx = np.argmin(simplex_f)
            return simplex[best_idx], simplex_f[best_idx]

        # --- Main search: (mu+lambda)-ES with periodic restarts ---
        mu = 4
        lam = 12
        sigma = 2.5
        # Initial population
        pop = np.array([np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(mu)])
        pop_f = np.array([eval_func(p) for p in pop])
        # Sort
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]

        generation = 0
        no_improve_count = 0
        best_f_prev = best_f

        while call_count < budget and time.time() < deadline:
            generation += 1

            # Adaptive sigma based on progress
            if generation % 10 == 0:
                if best_f < best_f_prev - 1e-6:
                    sigma = min(sigma * 1.2, 6.0)
                    best_f_prev = best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        sigma = max(sigma * 0.75, 0.05)
                        no_improve_count = 0

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                if call_count >= budget or time.time() > deadline:
                    break
                # Select parent (tournament from best mu)
                parent_idx = rng.integers(0, mu)
                parent = pop[parent_idx]
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                child_f = eval_func(child)
                offspring.append(child)
                offspring_f.append(child_f)

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Combine parents and offspring, select best mu
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([pop_f, offspring_f])
            idx = np.argsort(all_f)[:mu]
            pop = all_x[idx]
            pop_f = all_f[idx]

            # Periodic local refinement on the best point
            if generation % 10 == 0:
                # First try Rastrigin-aware refinement
                x_ref, f_ref = rastrigin_local_refine(pop[0].copy(), pop_f[0], max_rounds=5)
                if f_ref < pop_f[0]:
                    pop[0] = x_ref
                    pop_f[0] = f_ref
                
                # If very close to a good value, try Nelder-Mead
                if f_ref < 50:  # Heuristic: if value is reasonably good
                    x_nm, f_nm = nelder_mead_refine(pop[0].copy(), pop_f[0], max_iter=20)
                    if f_nm < pop_f[0]:
                        pop[0] = x_nm
                        pop_f[0] = f_nm

            # Restart if stuck
            if sigma < 0.1 and generation % 20 == 0:
                for _ in range(3):
                    if call_count >= budget or time.time() > deadline:
                        break
                    new_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    new_f = eval_func(new_x)
                    worst_idx = np.argmax(pop_f)
                    if new_f < pop_f[worst_idx]:
                        pop[worst_idx] = new_x
                        pop_f[worst_idx] = new_f
                sigma = 2.0
                no_improve_count = 0

        # Final phase: intensive local refinement on best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            x_final = best_x.copy()
            f_final = best_f
            
            # Multiple rounds of Rastrigin-aware refinement
            for _ in range(5):
                if call_count >= budget or time.time() > deadline:
                    break
                x_final, f_final = rastrigin_local_refine(x_final, f_final, max_rounds=8)
                if f_final >= best_f:
                    break
            
            # If we're in a good region, try Nelder-Mead for final polish
            if f_final < 100 and call_count < budget and time.time() < deadline:
                x_nm, f_nm = nelder_mead_refine(x_final.copy(), f_final, max_iter=30)
                if f_nm < f_final:
                    x_final = x_nm
                    f_final = f_nm

        return best_x, best_f
# EVOLVE-BLOCK-END
