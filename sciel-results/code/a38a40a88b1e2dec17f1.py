"""Hybrid Differential Evolution and Nelder-Mead optimizer."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid DE and Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution
        # Reserve some budget for Nelder-Mead
        nm_budget = 50
        de_budget = budget - nm_budget
        
        if de_budget < 10:
            # If budget is very small, just do Nelder-Mead
            de_budget = 0
            nm_budget = budget

        best_x = np.zeros(dim)
        best_f = np.inf

        if de_budget > 0:
            N_p = 12
            F_min = 0.1
            F_max = 0.9
            CR = 0.5

            pop = rng.uniform(-5, 5, size=(N_p, dim))
            fpop = np.array([func(x) for x in pop])
            calls = N_p

            best_idx = np.argmin(fpop)
            best_x = pop[best_idx].copy()
            best_f = fpop[best_idx]

            max_gen = de_budget // N_p
            
            for gen in range(max_gen):
                if calls >= de_budget:
                    break
                progress = gen / max_gen if max_gen > 0 else 1.0
                F = F_max - (F_max - F_min) * progress
                
                for i in range(N_p):
                    if calls >= de_budget:
                        break
                    
                    # Select distinct r1, r2, r3
                    while True:
                        r1, r2, r3 = rng.choice(N_p, size=3, replace=False)
                        if r1 != r2 and r2 != r3 and r1 != r3:
                            break
                    
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)
                    
                    j0 = rng.integers(0, dim)
                    trial = pop[i].copy()
                    mask = rng.random(dim) < CR
                    mask[j0] = True
                    trial[mask] = v[mask]
                    
                    f_trial = func(trial)
                    calls += 1
                    
                    if f_trial <= fpop[i]:
                        pop[i] = trial
                        fpop[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()

        # Phase 2: Nelder-Mead
        if nm_budget > 0:
            # Initialize simplex
            # Start from best_x
            # Create simplex: best_x + delta * e_i
            delta = 0.5
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(dim):
                simplex[i + 1] = best_x.copy()
                simplex[i + 1][i] += delta
            
            # Evaluate simplex
            f_simplex = np.array([func(x) for x in simplex])
            calls += dim + 1
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            for _ in range(nm_budget):
                if calls >= budget:
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Check for convergence (optional, but we have a fixed budget)
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                
                if f_simplex[0] <= f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                elif f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                else:
                    # Contraction
                    # Check if worst is better than reflection
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (centroid - worst)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                        if calls >= budget:
                            break

            # Update best from simplex
            min_idx = np.argmin(f_simplex)
            if f_simplex[min_idx] < best_f:
                best_f = f_simplex[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
