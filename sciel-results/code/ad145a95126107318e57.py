import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit with 20s safety margin
        
        call_count = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= self.budget:
                return float('inf')
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            fx = float(func(x))
            call_count[0] += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution
        N = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N, self.dim))
        fit = np.empty(N)
        for i in range(N):
            if time_left() < 5:
                break
            fit[i] = wrapped_func(pop[i])
        
        # Handle case where not all initialized due to time
        if len(fit) < N or any(~np.isfinite(fit)):
             # This shouldn't happen normally, but just in case
             fit = fit[:len(fit)]
             pop = pop[:len(fit)]
             N = len(fit)
             if N == 0:
                 return best_x if best_x is not None else np.zeros(self.dim), best_f if np.isfinite(best_f) else 0.0

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]

        # DE Loop
        # Each iteration evaluates N new points.
        max_iter_de = max(1, (self.budget - call_count[0]) // N)
        
        for gen in range(max_iter_de):
            if time_left() < 10:
                break
            if call_count[0] >= self.budget - 10:
                break
            
            trial_pop = np.empty_like(pop)
            trial_fit = np.empty(N)
            
            for i in range(N):
                if time_left() < 5:
                    break
                if call_count[0] >= self.budget - 10:
                    break
                
                # Select random distinct individuals
                candidates = list(range(N))
                candidates.remove(i)
                if len(candidates) < 3:
                    r1, r2, r3 = i, (i+1)%N, (i+2)%N
                else:
                    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(0, self.dim)
                cross_mask = rng.random(self.dim) < CR
                cross_mask[j_rand] = True
                
                u = np.where(cross_mask, v, pop[i])
                u = np.clip(u, -5, 5)
                
                trial_pop[i] = u
                trial_fit[i] = wrapped_func(u)
                
                # Selection
                if trial_fit[i] < fit[i]:
                    pop[i] = u
                    fit[i] = trial_fit[i]
                    if trial_fit[i] < best_f:
                        best_f = trial_fit[i]
                        best_x = u.copy()
                        
        # Phase 2: Local Refinement using Nelder-Mead (manual implementation for control)
        remaining_budget = self.budget - call_count[0]
        if remaining_budget > 20 and time_left() > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f

            # Initialize simplex
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x
            for i in range(self.dim):
                simplex[i + 1] = x
                simplex[i + 1][i] += 0.5 if x[i] < 4.5 else -0.5
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

            simplex_f = np.empty(self.dim + 1)
            for i in range(self.dim + 1):
                if time_left() < 5 or call_count[0] >= self.budget:
                    break
                simplex_f[i] = wrapped_func(simplex[i])
            
            # Check if we managed to evaluate the whole simplex
            if np.any(~np.isfinite(simplex_f)):
                pass # Skip NM if incomplete

            # Nelder-Mead iterations
            max_iter_nm = remaining_budget // (self.dim + 2)
            for _ in range(max_iter_nm):
                if time_left() < 5:
                    break
                if call_count[0] >= self.budget:
                    break

                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)

                if simplex_f[0] < f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                elif f_r < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - beta * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)

                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = wrapped_func(simplex[i])
                            if time_left() < 5 or call_count[0] >= self.budget:
                                break

        return best_x, best_f
