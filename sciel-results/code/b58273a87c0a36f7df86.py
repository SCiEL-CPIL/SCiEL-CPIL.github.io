"""Memetic Optimizer for BBOB f23 (Katsuura): Differential Evolution + Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE-NM optimizer for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        rng = np.random.default_rng(42)
        
        # --- Phase 1: Differential Evolution for Global Search ---
        # DE is effective at finding the basin of the global minimum in multimodal landscapes.
        pop_size = 12
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-4.5, 4.5] to avoid boundaries initially
        pop = rng.uniform(-4.5, 4.5, (pop_size, d))
        pop_f = np.array([safe_call(x)[1] for x in pop])
        
        # Find the best index
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        max_gen = (B * 35) // 100  # Use 35% of budget for DE
        for g in range(max_gen):
            if calls >= B * 0.35 or time.time() > deadline - 60:
                break
            
            for i in range(pop_size):
                if calls >= B * 0.35 or time.time() > deadline - 60:
                    break
                
                # Choose 3 distinct random indices different from i
                r1, r2, r3 = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                trial = pop[i].copy()
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Bound handling
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                trial_x, trial_f = safe_call(trial)
                
                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial_x
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial_x.copy()

        # --- Phase 2: Nelder-Mead Local Refinement ---
        # Polish the best point found by DE.
        if best_x is not None and calls < B - 10 and time.time() < deadline - 10:
            x0, f0 = best_x.copy(), best_f
            
            # Initialize Simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Create initial simplex with small steps to respect local structure
            step_size = 0.1
            for i in range(d):
                xi = x0.copy()
                xi[i] += step_size
                xi = np.clip(xi, -5.0, 5.0)
                xi, fi = safe_call(xi)
                simplex.append(xi)
                simplex_f.append(fi)
            
            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Run NM until budget or time runs out
            while calls < B - 5 and time.time() < deadline - 5:
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                xr, fr = safe_call(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    xe, fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept Reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    xc, fc = safe_call(xc)
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            xi, fi = safe_call(simplex[i])
                            simplex[i] = xi
                            simplex_f[i] = fi
                            
                # Update global best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
