"""Rewritten optimizer using Differential Evolution and Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid DE + Nelder-Mead optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, rng):
        """Run Nelder-Mead with a strict budget."""
        n = self.dim
        # Initialize simplex
        simplex = []
        fvals = []
        
        # First point
        x = np.array(x0, dtype=float)
        x = np.clip(x, -5, 5)
        f = func(x)
        simplex.append(x.copy())
        fvals.append(f)
        calls = 1
        
        # Other points: offset by 0.1 or 0.05
        step = 0.1
        for i in range(n):
            xi = x.copy()
            if xi[i] + step <= 5.0:
                xi[i] += step
            else:
                xi[i] -= step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi.copy())
            fvals.append(fi)
            calls += 1
            
            if calls >= budget:
                break

        if calls >= budget:
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        while calls < budget:
            # Sort simplex by fvals
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[n])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if calls >= budget:
                # Check if x_r is better than current best
                if f_r < min(fvals):
                    return x_r, f_r
                else:
                    return simplex[0], fvals[0]

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if calls >= budget:
                    if f_e < f_r:
                        return x_e, f_e
                    else:
                        return x_r, f_r

                if f_e < f_r:
                    simplex[n] = x_e
                    fvals[n] = f_e
                else:
                    simplex[n] = x_r
                    fvals[n] = f_r
            elif f_r < fvals[n-1]:
                # Accept reflection
                simplex[n] = x_r
                fvals[n] = f_r
            else:
                # Contraction
                if f_r < fvals[n]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if calls >= budget:
                    if f_c < f_r:
                        return x_c, f_c
                    else:
                        return simplex[0], fvals[0]

                if f_c < min(f_r, fvals[n]):
                    simplex[n] = x_c
                    fvals[n] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        idx = np.argmin(fvals)
                        return simplex[idx], fvals[idx]

        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]

    def _differential_evolution(self, func, budget, rng):
        """Run Differential Evolution with a strict budget."""
        n = self.dim
        NP = 20  # Population size
        F = 0.8   # Mutation scale
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (NP, n))
        pop = np.clip(pop, -5, 5)
        fvals = np.zeros(NP)
        
        calls = 0
        for i in range(NP):
            fvals[i] = func(pop[i])
            calls += 1
            if calls >= budget:
                break
        
        if calls >= budget:
            idx = np.argmin(fvals)
            return pop[idx], fvals[idx]

        best_idx = np.argmin(fvals)
        best_f = fvals[best_idx]
        best_x = pop[best_idx].copy()

        # DE Loop
        while calls < budget:
            # Select base and two distinct random indices
            base_idx = rng.integers(NP)
            r1 = rng.integers(NP)
            r2 = rng.integers(NP)
            
            # Ensure distinct
            while r1 == r2 or r1 == base_idx or r2 == base_idx:
                r1 = rng.integers(NP)
                r2 = rng.integers(NP)
                
            # Mutation
            v = pop[base_idx] + F * (pop[r1] - pop[r2])
            
            # Crossover
            trial = pop[base_idx].copy()
            j_rand = rng.integers(n)
            for j in range(n):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            trial = np.clip(trial, -5, 5)
            
            # Evaluate
            f_trial = func(trial)
            calls += 1
            
            # Selection
            if f_trial <= fvals[base_idx]:
                pop[base_idx] = trial
                fvals[base_idx] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    best_idx = base_idx

        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Allocate budget: 60% to DE, 40% to NM
        budget_de = int(budget * 0.6)
        budget_nm = budget - budget_de

        # Run DE
        x_de, f_de = self._differential_evolution(func, budget_de, rng)
        
        # Run NM from best DE point
        x_nm, f_nm = self._nelder_mead(func, x_de, budget_nm, rng)
        
        # Return the overall best
        if f_nm < f_de:
            return x_nm, f_nm
        else:
            return x_de, f_de
# EVOLVE-BLOCK-END
