"""Multi-objective optimizer on RE37 using Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Differential Evolution (DE) on RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _is_non_dominated(self, f_new, F_list):
        """Check if f_new is non-dominated by any point in F_list."""
        for f in F_list:
            if self._dominates(f, f_new):
                return False
        return True

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Latin Hypercube Sampling for initialization
        n_init = 60
        X_lhs = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            X_lhs[:, d] = lo[d] + (perm + 0.5) * (hi[d] - lo[d]) / n_init

        F_list = []
        X_list = []
        
        for x in X_lhs:
            f = problem(x)
            # Only keep non-dominated points to maintain a high-quality archive
            if self._is_non_dominated(f, F_list):
                # Remove points dominated by the new point
                F_list = [f_old for f_old in F_list if not self._dominates(f, f_old)]
                F_list.append(f)
                X_list.append(x)
            
            used = len(F_list)
            # We don't track 'used' strictly here for budget, just for archive size
            # But we need to ensure we don't exceed total budget later.
            # Actually, every call to problem counts.
            # Let's track total evaluations properly.
        
        # Re-evaluating total evaluations used so far
        # The loop above called problem n_init times.
        evals_used = n_init
        
        if time.time() > deadline:
            return np.array(F_list)

        # Phase 2: Differential Evolution
        # Population size for DE
        pop_size = 40
        F_scale = 0.8
        CR = 0.5
        
        # We will maintain a population of size pop_size.
        # Initialize population with random points or from archive?
        # Let's initialize with random points, but keep the archive separate for final result.
        # Actually, standard DE maintains a population. 
        # Let's use a mixed approach: 
        # 1. Keep a global archive of all non-dominated points found.
        # 2. Use a population of size pop_size for DE.
        
        # Initialize DE population
        # Use points from the initial LHS if possible, else random
        if len(X_list) >= pop_size:
            # Randomly select from LHS results
            indices = np.random.choice(len(X_list), pop_size, replace=False)
            X_pop = np.array([X_list[i] for i in indices])
            F_pop = np.array([F_list[i] for i in indices])
        else:
            # Pad with random points
            X_pop = np.zeros((pop_size, self.dim))
            F_pop = np.zeros((pop_size, self.n_obj))
            for i in range(pop_size):
                if i < len(X_list):
                    X_pop[i] = X_list[i]
                    F_pop[i] = F_list[i]
                else:
                    X_pop[i] = np.random.uniform(lo, hi)
                    F_pop[i] = problem(X_pop[i])
                    evals_used += 1
                    # Update global archive
                    if self._is_non_dominated(F_pop[i], F_list):
                        F_list = [f_old for f_old in F_list if not self._dominates(F_pop[i], f_old)]
                        F_list.append(F_pop[i])
                        X_list.append(X_pop[i])

        if time.time() > deadline or evals_used >= self.budget:
            return np.array(F_list)

        # DE Loop
        while evals_used < self.budget and time.time() < deadline - 5.0:
            # Select base point (random)
            i = np.random.randint(pop_size)
            x_base = X_pop[i]
            f_base = F_pop[i]
            
            # Select two other distinct points
            r1, r2 = np.random.choice(pop_size, 2, replace=False)
            while r1 == i or r2 == i or r1 == r2:
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
            
            x_r1 = X_pop[r1]
            x_r2 = X_pop[r2]
            
            # Mutation
            v = x_base + F_scale * (x_r1 - x_r2)
            
            # Crossover
            j_rand = np.random.randint(self.dim)
            trial = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() <= CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = x_base[j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            evals_used += 1
            
            # Selection
            # If trial dominates base, replace base
            if self._dominates(f_trial, f_base):
                X_pop[i] = trial
                F_pop[i] = f_trial
                # Update global archive
                if self._is_non_dominated(f_trial, F_list):
                    F_list = [f_old for f_old in F_list if not self._dominates(f_trial, f_old)]
                    F_list.append(f_trial)
                    X_list.append(trial)
            else:
                # If base dominates trial, discard trial (already in archive or not)
                # If they are non-dominated, we might still want to keep trial in population
                # to maintain diversity, but standard DE replaces only if better.
                # For MOO, we often use a "Pareto-tournament" or keep if non-dominated.
                # Let's keep it in population if it is non-dominated by the population?
                # No, that's expensive. 
                # Let's stick to: replace if dominates. 
                # To ensure diversity, we can occasionally replace if non-dominated by base?
                # No, that leads to stagnation.
                # Better: If trial is non-dominated by the *current population*, we might want to swap it in?
                # Let's try: If trial does not dominate base, but base does not dominate trial,
                # and trial is non-dominated by the rest of the population, replace base with trial?
                # This is risky. 
                
                # Alternative: Just update the global archive if trial is non-dominated there.
                if self._is_non_dominated(f_trial, F_list):
                    F_list = [f_old for f_old in F_list if not self._dominates(f_trial, f_old)]
                    F_list.append(f_trial)
                    X_list.append(trial)

        return np.array(F_list)
# EVOLVE-BLOCK-END
