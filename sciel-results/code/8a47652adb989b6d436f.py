"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Differential Evolution and Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(x, -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, step_size=0.5, max_iter=100):
            """Run Nelder-Mead from x0 with given initial step size."""
            nonlocal calls, best_x, best_f, deadline
            
            n = self.dim
            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += step_size
            
            # Evaluate initial simplex
            simplex_f = np.array([eval_func(simplex[i]) for i in range(n + 1)])
            
            if calls >= self.budget:
                return
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            it = 0
            while calls < self.budget and it < max_iter:
                if time.time() > deadline:
                    break
                it += 1
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_func(x_r)
                if calls >= self.budget:
                    break
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_func(x_e)
                    if calls >= self.budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = eval_func(x_c)
                    if calls >= self.budget:
                        break
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])
                        if calls >= self.budget:
                            break

        def differential_evolution(x0, budget_left):
            """Run Differential Evolution."""
            nonlocal calls, best_x, best_f, deadline
            
            n = self.dim
            rng = np.random.default_rng(123)
            
            # Population size
            npop = 10
            # Reserve budget for Nelder-Mead polish
            de_budget = int(budget_left * 0.8)
            
            # Initialize population
            pop = np.zeros((npop, n))
            pop[0] = x0
            for i in range(1, npop):
                pop[i] = rng.uniform(-self.bound, self.bound, n)
            
            # Evaluate initial population
            pop_f = np.array([eval_func(pop[i]) for i in range(npop)])
            
            # Sort population
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            if calls >= self.budget:
                return
            
            F = 0.8  # Scale factor
            CR = 0.5  # Crossover rate
            
            while calls < self.budget - de_budget:
                if time.time() > deadline:
                    break
                
                # Generate mutant vector
                i = np.random.randint(npop)
                # DE/rand/1
                a, b, c = rng.choice(npop, size=3, replace=False)
                while a == b or a == c or b == c:
                    a, b, c = rng.choice(npop, size=3, replace=False)
                
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                u = np.zeros(n)
                j_rand = rng.randint(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Evaluate trial vector
                f_u = eval_func(u)
                
                # Selection
                if f_u <= pop_f[i]:
                    pop[i] = u
                    pop_f[i] = f_u
                    # Re-sort
                    order = np.argsort(pop_f)
                    pop = pop[order]
                    pop_f = pop_f[order]
                
                # Check convergence
                if pop_f[0] == pop_f[-1]:
                    break

        # Phase 1: Initial exploration
        rng = np.random.default_rng(42)
        
        # Start with a few random points to find a good region
        init_points = [np.zeros(self.dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-4.0, 4.0, self.dim))
        
        for p in init_points:
            eval_func(p)
        
        # Phase 2: Differential Evolution from the best point
        if calls < self.budget:
            differential_evolution(best_x.copy(), self.budget - calls)
        
        # Phase 3: Final fine-tuning with Nelder-Mead
        if calls < self.budget and time.time() < deadline:
            # Use a small step size for polishing
            nelder_mead(best_x.copy(), step_size=0.2, max_iter=100)

        return best_x, best_f
# EVOLVE-BLOCK-END
