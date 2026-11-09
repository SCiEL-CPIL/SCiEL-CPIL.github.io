"""Improved BBOB f22 optimizer: Hybrid DE + Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Differential Evolution and Nelder-Mead for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def nelder_mead(x_start, f_start, max_steps=40, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                    break

        # Phase 1: Differential Evolution
        # DE is robust for global search.
        pop_size = 10
        generations = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(pop_size, n))
        pop_f = np.array([safe_func(x) for x in pop])
        
        # DE Loop
        for gen in range(generations):
            if not check_time() or calls_made >= self.budget:
                break
            
            # Sort population
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            new_pop = []
            for i in range(pop_size):
                if not check_time() or calls_made >= self.budget:
                    break
                
                # Select three distinct random individuals
                idxs = rng.choice(pop_size, 3, replace=False)
                a, b, c = idxs
                while a == i or b == i or c == i or a == b or a == c or b == c:
                    idxs = rng.choice(pop_size, 3, replace=False)
                    a, b, c = idxs
                
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                u = np.zeros(n)
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                u = np.clip(u, -5, 5)
                f_u = safe_func(u)
                
                if f_u <= pop_f[i]:
                    new_pop.append(u)
                    new_pop.append(f_u)
                else:
                    new_pop.append(pop[i])
                    new_pop.append(pop_f[i])
            
            # Reconstruct population
            if len(new_pop) == pop_size * 2:
                pop = np.array(new_pop[::2])
                pop_f = np.array(new_pop[1::2])

        # Phase 2: Nelder-Mead Polish
        # Sort by fitness
        order = np.argsort(pop_f)
        top_k = 5
        remaining = self.budget - calls_made
        nm_budget_per_start = max(20, remaining // (top_k + 1))
        
        for i in range(min(top_k, pop_size)):
            if not check_time() or calls_made >= self.budget:
                break
            x_start = pop[order[i]].copy()
            f_start = pop_f[order[i]]
            delta = 0.5 if i == 0 else 0.3
            nelder_mead(x_start, f_start, max_steps=nm_budget_per_start, initial_delta=delta)

        return best_x, best_f

# EVOLVE-BLOCK-END
