"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining random initialization, local coordinate descent, and evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Phase 1: Random initialization with local coordinate descent
        n_init = 30
        X_list = []
        F_list = []
        used = 0
        
        # Generate initial random points
        X_rand = lo + span * rng.random((n_init, self.dim))
        for x in X_rand:
            if used >= self.budget or time.time() > deadline:
                break
            f = problem(x)
            used += 1
            X_list.append(x.copy())
            F_list.append(f)
        
        # Apply coordinate descent to each initial point
        for i in range(len(X_list)):
            if used >= self.budget or time.time() > deadline:
                break
            x = X_list[i].copy()
            f = F_list[i].copy()
            # Coordinate descent with ternary search on each dimension
            for dim_idx in range(self.dim):
                if used >= self.budget or time.time() > deadline:
                    break
                # Ternary search over this dimension
                a = lo[dim_idx]
                b = hi[dim_idx]
                for _ in range(3):  # 3 iterations of ternary search
                    if used >= self.budget or time.time() > deadline:
                        break
                    m1 = a + (b - a) / 3
                    m2 = b - (b - a) / 3
                    x1 = x.copy(); x1[dim_idx] = m1
                    x2 = x.copy(); x2[dim_idx] = m2
                    f1 = problem(x1)
                    used += 1
                    if used >= self.budget or time.time() > deadline:
                        break
                    f2 = problem(x2)
                    used += 1
                    # Use sum of objectives as scalar proxy for comparison
                    if np.sum(f1) < np.sum(f2):
                        b = m2
                    else:
                        a = m1
                
                # Check budget before final midpoint evaluation
                if used < self.budget and time.time() < deadline:
                    # Evaluate at the midpoint of remaining interval
                    mid = (a + b) / 2
                    x_new = x.copy()
                    x_new[dim_idx] = mid
                    f_new = problem(x_new)
                    used += 1
                    # Accept if it improves the scalar sum
                    if np.sum(f_new) < np.sum(f):
                        x = x_new
                        f = f_new
            
            # Update the population with the improved point
            X_list[i] = x
            F_list[i] = f
        
        X = np.array(X_list)
        F = np.array(F_list)
        
        # Phase 2: Evolutionary search with better mutation
        pop_size = len(X)
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            parents = X[~dominated] if np.any(~dominated) else X
            F_parents = F[~dominated] if np.any(~dominated) else F
            
            # Select parent with tournament selection
            n_tournament = min(3, len(parents))
            tournament_idx = rng.integers(len(parents), size=n_tournament)
            # Select the one with best scalar sum among tournament
            best_parent_idx = tournament_idx[np.argmin(np.sum(F_parents[tournament_idx], axis=1))]
            parent = parents[best_parent_idx]
            
            # Gaussian mutation with adaptive step size
            step_size = 0.15
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace the worst point based on scalar sum
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        return X
# EVOLVE-BLOCK-END
