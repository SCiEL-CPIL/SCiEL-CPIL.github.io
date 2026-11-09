import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None

    def __call__(self, func):
        self.func = func
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        calls_made = 0
        max_calls = self.budget
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            return fx

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, max_iters, initial_step=0.5):
            nonlocal calls_made
            dim = self.dim
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += initial_step
            
            fvals = np.zeros(dim + 1)
            fvals[0] = f0
            for i in range(1, dim + 1):
                fvals[i] = evaluate(simplex[i])
                if calls_made >= max_calls:
                    break

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iters):
                if calls_made >= max_calls or time.time() > deadline:
                    break
                
                # Sort
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Convergence check
                if np.abs(fvals[0] - fvals[-1]) < 1e-10:
                    break
                # Check simplex size
                centroid = np.mean(simplex[:-1], axis=0)
                if np.max(np.abs(simplex - centroid)) < 1e-8:
                    break

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = evaluate(x_ref)
                
                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    
                    f_con = evaluate(x_con)
                    
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])
                            if calls_made >= max_calls:
                                break
                        if calls_made >= max_calls:
                            break

            # Return best
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Phase 1: Global Scan
        n_scan1 = 30
        X1 = np.random.uniform(-5, 5, (n_scan1, dim))
        F1 = np.array([evaluate(x) for x in X1])
        
        best_idx1 = np.argmin(F1)
        x_best = X1[best_idx1]
        f_best = F1[best_idx1]

        # Phase 2: Local Search 1
        # Budget: 150 calls
        if calls_made < max_calls:
            x_local1, f_local1 = nelder_mead(x_best, f_best, max_iters=100, initial_step=0.5)
            if f_local1 < f_best:
                x_best = x_local1
                f_best = f_local1

        # Phase 3: Diversity Scan
        # 10 random, 10 Gaussian around best
        n_scan2 = 20
        X2 = np.zeros((n_scan2, dim))
        X2[:10] = np.random.uniform(-5, 5, (10, dim))
        X2[10:] = x_best + np.random.normal(0, 1.0, (10, dim))
        X2 = np.clip(X2, -5, 5)
        
        F2 = np.array([evaluate(x) for x in X2])
        
        best_idx2 = np.argmin(F2)
        if F2[best_idx2] < f_best:
            x_best = X2[best_idx2]
            f_best = F2[best_idx2]

        # Phase 4: Local Search 2
        # Budget: 150 calls
        if calls_made < max_calls:
            x_local2, f_local2 = nelder_mead(x_best, f_best, max_iters=100, initial_step=0.2)
            if f_local2 < f_best:
                x_best = x_local2
                f_best = f_local2

        # Phase 5: Polishing
        # Use remaining budget
        if calls_made < max_calls:
            # Try a few more small steps or just run NM again with smaller step
            x_local3, f_local3 = nelder_mead(x_best, f_best, max_iters=50, initial_step=0.05)
            if f_local3 < f_best:
                x_best = x_local3
                f_best = f_local3

        return x_best, f_best
