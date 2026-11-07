"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        
        # Helper to safely call func and track budget
        def safe_call(x):
            nonlocal best_x, best_f
            if self.calls_made >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            self.calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start sampling
        # Use 30% of budget for random sampling to find good basins
        n_samples = max(10, int(budget * 0.3))
        for i in range(n_samples):
            if time.time() > deadline or self.calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)
            
        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            safe_call(x)

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget for local optimization
        remaining_budget = budget - self.calls_made
        if remaining_budget > 10 and time.time() < deadline:
            nm_budget = min(remaining_budget, int(budget * 0.4))
            self._nelder_mead(func, best_x, nm_budget, dim, deadline, safe_call)
            
        # Phase 3: If budget remains, try another random restart
        if self.calls_made < budget - 10 and time.time() < deadline - 5:
            x = rng.uniform(-5, 5, dim)
            f = safe_call(x)
            if f < best_f:
                # Run a short NM from this new point
                nm_budget = min(budget - self.calls_made, int(budget * 0.2))
                if nm_budget > 10:
                    self._nelder_mead(func, best_x, nm_budget, dim, deadline, safe_call)
            
        return best_x, best_f

    def _nelder_mead(self, func, x0, budget, dim, deadline, safe_call):
        """Nelder-Mead simplex method with strict budget check."""
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            if abs(xi[i]) < 1e-8:
                xi[i] += 0.25
            else:
                xi[i] += 0.25 * np.sign(xi[i])
            simplex.append(xi)
            
        fvals = [safe_call(s) for s in simplex]
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        calls_used = dim + 1
        max_iter = budget
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if self.calls_made >= self.budget:
                break
                
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-10:
                break
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = safe_call(xr)
            calls_used += 1
            
            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = safe_call(xe)
                calls_used += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                fc = safe_call(xc)
                calls_used += 1
                
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = safe_call(simplex[i])
                        calls_used += 1
                        if self.calls_made >= self.budget:
                            return
                            
        return
# EVOLVE-BLOCK-END
