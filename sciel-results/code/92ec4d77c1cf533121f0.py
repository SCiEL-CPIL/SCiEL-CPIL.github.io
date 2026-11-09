"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer with Sobol initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sample(self, n, num_samples, seed):
        """Generate Sobol sequence points in [-5, 5]^n."""
        # Use scipy if available for better Sobol, else fallback to halton or random
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n, scramble=True, seed=seed)
            # Generate 2^k points, take first num_samples
            k = max(1, int(np.ceil(np.log2(num_samples))))
            points = sampler.random(2**k)
            # Scale from [0, 1] to [-5, 5]
            points = 10.0 * points - 5.0
            return points[:num_samples]
        except ImportError:
            # Fallback to Halton-like or just stratified random
            # Simple stratified random for robustness
            rng = np.random.default_rng(seed)
            points = np.zeros((num_samples, n))
            for i in range(n):
                # Divide range into num_samples bins, pick random point in each
                bins = np.linspace(-5, 5, num_samples + 1)
                for j in range(num_samples):
                    points[j, i] = rng.uniform(bins[j], bins[j+1])
            return points

    def _nelder_mead(self, func, x0, budget, rng, deadline):
        """
        Run Nelder-Mead from x0.
        Returns best_x, best_f, calls_used.
        """
        n = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        f_vals = [func(x0)]
        calls = 1
        
        # Expand simplex
        for i in range(n):
            xi = x0.copy()
            # Initial step size
            step = 0.5
            # Ensure within bounds
            if xi[i] + step > 5:
                xi[i] += 5 - xi[i]
            elif xi[i] - step < -5:
                xi[i] -= xi[i] + 5
            else:
                xi[i] += step
            simplex.append(xi)
            f_vals.append(func(xi))
            calls += 1
            if calls >= budget or time.time() >= deadline:
                break
        
        if len(simplex) < 2:
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx], calls

        best_x = simplex[np.argmin(f_vals)].copy()
        best_f = min(f_vals)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while calls < budget and time.time() < deadline:
            # Sort simplex by fitness
            indices = np.argsort(f_vals)
            simplex = [simplex[i] for i in indices]
            f_vals = [f_vals[i] for i in indices]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            
            if fr < f_vals[0]:
                # Expand
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
            elif fr < f_vals[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_vals[-1] = fr
            else:
                # Contract
                if fr < f_vals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                
                if fc <= min(fr, f_vals[-1]):
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget or time.time() >= deadline:
                            break
                    if calls >= budget or time.time() >= deadline:
                        break
            
            # Update best
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < best_f:
                best_f = f_vals[min_idx]
                best_x = simplex[min_idx].copy()
                
            # Check for convergence (simplex size very small)
            # If simplex is too small, we might be stuck, but NM handles this via shrinking
            # We rely on budget/time for termination to maximize early progress
            # Add a small check: if best_f hasn't improved in last 10 iterations, maybe break?
            # No, let it run to use budget, but NM can get stuck in local mins.
            # However, f14 is unimodal, so local min = global min.
            
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # 5 second safety margin
        n = self.dim
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Global exploration using Sobol sequence
        # Use 30% of budget for exploration
        explore_budget = max(20, int(self.budget * 0.3))
        
        # Generate Sobol points
        sobol_points = self._sobol_sample(n, explore_budget, seed=42)
        
        for i in range(explore_budget):
            if time.time() >= deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            total_calls += 1
            
        # Phase 2: Local refinement using Nelder-Mead
        # We can run NM from the best point found
        # If we have budget left, run NM
        remaining = self.budget - total_calls
        if remaining > 5:
            x_opt, f_opt, calls_used = self._nelder_mead(
                func, best_x, remaining, rng, deadline
            )
            total_calls += calls_used
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
                
        # Phase 3: If still budget left, try another NM from a slightly perturbed best
        # Or restart NM if it converged too early
        remaining = self.budget - total_calls
        if remaining > 10 and time.time() < deadline:
            # Perturb best solution slightly to escape potential flat spots (though f14 is smooth)
            # Or just run NM again from best_x with a fresh simplex
            x_opt2, f_opt2, calls_used2 = self._nelder_mead(
                func, best_x, remaining, rng, deadline
            )
            total_calls += calls_used2
            if f_opt2 < best_f:
                best_f = f_opt2
                best_x = x_opt2.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
