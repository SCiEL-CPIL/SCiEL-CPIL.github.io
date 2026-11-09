import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        """Compute the Halton sequence value for a given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, n, dim, offset=0):
        """Generate n points in [0,1)^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._halton(i + 1 + offset, j + 2)
        return points

    def _nelder_mead(self, func, x0, max_calls, tol=1e-12, deadline=None):
        """
        Adaptive Nelder-Mead simplex method.
        Returns (best_x, best_f, calls_used).
        """
        dim = len(x0)
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.1 if x0[i - 1] != 0 else 0.1
        
        # Evaluate initial simplex
        f_vals = np.zeros(dim + 1)
        calls_used = 0
        
        # Check if we have enough budget for initial simplex
        if max_calls < dim + 1:
            # If not enough budget, just evaluate as many as possible
            for i in range(min(max_calls, dim + 1)):
                if deadline and time.time() > deadline:
                    break
                f_vals[i] = func(simplex[i])
                calls_used += 1
            if calls_used == 0:
                return x0, np.inf, 0
            best_idx = np.argmin(f_vals[:calls_used])
            return simplex[best_idx].copy(), f_vals[best_idx], calls_used

        for i in range(dim + 1):
            if deadline and time.time() > deadline:
                best_idx = np.argmin(f_vals[:calls_used]) if calls_used > 0 else 0
                return simplex[best_idx], f_vals[best_idx] if calls_used > 0 else np.inf, calls_used
            f_vals[i] = func(simplex[i])
            calls_used += 1
        
        best_idx = np.argmin(f_vals)
        best_x = simplex[best_idx].copy()
        best_f = f_vals[best_idx]
        
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrinkage
        
        while calls_used < max_calls:
            if deadline and time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            # Check convergence
            if abs(f_vals[0] - f_vals[-1]) < tol:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            
            if calls_used >= max_calls: break
            fr = func(xr)
            calls_used += 1
            
            if fr < f_vals[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                
                if calls_used >= max_calls:
                    # No budget for expansion, accept reflection if better
                    if fr < f_vals[-1]:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                    break
                
                fe = func(xe)
                calls_used += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
            elif fr < f_vals[-2]:
                # Reflection is better than worst but not best
                simplex[-1] = xr
                f_vals[-1] = fr
            else:
                # Contraction
                if fr < f_vals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                
                xc = np.clip(xc, -5, 5)
                
                if calls_used >= max_calls:
                    # No budget for contraction, accept reflection if better
                    if fr < f_vals[-1]:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                    break
                
                fc = func(xc)
                calls_used += 1
                
                if fc < min(fr, f_vals[-1]):
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    # Shrink
                    # We need dim calls for shrinking, check budget
                    if calls_used + dim > max_calls:
                        break
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = func(simplex[i])
                        calls_used += 1
            
            # Update best
            current_best_idx = np.argmin(f_vals)
            if f_vals[current_best_idx] < best_f:
                best_f = f_vals[current_best_idx]
                best_x = simplex[current_best_idx].copy()
        
        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        
        # Phase 1: Halton sequence exploration
        # Use first 100 evaluations for Halton
        n_halton = min(100, budget)
        halton_points = self._halton_sequence(n_halton, dim)
        halton_points = halton_points * 10.0 - 5.0  # Map to [-5, 5]
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_halton):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            x = halton_points[i]
            f_val = func(x)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead local search from best point
        if best_x is not None and calls_made < budget:
            remaining_budget = budget - calls_made
            
            nm_x, nm_f, nm_calls = self._nelder_mead(
                func, best_x, max_calls=remaining_budget, deadline=deadline
            )
            calls_made += nm_calls
            
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x
        
        # If we still have budget and time, do another round of exploration + refinement
        if calls_made < budget and time.time() < deadline - 10:
            # Use a few random points around the best
            rng = np.random.default_rng(42)
            n_extra = min(20, budget - calls_made)
            
            for _ in range(n_extra):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Generate point near best
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                f_val = func(x)
                calls_made += 1
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
            
            # Final Nelder-Mead refinement
            if calls_made < budget and best_x is not None:
                remaining = budget - calls_made
                if remaining > 10:
                    nm_x, nm_f, nm_calls = self._nelder_mead(
                        func, best_x, max_calls=remaining, deadline=deadline
                    )
                    calls_made += nm_calls
                    if nm_f < best_f:
                        best_f = nm_f
                        best_x = nm_x
        
        return best_x, best_f
