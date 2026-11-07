"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multiple restarts for BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, budget, dim, rng, start_x, deadline):
        """Run Nelder-Mead simplex and return (best_x, best_f, evals_used)."""
        # Initialize simplex
        x0 = start_x.copy()
        alpha = 0.5  # initial step size
        
        # Create simplex vertices
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += alpha
        
        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        evals = 0
        for i in range(dim + 1):
            if time.time() > deadline or evals >= budget:
                break
            try:
                fvals[i] = func(simplex[i])
            except Exception:
                fvals[i] = np.inf
            evals += 1
        
        if evals == 0:
            return x0, np.inf, 0
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Nelder-Mead parameters
        alpha_nm = 1.0   # reflection
        gamma = 2.0      # expansion
        rho = 0.5        # contraction
        sigma_nm = 0.5   # shrink
        
        max_iter = max(1, budget - evals)
        
        for it in range(max_iter):
            if time.time() > deadline or evals >= budget:
                break
            
            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha_nm * (centroid - simplex[-1])
            np.clip(x_ref, -5, 5, out=x_ref)
            try:
                f_ref = func(x_ref)
            except Exception:
                f_ref = np.inf
            evals += 1
            
            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                np.clip(x_exp, -5, 5, out=x_exp)
                try:
                    f_exp = func(x_exp)
                except Exception:
                    f_exp = np.inf
                evals += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Just reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_ref - centroid)
                np.clip(x_con, -5, 5, out=x_con)
                try:
                    f_con = func(x_con)
                except Exception:
                    f_con = np.inf
                evals += 1
                
                if f_con <= f_ref:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        np.clip(simplex[i], -5, 5, out=simplex[i])
                        try:
                            fvals[i] = func(simplex[i])
                        except Exception:
                            fvals[i] = np.inf
                        evals += 1
            
            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break
        
        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x_global = None
        best_f_global = np.inf
        
        rng = np.random.default_rng(42)
        
        # Strategy: Multiple restarts of Nelder-Mead from different starting points
        # Phase 1: A few random starts with large initial simplex
        # Phase 2: Refinement around best found point
        
        num_restarts = 8
        per_restart_budget = max(1, budget // num_restarts)
        
        # First, try center
        starts = [np.zeros(dim)]
        
        # Add random starts
        for _ in range(num_restarts - 1):
            starts.append(rng.uniform(-4, 4, dim))
        
        for start_x in starts:
            if time.time() > deadline:
                break
            if budget <= 0:
                break
            
            try:
                best_x, best_f, evals_used = self._nelder_mead(
                    func, per_restart_budget, dim, rng, start_x, deadline
                )
                budget -= evals_used
                
                if best_f < best_f_global:
                    best_f_global = best_f
                    best_x_global = best_x
            except Exception:
                continue
        
        # Phase 2: Local refinement around the best point found
        if best_x_global is not None and budget > 10 and time.time() < deadline:
            # Do a few more Nelder-Mead runs with smaller initial step around best
            for r in range(3):
                if time.time() > deadline or budget <= 5:
                    break
                
                perturb_scale = 0.5 * (0.5 ** r)
                start_x = best_x_global + perturb_scale * rng.standard_normal(dim)
                start_x = np.clip(start_x, -5, 5)
                
                local_budget = min(budget, max(5, budget // (3 - r)))
                
                try:
                    best_x, best_f, evals_used = self._nelder_mead(
                        func, local_budget, dim, rng, start_x, deadline
                    )
                    budget -= evals_used
                    
                    if best_f < best_f_global:
                        best_f_global = best_f
                        best_x_global = best_x
                except Exception:
                    continue
        
        # Fallback if nothing was found
        if best_x_global is None:
            x = rng.uniform(-5, 5, dim)
            try:
                f = func(x)
            except Exception:
                f = np.inf
            best_x_global = x
            best_f_global = f
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
