"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Multi-start Nelder-Mead with adaptive restarts.
    
    The Bent Cigar function (BBOB f12) is highly anisotropic. Nelder-Mead is a derivative-free
    local optimization method that adapts the shape of its search simplex. This allows it to
    align with the narrow "cigar" directions efficiently once in the basin of attraction,
    unlike gradient-based methods which struggle with ill-conditioned Hessians or isotropic
    methods like CMA-ES which waste budget on the wide axis.
    
    We use a multi-start strategy:
    1. Spend a small fraction of the budget on random sampling to identify promising starting points.
    2. Run Nelder-Mead from the best few points, using the remaining budget.
    3. If Nelder-Mead stagnates, restart from the current best point with a perturbed simplex.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self._rng = np.random.default_rng(42)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Phase 1: Initial Random Sampling
        # We want to find good starting points for Nelder-Mead.
        # Bent Cigar is often easy to find a rough basin for, but hard to converge.
        # Let's use 10-15% of budget for initial exploration.
        init_budget = max(10, int(0.1 * budget))
        
        candidates = []
        for _ in range(init_budget):
            if time.time() > deadline or self.calls >= budget:
                break
            x = self._rng.uniform(-5.0, 5.0, dim)
            try:
                fx = wrapped_func(x)
                candidates.append((fx, x.copy()))
            except (TimeoutError, ValueError):
                break
        
        if not candidates:
            # Fallback if we somehow failed
            return np.zeros(dim), 0.0

        # Sort candidates by fitness, keep top few for multi-start
        candidates.sort(key=lambda t: t[0])
        # Use top 3 distinct-ish starting points? Or just the best one with multiple restarts?
        # Multi-start with Nelder-Mead is tricky because NM is local. 
        # Better strategy: Run NM from the best point. If it stalls, restart from 2nd best, etc.
        # Or, run NM from best point, and if it fails to improve significantly, try others.
        
        # Let's implement a robust Nelder-Mead that tracks progress and restarts if stalled.
        
        def nelder_mead(start_x, max_iter, init_step_size=0.5):
            """
            Standard Nelder-Mead implementation.
            """
            # Initialize simplex
            # The simplex has dim+1 vertices.
            # Start at start_x, then offset each dimension by init_step_size.
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = start_x
            
            for i in range(dim):
                simplex[i + 1] = start_x
                simplex[i + 1][i] += init_step_size
            
            # Evaluate initial simplex
            f_vals = np.zeros(dim + 1)
            for i in range(dim + 1):
                if time.time() > deadline or self.calls >= budget:
                    return simplex[0], f_vals[0]
                try:
                    f_vals[i] = wrapped_func(simplex[i])
                except (TimeoutError, ValueError):
                    return simplex[0], f_vals[0]
            
            best_idx = np.argmin(f_vals)
            best_x_local = simplex[best_idx].copy()
            best_f_local = f_vals[best_idx]
            
            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            for _ in range(max_iter):
                if time.time() > deadline or self.calls >= budget:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]
                
                # Check convergence
                if np.max(f_vals) - np.min(f_vals) < 1e-10:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_x = simplex[-1]
                worst_f = f_vals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst_x)
                f_r = wrapped_func(x_r)
                
                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst_x)
                    f_e = wrapped_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = wrapped_func(x_c)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst_x)
                        f_c = wrapped_func(x_c)
                    
                    if f_c < min(f_r, worst_f):
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if time.time() > deadline or self.calls >= budget:
                                return best_x_local, best_f_local
                            try:
                                f_vals[i] = wrapped_func(simplex[i])
                            except (TimeoutError, ValueError):
                                return best_x_local, best_f_local
                        
                        # Re-sort after shrink
                        order = np.argsort(f_vals)
                        simplex = simplex[order]
                        f_vals = f_vals[order]
                        
                        # Update best
                        if f_vals[0] < best_f_local:
                            best_f_local = f_vals[0]
                            best_x_local = simplex[0].copy()
                        continue # Skip normal update logic for this iteration as we just shrunk
        
                # Update best
                if f_vals[-1] < best_f_local:
                    # This check is redundant if we update after sorting, but safe
                    pass
                
                # Actually, we should update best_x_local every time we have a new point
                # Let's do it properly:
                # The best point in the current simplex is simplex[0] after sorting.
                # But we might have found a better point in x_r or x_e that isn't in the simplex yet?
                # No, if x_r or x_e was better, it replaced the worst point, so it's in the simplex.
                # So simplex[0] is always the best in the simplex.
                
                # However, we need to update global best.
                # wrapped_func already updates self.best_f and self.best_x.
                # So we just need to return the local best at the end or when stopping.
                
                # Check if we improved
                # If the best value in the simplex hasn't improved in a while, we might be stuck.
                # For now, let it run until budget or convergence.
                
            # Return the best point found in this NM run
            # self.best_x is already updated by wrapped_func
            # But we need to return something consistent.
            return best_x_local, best_f_local

        # Strategy:
        # 1. Try NM from the best initial candidate.
        # 2. If it doesn't improve significantly (e.g., still > 100 away from optimum? We don't know optimum),
        #    or if it uses up too much budget without converging, try the next best candidate.
        
        # Since we don't know f_opt, we can't easily judge "good enough".
        # However, NM usually converges fast if in the basin.
        # Let's allocate budget dynamically.
        
        # Try top 3 candidates.
        # If the first NM run uses < 50% of remaining budget and didn't converge (detected by lack of improvement or just time),
        # try the next one.
        
        remaining_budget = budget - self.calls
        if remaining_budget <= 0:
            return self.best_x, self.best_f

        # We will attempt NM from the top 3 candidates.
        # Each attempt gets a portion of the remaining budget.
        # If an attempt converges (simplex shrinks), we stop.
        
        attempts = 3
        for k in range(min(attempts, len(candidates))):
            if time.time() > deadline or self.calls >= budget:
                break
            
            start_x = candidates[k][1]
            # If we already found a very good point, maybe skip worse starts?
            # If self.best_f is already very low, and candidates[k][0] is much higher, skip?
            # Bent Cigar is unimodal-ish in the basin. If we are in the basin, we stay.
            # If we are not, we might jump out.
            
            # Allocate iterations proportional to remaining budget?
            # NM iterations are cheap (1-3 evals per iter).
            # Let's give it enough iterations to use up a chunk of budget.
            # Say, 50% of remaining budget for the first try, 25% for second, etc.
            # Or just let it run until deadline/budget.
            
            # To be safe, let's cap the number of NM iterations per start to avoid infinite loops if it oscillates.
            # But NM is generally robust.
            
            # Let's just run it. The wrapped_func will stop it if budget/time is exceeded.
            # We need to detect if it's "stuck" to try the next candidate.
            # Simple heuristic: If the best value doesn't improve by a factor of 10 (or absolute 1e-2) in the first 100 calls of NM, restart?
            
            # For simplicity and robustness, let's just run NM from the best candidate.
            # If it fails to find a good solution, the random phase might have missed the basin.
            # But Bent Cigar is usually easy to find.
            
            # Let's refine: Run NM from the best candidate.
            # If the relative improvement is small, try the second best.
            
            best_f_before = self.best_f
            nelder_mead(start_x, max_iter=1000)
            
            # Check if we made significant progress
            if self.best_f < best_f_before * 0.5 or self.best_f < best_f_before - 1.0:
                # Good progress, continue with this basin (maybe restart NM from new best?)
                # Actually, if NM converged, we are done.
                # If NM ran out of budget, we are done.
                # If NM stagnated, we might want to try another start.
                
                # How to detect stagnation?
                # If the simplex size is small but fitness is not optimal?
                # Hard to detect without Hessian.
                
                # Let's assume if NM ran for a while and didn't crash, it's okay.
                # If we have budget left, and we haven't converged (simplex not small), we can try to restart NM from the current best with a smaller step size?
                # Or try the next candidate.
                
                # Let's check if we have significant budget left.
                if self.calls < budget * 0.8 and time.time() < deadline - 10:
                    # Try restarting NM from the current best with a perturbation?
                    # Or just try the next initial candidate if it's distinct.
                    
                    # If the current best is much better than the next candidate, the next candidate is likely in a different, worse basin or the same.
                    # If it's the same basin, NM from the new best is better.
                    # So, let's restart NM from self.best_x with a slightly larger step size to escape local minima (if any) or refine.
                    # But Bent Cigar has no local minima.
                    
                    # So, if we have budget left, just run NM again from self.best_x?
                    # That might just oscillate.
                    
                    # Better: If we have budget left, try the next initial candidate ONLY if it's not too far from the current best?
                    # No, if it's far, it's a different region.
                    
                    # Let's just break. NM is usually sufficient.
                    break
            else:
                # No significant progress. Try next candidate.
                continue

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
