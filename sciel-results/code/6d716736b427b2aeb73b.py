"""Nelder-Mead Simplex with multi-restart for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, dim, deadline, calls_made, best_x, best_f):
        """
        Runs Nelder-Mead optimization starting from x0.
        Returns updated best_x, best_f, and calls_made.
        """
        n = dim
        # Initial simplex: x0 and x0 + delta * e_i
        delta = 2.0
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            v[i] += delta
            simplex.append(v)
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            if calls_made >= budget or time.time() > deadline:
                return best_x, best_f, calls_made
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
            calls_made += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()
        
        # Sort simplex by function value
        idx = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in idx]
        fvals = fvals[idx].copy()

        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        max_iter = 2000
        iter_count = 0
        
        while iter_count < max_iter:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Check for convergence: if the spread of function values is small
            if np.ptp(fvals) < 1e-8:
                break
            
            # Sort again to ensure order
            idx = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in idx]
            fvals = fvals[idx].copy()
            
            # Centroid of all points except the worst
            centroid = np.sum(simplex[:n], axis=0) / n
            
            # 1. Reflection
            xr = centroid + alpha * (centroid - simplex[n])
            xr = np.clip(xr, -5, 5)
            if calls_made >= budget or time.time() > deadline:
                break
            fr = func(xr)
            calls_made += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
            
            if fr < fvals[0]:
                # 2. Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                if calls_made >= budget or time.time() > deadline:
                    break
                fe = func(xe)
                calls_made += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                
                if fe < fr:
                    simplex[n] = xe
                    fvals[n] = fe
                else:
                    simplex[n] = xr
                    fvals[n] = fr
            elif fr < fvals[n-1]:
                # Reflection is good, but not better than best
                simplex[n] = xr
                fvals[n] = fr
            else:
                # 3. Contraction
                if fr < fvals[n]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (xr - centroid)
                
                xc = np.clip(xc, -5, 5)
                if calls_made >= budget or time.time() > deadline:
                    break
                fc = func(xc)
                calls_made += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                
                if fc < min(fr, fvals[n]):
                    simplex[n] = xc
                    fvals[n] = fc
                else:
                    # 4. Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_made >= budget or time.time() > deadline:
                            return best_x, best_f, calls_made
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                    # Reset delta if we shrink too much? 
                    # For now, just let it converge or break via stagnation logic below
                    # Actually, if we shrink, the simplex becomes very small. 
                    # We should check if we are stuck.
                    if np.ptp(fvals) < 1e-10:
                        break

            iter_count += 1
            
            # Stagnation check: if no improvement in best_f for a while, break to restart
            # We can track this externally, but here we just let it run until convergence or time/budget
            # A better strategy is to break if the simplex size becomes tiny.
            simplex_size = np.max(np.linalg.norm(np.array(simplex) - simplex[0], axis=1))
            if simplex_size < 1e-4:
                break

        return best_x, best_f, calls_made

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Multi-start strategy
        # We try several random starting points. If a run doesn't improve the global best
        # significantly or takes too long, we move to the next.
        
        # Initial set of diverse starting points
        # Use a grid-like or deterministic random set to ensure coverage
        rng = np.random.default_rng(42)
        
        # Generate a pool of start points
        # 1. Origin
        # 2. Corners (scaled)
        # 3. Random points
        
        start_points = []
        start_points.append(np.zeros(dim))
        
        # Add some structured points
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 4.0; start_points.append(p1)
            p2 = np.zeros(dim); p2[i] = -4.0; start_points.append(p2)
            
        # Add random points
        for _ in range(10):
            start_points.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle start points to mix deterministic and random
        rng.shuffle(start_points)

        # Allocate budget per start
        # If we have 500 calls, and 15 start points, that's ~33 calls each.
        # Nelder-Mead in 5D needs at least 6 calls for init. 
        # 33 calls is enough for a few iterations.
        # Let's be adaptive: give more budget to starts that find good solutions.
        
        current_best_f = np.inf
        min_improvement = 1e-5
        
        for idx, x0 in enumerate(start_points):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Check if we've already found a very good solution?
            # f_opt is in [-100, 100]. If best_f is close to -100, we might stop.
            # But we don't know f_opt. So we just run through the list.
            
            # If this start point is already worse than current best by a lot, maybe skip?
            # No, because the function is rotated/shifted, so a "bad" point might be in a basin.
            
            # Run Nelder-Mead from this start
            # Limit calls for this specific run to avoid getting stuck
            # If we have 500 total, and 15 starts, allow ~30 calls each.
            # But if we find a very good point, we might want to polish it.
            
            # Strategy: 
            # 1. Run NM for a limited number of calls (e.g., 20-30)
            # 2. If it improves the global best, continue running NM from the new best with a smaller initial delta?
            #    Or just let the main loop handle restarts.
            
            # Simpler: Just run NM until it converges or uses up its allocated slice.
            # Let's allocate a fixed slice of budget for each start, but allow the last few to use remaining.
            
            slice_budget = max(10, budget // len(start_points))
            
            # If we are near the end of the budget, give more time
            if idx == len(start_points) - 1:
                 slice_budget = budget - calls_made
            
            prev_best_f = best_f
            best_x, best_f, calls_made = self._nelder_mead(
                func, x0, budget, dim, deadline, calls_made, best_x, best_f
            )
            
            # If we didn't improve, or improved very little, and we have budget left,
            # we might want to do a "polish" run from the current best with a very small simplex.
            # But the next start point might be better.
            
        # Final Polish
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Run NM again from the best point found, but with a very small initial delta
            # to ensure we hit the local minimum precisely.
            x0 = best_x.copy()
            # We need to modify _nelder_mead to accept delta or create a variant
            # Let's just call it again. It will start with delta=2.0 which might be too big.
            # Let's create a specific polish step.
            
            # Simple coordinate descent polish
            for i in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break
                step = 0.1
                for _ in range(20):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    y = best_x.copy()
                    y[i] += step
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_x = y
                        best_f = fy
                    else:
                        y = best_x.copy()
                        y[i] -= step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < best_f:
                            best_x = y
                            best_f = fy
                        else:
                            step *= 0.5
                            if step < 1e-4:
                                break
                    step *= 0.5

        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            if calls_made < budget:
                best_f = func(x)
                calls_made += 1
            else:
                best_f = 0.0
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
