"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start Nelder-Mead optimizer for Büche-Rastrigin f4.
    
    Uses Halton sequence for well-distributed initial points to ensure early
    rapid convergence in multiple basins, maximizing early AOCC.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _halton_sequence(self, num_points, dim, skip=1):
        """Generate Halton sequence points in [0, 1)^dim, skipping first 'skip' points."""
        points = []
        # First primes for each dimension
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for n in range(skip, skip + num_points):
            point = []
            for d in range(dim):
                p = primes[d]
                f = 1.0
                i = n
                while i > 0:
                    f /= p
                    i = i // p
                point.append(f)
            points.append(point)
        return np.array(points)

    def _nelder_mead(self, func, x0, budget, deadline, dim):
        """Run Nelder-Mead from x0, returning (best_x, best_f, calls_made)."""
        # Initial simplex: x0 and x0 + delta*e_i
        delta = 0.5
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0.copy()
            v[i] += delta
            simplex.append(v)
        
        # Evaluate initial simplex
        fvals = []
        calls_made = 0
        for i in range(dim + 1):
            if calls_made >= budget or time.time() >= deadline:
                break
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals.append(func(simplex[i]))
            calls_made += 1
        
        if len(fvals) < dim + 1:
            min_idx = np.argmin(fvals) if fvals else 0
            if fvals:
                return simplex[min_idx].copy(), fvals[min_idx], calls_made
            return x0.copy(), np.inf, calls_made

        # Nelder-Mead iteration
        alpha, gamma, rho = 1.0, 2.0, 0.5  # reflection, expansion, contraction
        max_iter = 100
        
        for iteration in range(max_iter):
            if calls_made >= budget or time.time() >= deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made >= budget or time.time() >= deadline:
                            break
                    if calls_made >= budget or time.time() >= deadline:
                        break
            
            # Check convergence
            if len(fvals) > 0:
                spread = np.std(fvals)
                if spread < 1e-10:
                    break
        
        # Find best in final simplex
        min_idx = np.argmin(fvals)
        return simplex[min_idx].copy(), fvals[min_idx], calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Halton sequence initialization - 25 well-distributed points
        # Map from [0,1)^dim to [-5,5]^dim
        num_halton = 25
        halton_unit = self._halton_sequence(num_halton, dim, skip=1)
        
        # Scale to [-5, 5]
        halton_points = halton_unit * 10.0 - 5.0
        
        # Also add some key structured points
        key_points = [
            np.zeros(dim),  # Origin
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            np.array([1.0] * dim),
            np.array([-1.0] * dim),
            np.array([0.0, 2.0, 0.0, -2.0, 0.0]),
            np.array([2.0, 0.0, -2.0, 0.0, 2.0]),
        ]
        
        # Combine and deduplicate
        all_init_points = list(halton_points) + key_points
        seen = set()
        unique_points = []
        for pt in all_init_points:
            pt_c = np.clip(pt, -5, 5)
            key = tuple(np.round(pt_c, 4))
            if key not in seen:
                seen.add(key)
                unique_points.append(pt_c)
        
        # Evaluate all initial points
        init_results = []
        for pt in unique_points:
            if calls_made >= total_budget or time.time() >= deadline:
                break
            f = evaluate(pt)
            init_results.append((f, pt.copy()))
        
        # Sort by function value - best first
        init_results.sort(key=lambda x: x[0])
        
        # Phase 2: Run Nelder-Mead from top candidates
        # Allocate budget: give more to better initial points
        # Use top 10-15 points, with budget proportional to their rank
        num_nm_runs = min(12, len(init_results))
        
        # Budget allocation: better points get more budget
        # Total remaining budget
        remaining = total_budget - calls_made
        if remaining > 0:
            # Allocate budget: top point gets most, decreasing
            # Use a simple allocation: 40% to top, then decreasing
            budgets = []
            for i in range(num_nm_runs):
                if i == 0:
                    b = int(remaining * 0.35)
                elif i == 1:
                    b = int(remaining * 0.20)
                elif i == 2:
                    b = int(remaining * 0.15)
                else:
                    b = int(remaining * 0.08)
                budgets.append(max(b, 5))  # At least 5 calls each
            
            # Ensure total doesn't exceed remaining
            total_alloc = sum(budgets)
            if total_alloc > remaining:
                scale = remaining / total_alloc
                budgets = [max(int(b * scale), 3) for b in budgets]
            
            for i in range(num_nm_runs):
                if calls_made >= total_budget or time.time() >= deadline:
                    break
                
                x0 = init_results[i][1]
                nm_budget = min(budgets[i], total_budget - calls_made)
                if nm_budget < 5:
                    continue
                
                nm_x, nm_f, nm_calls = self._nelder_mead(
                    evaluate, x0, nm_budget, deadline, dim
                )
                
                # Update best
                if nm_f < best_f:
                    best_f = nm_f
                    best_x = nm_x.copy()
        
        # Phase 3: If budget remains, do a few more local refinements
        if calls_made < total_budget and time.time() < deadline and best_x is not None:
            # Try Nelder-Mead from best point with smaller delta
            remaining = total_budget - calls_made
            if remaining > 10:
                # Create a new simplex with smaller step
                x0 = best_x.copy()
                delta_small = 0.1
                simplex = [x0.copy()]
                for i in range(dim):
                    v = x0.copy()
                    v[i] += delta_small
                    simplex.append(v)
                
                fvals = []
                for i in range(dim + 1):
                    if calls_made >= total_budget or time.time() >= deadline:
                        break
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    fvals.append(func(simplex[i]))
                    calls_made += 1
                
                if len(fvals) == dim + 1:
                    # Quick Nelder-Mead refinement
                    alpha, gamma, rho = 1.0, 2.0, 0.5
                    for _ in range(50):
                        if calls_made >= total_budget or time.time() >= deadline:
                            break
                        
                        order = np.argsort(fvals)
                        simplex = [simplex[i].copy() for i in order]
                        fvals = [fvals[i] for i in order]
                        
                        centroid = np.mean(simplex[:-1], axis=0)
                        worst = simplex[-1]
                        f_worst = fvals[-1]
                        
                        x_r = centroid + alpha * (centroid - worst)
                        x_r = np.clip(x_r, -5, 5)
                        f_r = func(x_r)
                        calls_made += 1
                        
                        if f_r < fvals[0]:
                            x_e = centroid + gamma * (x_r - centroid)
                            x_e = np.clip(x_e, -5, 5)
                            f_e = func(x_e)
                            calls_made += 1
                            if f_e < f_r:
                                simplex[-1] = x_e
                                fvals[-1] = f_e
                            else:
                                simplex[-1] = x_r
                                fvals[-1] = f_r
                        elif f_r < fvals[-2]:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                        else:
                            if f_r < f_worst:
                                x_c = centroid + rho * (x_r - centroid)
                            else:
                                x_c = centroid - rho * (worst - centroid)
                            x_c = np.clip(x_c, -5, 5)
                            f_c = func(x_c)
                            calls_made += 1
                            
                            if f_c < min(f_r, f_worst):
                                simplex[-1] = x_c
                                fvals[-1] = f_c
                            else:
                                for j in range(1, dim + 1):
                                    simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                    simplex[j] = np.clip(simplex[j], -5, 5)
                                    fvals[j] = func(simplex[j])
                                    calls_made += 1
                                    if calls_made >= total_budget or time.time() >= deadline:
                                        break
                                if calls_made >= total_budget or time.time() >= deadline:
                                    break
                        
                        if np.std(fvals) < 1e-10:
                            break
                    
                    min_idx = np.argmin(fvals)
                    if fvals[min_idx] < best_f:
                        best_f = fvals[min_idx]
                        best_x = simplex[min_idx].copy()

        # Ensure we don't exceed budget
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
