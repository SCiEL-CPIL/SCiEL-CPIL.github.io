# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer for BBOB-style f13 (Sharp Ridge).
    
    Combines:
    1. Nelder-Mead simplex for robust global convergence.
    2. Powell's conjugate direction method for exploiting ridge structure.
       This rotates search axes to align with the ridge, far more efficient
       than coordinate descent on rotated landscapes.
    3. Adaptive switching between NM and Powell based on stall detection.
    4. Strict time and budget guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization - sample to find good basin
        n_init = 12
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            timed_call(x)
            
        if best_x is None:
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        x = best_x.copy()
        fx = best_f

        # Phase 2: Main Loop - Alternate Nelder-Mead and Powell
        n = self.dim
        
        # NM State
        simplex = np.zeros((n + 1, n))
        simplex[0] = x
        for i in range(n):
            simplex[i + 1] = x.copy()
            step_init = 0.5
            simplex[i + 1][i] += step_init if x[i] < 4.5 else -step_init
            simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
        
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            if calls_made >= self.budget or time.time() > deadline:
                break
            fvals[i] = timed_call(simplex[i])

        # Powell State: search directions
        directions = np.eye(n)
        no_improvement_count = 0
        max_no_improvement = 20
        
        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        use_nm = True
        nm_iterations = 0
        powell_iterations = 0

        while calls_made < self.budget and time.time() < deadline - 2:
            
            if use_nm:
                # --- Nelder-Mead Step ---
                if no_improvement_count > max_no_improvement:
                    use_nm = False
                    continue

                # Sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls_made >= self.budget or time.time() > deadline:
                    break
                fr = timed_call(xr)
                if fr is None:
                    break

                improved_nm = False
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    fe = timed_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    improved_nm = True
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    improved_nm = True
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    fc = timed_call(xc)
                    if fc is None:
                        break

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        improved_nm = True
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            fvals[i] = timed_call(simplex[i])
                            if fvals[i] is None:
                                break

                nm_iterations += 1
                if improved_nm:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Update best
                idx = np.argmin(fvals)
                if fvals[idx] < best_f:
                    best_f = fvals[idx]
                    best_x = simplex[idx].copy()
                x = simplex[idx].copy()
                fx = fvals[idx]

                # Switch to Powell every 8 NM iterations
                if nm_iterations >= 8:
                    use_nm = False
                    nm_iterations = 0
                    # Reset Powell directions to identity for fresh start
                    directions = np.eye(n)
                else:
                    use_nm = True

            else:
                # --- Powell's Conjugate Direction Step ---
                if no_improvement_count > max_no_improvement:
                    # Stall: reset simplex around current best
                    simplex[0] = best_x
                    for i in range(n):
                        simplex[i + 1] = best_x.copy()
                        simplex[i + 1][i] += 0.5 if best_x[i] < 4.5 else -0.5
                        simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
                    for i in range(n + 1):
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        fvals[i] = timed_call(simplex[i])
                        if fvals[i] is None:
                            break
                    no_improvement_count = 0
                    use_nm = True
                    nm_iterations = 0
                    continue

                # Powell's method: line search along each direction
                x_powell = x.copy()
                f_powell = fx
                min_dir = 0
                min_decrease = 0.0

                for d in range(n):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    # 1D line search along direction[d]
                    # Use golden section or simple bisection-like approach
                    # Try a few steps to find improvement
                    best_fd = f_powell
                    best_td = 0.0
                    t = 1.0
                    for _ in range(6):
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        x_test = x_powell + t * directions[d]
                        x_test = np.clip(x_test, -5, 5)
                        if np.allclose(x_test, x_powell):
                            break
                        ft = timed_call(x_test)
                        if ft is None:
                            break
                        if ft < best_fd:
                            best_fd = ft
                            best_td = t
                        else:
                            # If we got worse, halve the step
                            t *= 0.5
                            if t < 1e-6:
                                break
                            continue
                        t *= 1.2  # Try larger step if improved

                    if best_td != 0.0 and best_fd < f_powell:
                        decrease = f_powell - best_fd
                        x_powell = x_powell + best_td * directions[d]
                        x_powell = np.clip(x_powell, -5, 5)
                        f_powell = best_fd
                        if decrease > min_decrease:
                            min_decrease = decrease
                            min_dir = d

                # Update direction: Powell's conjugate direction update
                if min_dir < n - 1:
                    # Save the direction that gave max decrease
                    temp = directions[min_dir].copy()
                    for i in range(min_dir, n - 1):
                        directions[i] = directions[i + 1].copy()
                    directions[n - 1] = temp
                # The new direction is from old x to new x
                new_dir = x_powell - x
                norm_new = np.linalg.norm(new_dir)
                if norm_new > 1e-10:
                    directions[n - 1] = new_dir / norm_new

                # Check for convergence/stall
                if min_decrease > 1e-8:
                    no_improvement_count = 0
                    if f_powell < best_f:
                        best_f = f_powell
                        best_x = x_powell.copy()
                    x = x_powell.copy()
                    fx = f_powell
                else:
                    no_improvement_count += 1

                powell_iterations += 1
                # Switch back to NM after 3 Powell iterations
                if powell_iterations >= 3:
                    use_nm = True
                    powell_iterations = 0
                    nm_iterations = 0
                else:
                    use_nm = False

        # Phase 3: Final coordinate-descent polish if budget remains
        if calls_made < self.budget and time.time() < deadline - 5:
            x_polish = best_x.copy()
            f_polish = best_f
            step = 0.3
            for sweep in range(4):
                improved = False
                for d in range(n):
                    for direction in (1, -1):
                        for _ in range(15):
                            if calls_made >= self.budget or time.time() > deadline - 2:
                                break
                            x_new = x_polish.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            if np.allclose(x_new, x_polish):
                                break
                            fx_new = timed_call(x_new)
                            if fx_new is None:
                                break
                            if fx_new < f_polish:
                                x_polish = x_new
                                f_polish = fx_new
                                improved = True
                            else:
                                break
                        if calls_made >= self.budget or time.time() > deadline - 2:
                            break
                    if calls_made >= self.budget or time.time() > deadline - 2:
                        break
                step *= 0.6
                if not improved:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
