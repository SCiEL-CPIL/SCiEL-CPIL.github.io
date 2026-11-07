"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and time guard for Sphere optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                if x.shape != (self.dim,):
                    x = x[:self.dim]
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                calls_made += 1
                return f
            except Exception:
                calls_made += 1
                return None
        
        # Phase 1: Initial random sampling to find a good starting region
        n_initial = min(20, self.budget // 10)
        candidates = []
        for _ in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = safe_call(x)
            if f is not None:
                candidates.append((f, x))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
        
        if not candidates:
            # Fallback: single random point
            x = rng.uniform(-5, 5, self.dim)
            f = safe_call(x)
            if f is not None:
                best_f = f
                best_x = x.copy()
            return best_x, best_f
        
        # Sort by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Nelder-Mead from the best candidate
        def nelder_mead(func_eval, x0, max_iter, initial_step):
            """Nelder-Mead simplex optimization."""
            n = len(x0)
            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += initial_step
                simplex.append(xi)
            
            # Evaluate initial simplex
            simplexF = []
            for i in range(n + 1):
                f = func_eval(simplex[i])
                if f is None:
                    f = 1e10
                simplexF.append(f)
            
            # Sort simplex by fitness
            def sort_simplex():
                nonlocal simplex, simplexF
                idx = np.argsort(simplexF)
                simplex = [simplex[i] for i in idx]
                simplexF = [simplexF[i] for i in idx]
            
            sort_simplex()
            
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Check for convergence
                if len(simplexF) >= 2:
                    if abs(simplexF[0] - simplexF[-1]) < 1e-12 * (1 + abs(simplexF[0])):
                        break
                
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = func_eval(x_r)
                if f_r is None:
                    f_r = 1e10
                
                if f_r < simplexF[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = func_eval(x_e)
                    if f_e is None:
                        f_e = 1e10
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplexF[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplexF[-1] = f_r
                elif f_r < simplexF[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    simplexF[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplexF[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = func_eval(x_c)
                    if f_c is None:
                        f_c = 1e10
                    if f_c < min(f_r, simplexF[-1]):
                        simplex[-1] = x_c
                        simplexF[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_s = func_eval(simplex[i])
                            if f_s is None:
                                f_s = 1e10
                            simplexF[i] = f_s
                
                sort_simplex()
            
            return simplex[0], simplexF[0]
        
        # Try Nelder-Mead from the best initial point
        best_initial = candidates[0]
        nm_x, nm_f = nelder_mead(safe_call, best_initial[1].copy(), max_iter=200, initial_step=1.0)
        if nm_f < best_f:
            best_f = nm_f
            best_x = nm_x.copy()
        
        # Phase 3: Try additional restarts from other good initial points if budget remains
        for f_init, x_init in candidates[1:min(5, len(candidates))]:
            if time.time() > deadline or calls_made >= self.budget - 10:
                break
            # Only restart if this initial point is reasonably good
            if f_init > best_f * 1.1 + 10:
                continue
            nm_x2, nm_f2 = nelder_mead(safe_call, x_init.copy(), max_iter=50, initial_step=0.5)
            if nm_f2 < best_f:
                best_f = nm_f2
                best_x = nm_x2.copy()
        
        # Final local refinement with small steps if budget remains
        if time.time() < deadline and calls_made < self.budget - 5 and best_x is not None:
            sigma = 0.1
            for _ in range(min(20, self.budget - calls_made)):
                if time.time() > deadline:
                    break
                x_trial = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                f_trial = safe_call(x_trial)
                if f_trial is None:
                    break
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8
        
        return best_x, best_f
# EVOLVE-BLOCK-END
