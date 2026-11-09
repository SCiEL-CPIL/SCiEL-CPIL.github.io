"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: coarse sampling + anisotropic 1D search + refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse random sampling to find basin
        n_sample = min(40, budget // 10)
        samples = rng.uniform(-5, 5, (n_sample, dim))
        for i in range(n_sample):
            if time.time() > deadline:
                break
            eval_func(samples[i])

        if calls >= budget or best_x is None:
            return best_x, best_f

        # Phase 2: Estimate the steep direction using local points
        # Sort samples by function value
        sample_vals = np.array([eval_func(s) for s in samples[:10]]) if calls < budget else np.array([])
        
        # Actually, let's be more careful with budget tracking
        # Let me redo this with better budget management
        
        # Reset and do it properly
        # We've used n_sample calls so far
        
        if calls >= budget:
            return best_x, best_f

        # Get the best few points to estimate direction
        # We need to collect values for the samples we already evaluated
        # But we didn't store them, so let's re-evaluate the best candidate neighborhood
        
        # Better approach: use the best point and do local exploration
        x0 = best_x.copy()
        f0 = best_f

        # Phase 2: Identify steep direction via local finite differences
        # Use a small step to probe each direction
        step_size = 0.5
        dir_effects = np.zeros(dim)
        probes = 0
        for d in range(dim):
            if time.time() > deadline or calls >= budget - 5:
                break
            for sign in [1, -1]:
                if time.time() > deadline or calls >= budget - 5:
                    break
                xp = x0.copy()
                xp[d] += sign * step_size
                xp = np.clip(xp, -5, 5)
                fp = eval_func(xp)
                dir_effects[d] += fp - f0
                probes += 1

        # The direction with the largest |effect| is the steep one
        # But we need to be careful: for Bent Cigar, one direction is very steep
        # Let's find the direction with maximum absolute change
        if probes > 0:
            steep_dir = np.argmax(np.abs(dir_effects))
        else:
            steep_dir = 0

        # Phase 3: 1D search along the steep direction
        # Fix other coordinates, search along steep_dir
        # Use golden section search
        def golden_search_1d(x_base, axis, f_base, n_iters=15):
            """Golden section search along one axis."""
            gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
            lo = -5.0
            hi = 5.0
            # Evaluate at initial points
            x1 = x_base.copy()
            x1[axis] = lo + (1 - gr) * (hi - lo)
            x2 = x_base.copy()
            x2[axis] = lo + gr * (hi - lo)
            
            f1 = f_base
            f2 = f_base
            # Actually we need to evaluate these
            if time.time() > deadline or calls >= budget:
                return x_base, f_base
            
            f1 = eval_func(x1)
            if time.time() > deadline or calls >= budget:
                return best_x, best_f
            f2 = eval_func(x2)
            
            for _ in range(n_iters):
                if time.time() > deadline or calls >= budget:
                    break
                if f1 < f2:
                    hi = x2[axis]
                    x2 = x1.copy()
                    f2 = f1
                    x1[axis] = lo + (1 - gr) * (hi - lo)
                    f1 = eval_func(x1)
                else:
                    lo = x1[axis]
                    x1 = x2.copy()
                    f1 = f2
                    x2[axis] = lo + gr * (hi - lo)
                    f2 = eval_func(x2)
            
            # Return the better point
            if f1 < f2:
                return x1, f1
            else:
                return x2, f2

        # Perform 1D search along the steep direction
        x_steep, f_steep = golden_search_1d(x0, steep_dir, f0, n_iters=12)
        if f_steep < best_f:
            best_x = x_steep.copy()
            best_f = f_steep

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Refinement using Nelder-Mead on the remaining dimensions
        # Or just do coordinate descent on all dimensions
        # Let's do a simple coordinate descent refinement
        x = best_x.copy()
        f = best_f
        step = 0.5
        for _ in range(50):
            if time.time() > deadline or calls >= budget:
                break
            improved = False
            for d in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                # Try moving in positive and negative direction
                for sign in [1, -1]:
                    if time.time() > deadline or calls >= budget:
                        break
                    xp = x.copy()
                    xp[d] += sign * step
                    xp = np.clip(xp, -5, 5)
                    fp = eval_func(xp)
                    if fp < f:
                        x = xp
                        f = fp
                        improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        if f < best_f:
            best_x = x.copy()
            best_f = f

        return best_x, best_f
# EVOLVE-BLOCK-END
