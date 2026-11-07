"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """
    Hybrid optimizer for BBOB f4 (Büche-Rastrigin).
    Strategy:
    1. Sobol Sequence for dense global exploration (first 80 calls).
    2. Multi-Start (1+1)-ES with adaptive step size and stagnation-based restarts.
       This leverages the structural idea of Multi-Start ES which outperforms 
       single-trajectory approaches on multimodal landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        t0 = time.time()
        deadline = t0 + 285.0

        calls = 0
        best_x = None
        best_f = float('inf')

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def check_deadline():
            nonlocal calls
            return calls >= budget or time.time() > deadline

        # --- Phase 1: Coarse exploration with Sobol-like sampling ---
        # Büche-Rastrigin is multimodal. Good initial coverage is key.
        n_phase1 = 80
        
        # Try to use Sobol for low-discrepancy sampling
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            # Generate power of 2 >= n_phase1
            next_pow2 = 1
            while next_pow2 < n_phase1:
                next_pow2 *= 2
            points = sampler.random(next_pow2)[:n_phase1]
        except Exception:
            # Fallback to random if scipy is not available or fails
            points = rng.random((n_phase1, d))
        
        # Map from [0, 1]^d to [-5, 5]^d
        points = points * 10.0 - 5.0

        # Evaluate phase 1 points
        candidates = []
        for i in range(n_phase1):
            if check_deadline():
                break
            x = clip(points[i])
            fx = float(func(x))
            calls += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x0 = np.zeros(d)
            best_f = float(func(x0))
            best_x = x0.copy()
            calls += 1
            candidates.append((best_f, best_x.copy()))

        # Sort candidates by fitness to identify promising basins
        candidates.sort(key=lambda item: item[0])

        # --- Phase 2: Multi-restart (1+1)-ES ---
        # We use a loop that runs until budget or time is exhausted.
        # It starts from the current best (or a restart point) and refines.
        # If it stagnates, it restarts from a different promising point or random.
        
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # Parameters
        sigma_min = 1e-5
        sigma_max = 5.0
        
        # Index into sorted candidates for restarts
        restart_idx = 0
        max_restarts = 50 # Safety cap on logical restarts, though budget limits it more
        
        # We will iterate through the sorted candidates as potential restart seeds
        # but primarily we rely on the "stagnation" logic to trigger new starts.
        
        # State for the current trajectory
        # Start with the global best found in Phase 1
        x_cur = best_x.copy()
        f_cur = best_f
        sigma = 1.5 # Initial step size
        
        # To allow "multi-start" effectively, we track which candidates we've used as seeds
        # or we just let the stagnation logic pull from the list of good candidates.
        
        # A simpler and effective strategy:
        # Run (1+1) ES from the global best. 
        # If it stagnates, pick the next best candidate from Phase 1 that hasn't been "explored" deeply,
        # or a random point if we run out of good candidates.
        
        # Let's use a list of "seeds" to try.
        seeds = [c[1] for c in candidates]
        seed_ptr = 0
        
        # We will loop until budget/time.
        # Each "epoch" is a (1+1) ES run from a seed.
        
        while not check_deadline():
            # Determine start point for this epoch
            if seed_ptr < len(seeds):
                start_x = seeds[seed_ptr].copy()
                # We need to evaluate this point if we haven't already? 
                # Actually, we have evaluated them in Phase 1.
                # But if we restart from a seed, we assume we know its f.
                # However, if we improved on a seed in a previous epoch, we should use the improved value.
                # For simplicity, we just use the Phase 1 value as the starting f_cur.
                # The ES will improve it.
                start_f = candidates[seed_ptr][0]
                seed_ptr += 1
            else:
                # If we ran out of Phase 1 seeds, do random restarts or perturb global best
                if rng.random() < 0.5:
                    start_x = rng.uniform(-5.0, 5.0, d)
                    start_f = float(func(start_x))
                    calls += 1
                    if check_deadline():
                        break
                    if start_f < best_f:
                        best_f = start_f
                        best_x = start_x.copy()
                else:
                    start_x = clip(best_x + rng.standard_normal(d) * 2.0)
                    start_f = float(func(start_x))
                    calls += 1
                    if check_deadline():
                        break
                    if start_f < best_f:
                        best_f = start_f
                        best_x = start_x.copy()

            # Initialize ES state
            x = start_x.copy()
            fx = start_f
            sigma = 1.5
            no_improve_count = 0
            max_no_improve = 25 # Stagnation threshold

            # Run (1+1) ES for this epoch
            while not check_deadline():
                # Generate candidate
                y = clip(x + sigma * rng.standard_normal(d))
                fy = float(func(y))
                calls += 1

                if fy < fx:
                    # Success
                    x = y.copy()
                    fx = fy
                    sigma *= 1.1 # Increase step size on success
                    no_improve_count = 0

                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure
                    sigma *= 0.9 # Decrease step size on failure
                    no_improve_count += 1

                # Clamp sigma
                sigma = np.clip(sigma, sigma_min, sigma_max)

                # Stagnation check
                if no_improve_count >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
