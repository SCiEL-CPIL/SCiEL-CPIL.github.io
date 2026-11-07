"""Optimized search for BBOB f15 (rotated Rastrigin) using lattice-based candidate generation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based optimizer exploiting the 2*pi periodicity of Rastrigin.
    Generates candidate minima on the periodic lattice and evaluates top-k closest
    to initial best estimate, bypassing continuous local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count < budget

        # --- Phase 1: Initial broad sampling to estimate rotation and shift ---
        # Sample a few random points to get an initial sense of the landscape
        n_init = min(10, budget)
        init_samples = []
        for _ in range(n_init):
            if not budget_ok() or not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            init_samples.append((x, fx))
        
        if not init_samples:
            return best_x, best_f

        # Sort by function value
        init_samples.sort(key=lambda t: t[1])
        best_init_x = init_samples[0][0].copy()
        best_init_f = init_samples[0][1]

        # --- Phase 2: Lattice-based candidate generation ---
        # The Rastrigin function has minima at x_i = 2*pi*k_i in the unrotated frame.
        # After rotation R and shift s, minima are at R @ (2*pi * k) + s for integer k.
        # We estimate R and s from the best initial sample, then generate candidate minima.
        
        # Estimate shift: assume best_init_x is near a minimum
        # Estimate rotation: use the periodic structure
        
        # For f15 (rotated Rastrigin), the function is:
        # f(x) = sum_i [10*(dim) + (x_i - x*_i)^2 - 10*cos(2*pi*(x_i - x*_i))]
        # where x* is the global minimum location.
        
        # The key insight: minima occur at x = x* + 2*pi * R @ k for integer vectors k,
        # where R is the rotation matrix.
        
        # We'll use a grid search approach: generate candidate points on the lattice
        # and evaluate the top-k closest to our best estimate.
        
        # First, let's do a coarse grid to estimate the rotation
        # Try different rotation angles and see which gives best results
        
        # Actually, let's use a simpler approach: 
        # 1. Do local refinement around the best initial point
        # 2. Use the periodicity to jump to other potential minima
        
        # Local refinement with periodic awareness
        def local_refine_periodic(x, fx, max_rounds=3):
            """Local refinement using periodic structure."""
            nonlocal call_count, best_x, best_f
            if not budget_ok() or not time_ok():
                return x, fx
            
            x_local = x.copy()
            f_local = fx
            
            for _ in range(max_rounds):
                if not budget_ok() or not time_ok():
                    break
                improved = False
                for axis in range(dim):
                    if not budget_ok() or not time_ok():
                        break
                    x0 = x_local[axis]
                    
                    # Try current value and nearby periodic candidates
                    candidates = []
                    
                    # Small perturbations
                    for d in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]:
                        candidates.append(x0 + d)
                    
                    # Periodic candidates: try snapping to nearby 2*pi multiples
                    # relative to the best estimate
                    k_base = np.round(x0 / (2.0 * np.pi))
                    for kk in [k_base - 2, k_base - 1, k_base, k_base + 1, k_base + 2]:
                        candidates.append(kk * 2.0 * np.pi)
                    
                    # Try halfway between minima (to avoid maxima)
                    for kk in [k_base - 1, k_base, k_base + 1]:
                        candidates.append(kk * 2.0 * np.pi + np.pi)
                    
                    # Evaluate candidates
                    best_cand = x0
                    best_cand_f = f_local
                    for c in candidates:
                        if not budget_ok():
                            break
                        c = float(np.clip(c, -5.0, 5.0))
                        if abs(c - x0) < 1e-10:
                            continue
                        y = x_local.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    
                    if best_cand != x0:
                        x_local[axis] = best_cand
                        f_local = best_cand_f
                        improved = True
                
                if not improved:
                    break
            
            return x_local, f_local

        # Refine the best initial point
        x_ref, f_ref = local_refine_periodic(best_init_x, best_init_f, max_rounds=5)
        
        # --- Phase 3: Lattice jump to find other minima ---
        # Use the refined point to estimate the lattice structure
        # Then jump to nearby lattice points
        
        # Estimate the "phase" of each dimension relative to the minimum
        # For each dimension, the minimum is near x_ref[i]
        # Other minima should be at x_ref[i] +/- 2*pi*k
        
        lattice_candidates = []
        
        # Generate candidates by shifting each dimension by 2*pi multiples
        # from the refined best point
        for axis in range(dim):
            if not budget_ok() or not time_ok():
                break
            x_base = x_ref.copy()
            x_base_f = f_ref
            
            # Try shifting this axis by 2*pi
            for shift in [-2*np.pi, 2*np.pi]:
                if not budget_ok() or not time_ok():
                    break
                x_test = x_base.copy()
                x_test[axis] = x_test[axis] + shift
                x_test = np.clip(x_test, -5, 5)
                
                # Check if this is within bounds and different
                if np.all(np.abs(x_test - x_base) > 1e-6):
                    fx_test = eval_func(x_test)
                    if fx_test < best_f:
                        lattice_candidates.append((x_test.copy(), fx_test))
        
        # Also try combinations of shifts
        for axis1 in range(dim):
            if not budget_ok() or not time_ok():
                break
            for axis2 in range(axis1 + 1, dim):
                if not budget_ok() or not time_ok():
                    break
                for shift1 in [-2*np.pi, 2*np.pi]:
                    if not budget_ok() or not time_ok():
                        break
                    for shift2 in [-2*np.pi, 2*np.pi]:
                        if not budget_ok() or not time_ok():
                            break
                        x_test = x_ref.copy()
                        x_test[axis1] = x_test[axis1] + shift1
                        x_test[axis2] = x_test[axis2] + shift2
                        x_test = np.clip(x_test, -5, 5)
                        
                        if np.all(np.abs(x_test - x_ref) > 1e-6):
                            fx_test = eval_func(x_test)
                            if fx_test < best_f:
                                lattice_candidates.append((x_test.copy(), fx_test))
        
        # Sort lattice candidates by function value
        lattice_candidates.sort(key=lambda t: t[1])
        
        # Refine the best lattice candidate
        if lattice_candidates and budget_ok() and time_ok():
            x_latt, f_latt = lattice_candidates[0]
            x_latt_ref, f_latt_ref = local_refine_periodic(x_latt, f_latt, max_rounds=3)
        
        # --- Phase 4: Final intensive refinement ---
        # Use remaining budget for intensive local search
        if best_x is not None and budget_ok() and time_ok():
            x_final = best_x.copy()
            f_final = best_f
            
            # Intensive local refinement
            for _ in range(10):
                if not budget_ok() or not time_ok():
                    break
                x_ref2, f_ref2 = local_refine_periodic(x_final, f_final, max_rounds=2)
                if f_ref2 < f_final:
                    f_final = f_ref2
                    x_final = x_ref2.copy()
                else:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
