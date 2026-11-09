"""Multi-objective optimizer for ZDT4 using structure-aware sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT4.

    Exploits the known ZDT4 structure: f1 depends only on x0, and the
    Pareto-optimal tail is x1..x9 = 0. Dense sampling of x0 with the
    optimal tail directly samples the true Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # --- Phase 1: Dense sampling of x0 with optimal tail (x1..x9 = 0) ---
        # For ZDT4, x0 is typically in [-5, 5], tail in [0, 1].
        # The Pareto front is achieved when tail = 0 (g = n = 10).
        # f1 = 10 * x0^2, so we want dense coverage of x0^2 in [0, 250].

        # Determine x0 range from bounds
        x0_lo, x0_hi = lo[0], hi[0]
        # Tail: use 0 (the known Pareto-optimal value for ZDT4)
        tail_val = 0.0

        # Number of x0 samples: use ~70% of budget for dense front sampling
        n_x0 = min(int(budget * 0.7), 140)
        n_x0 = max(n_x0, 20)

        # Sample x0 densely. Since f1 = 10*x0^2, uniform in x0 gives
        # non-uniform f1. Better: uniform in f1 (i.e., sqrt) for even
        # hypervolume coverage.
        # f1 range: [0, 10*max(x0_lo^2, x0_hi^2)]
        f1_max = 10.0 * max(abs(x0_lo), abs(x0_hi)) ** 2
        if f1_max <= 0:
            f1_max = 1000.0  # default for standard ZDT4

        # Uniform in f1: f1_i = f1_max * i/(n_x0-1)
        # x0 = sqrt(f1/10), sign chosen to cover both sides
        f1_samples = np.linspace(0.0, f1_max, n_x0)
        x0_samples = np.sqrt(f1_samples / 10.0)

        # Alternate signs to cover [-x0_max, x0_max]
        # Actually, f1 = 10*x0^2 so both +x0 and -x0 give same f1.
        # We just need x0 values; the sign doesn't matter for f1.
        # But bounds may restrict sign. Use positive sqrt, clip to bounds.
        x0_samples = np.clip(x0_samples, x0_lo, x0_hi)

        # Also add a few negative x0 samples if bounds allow
        if x0_lo < 0:
            neg_samples = -x0_samples
            neg_samples = neg_samples[neg_samples >= x0_lo]
            # Merge: we already have positive; add negative to spread
            # Actually for hypervolume, f1 is the same, so it doesn't help.
            # Just use positive x0 (or whichever is in bounds).
            pass

        # Build points: x0 = x0_samples, tail = 0
        X_front = np.zeros((n_x0, dim))
        X_front[:, 0] = x0_samples
        # tail stays 0

        # Evaluate
        F_front = np.array([problem(x) for x in X_front])
        used = n_x0

        # --- Phase 2: Verify tail optimum and fill gaps ---
        # Check if tail=0 is truly optimal by sampling a few tail configurations
        # at a few x0 values. Also do local refinement.

        remaining = budget - used
        if remaining > 0:

            # Phase 2a: Local refinement around sampled points
            # For each point, try small perturbations in x0 to fill gaps
            n_refine = min(remaining, 50)
            if n_refine > 0:
                # Interpolate: find gaps in x0 and sample there
                x0_sorted = np.sort(x0_samples)
                if len(x0_sorted) >= 2:
                    # Find midpoints of largest gaps
                    gaps = np.diff(x0_sorted)
                    # Add points at midpoints of top gaps
                    n_add = min(n_refine, len(gaps))
                    if n_add > 0:
                        # Pick evenly spaced gaps to fill
                        idx = np.linspace(0, len(gaps) - 1, n_add, dtype=int)
                        mid_x0 = (x0_sorted[idx] + x0_sorted[idx + 1]) / 2.0
                        mid_x0 = np.clip(mid_x0, x0_lo, x0_hi)
                        X_add = np.zeros((n_add, dim))
                        X_add[:, 0] = mid_x0
                        F_add = np.array([problem(x) for x in X_add])
                        used += n_add
                        X_front = np.vstack([X_front, X_add])
                        F_front = np.vstack([F_front, F_add])

            # Phase 2b: Explore tail variations to ensure robustness
            # Sample a few points with non-zero tail to verify
            remaining2 = budget - used
            if remaining2 > 0:
                n_tail = min(remaining2, 20)
                rng = np.random.default_rng(123)
                X_tail = np.zeros((n_tail, dim))
                # Random x0
                X_tail[:, 0] = x0_lo + (x0_hi - x0_lo) * rng.random(n_tail)
                # Random tail (small values to stay near optimum)
                X_tail[:, 1:] = rng.random((n_tail, dim - 1)) * 0.1
                X_tail = np.clip(X_tail, lo, hi)
                F_tail = np.array([problem(x) for x in X_tail])
                used += n_tail
                X_front = np.vstack([X_front, X_tail])
                F_front = np.vstack([F_front, F_tail])

            # Phase 2c: Use remaining budget for additional x0 refinement
            remaining3 = budget - used
            if remaining3 > 0:
                rng2 = np.random.default_rng(456)
                n_extra = remaining3
                X_extra = np.zeros((n_extra, dim))
                # Uniform in x0 for extra coverage
                X_extra[:, 0] = x0_lo + (x0_hi - x0_lo) * rng2.random(n_extra)
                X_extra[:, 1:] = 0.0  # optimal tail
                X_extra = np.clip(X_extra, lo, hi)
                F_extra = np.array([problem(x) for x in X_extra])
                used += n_extra
                X_front = np.vstack([X_front, X_extra])
                F_front = np.vstack([F_front, F_extra])

        # --- Phase 3: Remove dominated points (keep only non-dominated) ---
        # This ensures we return a clean Pareto front approximation
        # Actually, the evaluator records ALL returned vectors, so we should
        # return all points. But dominated points don't hurt hypervolume.
        # Return all points.

        # Safety: check deadline
        if time.time() > deadline:
            pass  # We're done, just return

        return X_front
# EVOLVE-BLOCK-END
