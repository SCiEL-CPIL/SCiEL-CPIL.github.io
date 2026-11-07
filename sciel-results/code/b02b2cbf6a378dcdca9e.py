"""New design: exploit Rastrigin periodicity + CMA-ES style global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer for rotated Rastrigin f15:
    1. Exploit the known 2*pi periodicity of Rastrigin in rotated coordinates
       to generate a lattice of candidate basin centers.
    2. Use a few well-chosen starting points (lattice + random) to find the
       global basin via local minimization.
    3. Refine with a CMA-ES-like adaptive step to converge to the local min.
    4. Time-guarded to stay within the 330 s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_used = 0
        self.deadline = None

    def __call__(self, func):
        self.calls_used = 0
        self.deadline = time.time() + 280.0  # 20 s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def f(x):
            self.calls_used += 1
            return func(x)

        # --- Phase 1: Generate candidate basin centers using periodicity ---
        # Rastrigin: f(z) = 10*dim + sum(z_i^2 - 10*cos(2*pi*z_i))
        # In the rotated frame, minima are at integer multiples of 2*pi in each
        # rotated coordinate, shifted by the hidden offset. The shift is in
        # [-4,4]^5, so the relevant lattice points in the original frame are
        # those whose rotated coordinates are near 2*pi*k for small integers k.
        # We don't know the rotation, so we generate a dense set of candidates
        # in the box and let the local search find the basin.

        # Strategy: multi-start local search with diverse starting points
        # 1) Grid points at the box corners and midpoints
        # 2) Random points
        # 3) Points based on the periodic structure: for each dimension,
        #    the Rastrigin has minima at 2*pi*k. In the original frame with
        #    shift s and rotation R, the minima are at R^T(2*pi*k) + s.
        #    Since we don't know R or s, we use a dense sampling.

        # Generate candidate starting points
        candidates = []

        # Center
        candidates.append(np.zeros(dim))

        # Axis-aligned grid: for each dim, sample at -4, -2, 0, 2, 4
        # This gives 5^5 = 3125 points, too many. Use a coarser set.
        # Use a Latin hypercube-like spread + random
        n_random = 200
        candidates.extend(rng.uniform(-4.5, 4.5, size=(n_random, dim)).tolist())

        # Also add points near the periodic minima in each coordinate
        # For Rastrigin, minima are at 2*pi*k ≈ 6.28*k. Since the box is [-5,5],
        # only k=0 is fully inside. But with rotation, the effective period
        # projects differently. Add points at multiples of ~3.14 (pi) in each dim.
        for k in [-1, 0, 1]:
            for axis in range(dim):
                pt = np.zeros(dim)
                pt[axis] = k * np.pi
                candidates.append(pt)

        candidates = np.array(candidates)

        # --- Phase 2: Evaluate all candidates, keep the best few ---
        best_x = None
        best_f = np.inf
        second_best_x = None
        second_best_f = np.inf

        for c in candidates:
            if self.calls_used >= budget or time.time() > self.deadline:
                break
            c = np.clip(c, -5, 5)
            fc = f(c)
            if fc < best_f:
                second_best_x, second_best_f = best_x, best_f
                best_x, best_f = c.copy(), fc
            elif fc < second_best_f:
                second_best_x, second_best_f = c.copy(), fc

        # --- Phase 3: Local refinement from the best starting points ---
        def local_refine(x0, max_iter=80):
            """Adaptive step-size local search (CMA-ES lite)."""
            x = x0.copy()
            fx = f(x)
            sigma = 1.5
            for it in range(max_iter):
                if self.calls_used >= budget or time.time() > self.deadline:
                    break
                # Generate a few perturbations
                n_pert = 3
                for _ in range(n_pert):
                    if self.calls_used >= budget or time.time() > self.deadline:
                        break
                    step = sigma * rng.standard_normal(dim)
                    y = np.clip(x + step, -5, 5)
                    fy = f(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.3
                    else:
                        sigma *= 0.85
            return x, fx

        # Refine the best candidate
        x_best, f_best = local_refine(best_x)

        # Refine the second best candidate
        if second_best_x is not None and self.calls_used < budget and time.time() < self.deadline:
            x_2nd, f_2nd = local_refine(second_best_x)
            if f_2nd < f_best:
                x_best, f_best = x_2nd, f_2nd

        # --- Phase 4: If budget remains, do a finer local search around the best ---
        if self.calls_used < budget and time.time() < self.deadline:
            # Tighten with smaller sigma
            x_tight, f_tight = local_refine(x_best, max_iter=100)
            if f_tight < f_best:
                x_best, f_best = x_tight, f_tight

        return x_best, f_best
# EVOLVE-BLOCK-END
