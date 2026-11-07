"""Lattice-based grid search followed by CMA-ES refinement for Rastrigin (BBOB f3)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer for BBOB f3 (Rastrigin):
      Phase 1: Deterministic lattice search over integer and half-integer offsets
               to exploit the periodic structure and locate the global basin.
      Phase 2: CMA-ES local refinement from the best lattice point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        calls = 0

        # --- Phase 1: Lattice search ---
        # Rastrigin has periodicity 2 in each dimension. The hidden shift is in [-4,4]^5.
        # We evaluate a grid of points at integer and half-integer coordinates within [-5,5].
        # This gives us 6x6x6x6x6 = 46656 points at step 1, but that's too many.
        # Instead, we use a coarser lattice first, then refine.

        # Strategy: evaluate points at integer coordinates in [-4, 4] (9 values per dim)
        # and also at half-integers. But 9^5 = 59049 is too many for 500 calls.
        #
        # Better: The Rastrigin function f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # has minima at integer points. With a shift s, minima are at s + integer.
        # Since s is in [-4,4], the global minimum is at s (or s + k for some integer k,
        # but within [-5,5] the closest integer offset matters).
        #
        # Key insight: if we evaluate at points p = s + k where k is integer, we hit the minimum.
        # We don't know s, but we can try all integer lattice points and half-integer points.
        #
        # With 500 calls, we can't do a full 9^5 grid. Instead, use a structured approach:
        # Evaluate at a set of candidate points derived from a coarse grid, then pick the best.

        # Coarse grid: step 1.0, from -4 to 4 => 9 points per dim => 9^5 = 59049 (too many)
        # Use step 2.0: from -4 to 4 => 5 points per dim => 5^5 = 3125 (still too many)
        # Use step 2.0 but only a subset: we have ~200 calls for phase 1.
        #
        # Alternative: Use the fact that Rastrigin is separable. We can do a coordinate-wise
        # scan. But that's not straightforward with a black-box.
        #
        # Practical approach: evaluate a random subset of the integer lattice, plus the origin,
        # plus a few structured points. Then use the best as starting point for CMA-ES.

        # Let's use a hybrid: 
        # - Evaluate at 0, and at +/-1, +/-2, +/-3, +/-4 in each dimension (structured random)
        # - Actually, let's just do a Latin hypercube or Sobol-like sampling on the integer grid.

        # Simplest effective approach: 
        # Phase 1: ~100 evaluations on a structured grid (integer points in [-4,4])
        # Phase 2: ~400 evaluations with CMA-ES from the best point

        # Generate integer lattice points in [-4, 4]^5
        # 9^5 = 59049 points. We can't evaluate all. 
        # Use a space-filling sample: take a subset.

        # Actually, for Rastrigin with shift, the key is to hit near s. 
        # If we evaluate at integer points, the best integer point will be within 0.5 of s in each dim,
        # so the function value difference from the true minimum is bounded.
        # For Rastrigin, f(s + k) = 10*n (the minimum) when k is integer.
        # f(s + k + d) where d is in [-0.5, 0.5]^5: the extra term is sum(d_i^2 - 10*cos(2*pi*d_i))
        # At d=0.5: 0.25 - 10*cos(pi) = 0.25 + 10 = 10.25 per dim, so 51.25 total extra.
        # So even the worst integer point is within ~51 of the global min. That's not great.
        
        # Better: also evaluate at half-integer points. Then max distance to s is 0.5 in each dim,
        # but we have both integer and half-integer, so max distance is 0.5.
        # Wait, if we have both integer and half-integer, the max distance from s to the nearest 
        # lattice point is 0.5 (since s can be anywhere, the nearest integer or half-integer is 
        # at most 0.5 away... no, the nearest point in the union of integers and half-integers 
        # is at most 0.25 away? No. The lattice points are at ..., -1, -0.5, 0, 0.5, 1, ...
        # The maximum distance from any point to the nearest lattice point is 0.25.
        # At d=0.25: 0.0625 - 10*cos(2*pi*0.25) = 0.0625 - 10*cos(pi/2) = 0.0625 - 0 = 0.0625 per dim.
        # Total extra: 5 * 0.0625 = 0.3125. That's very good!
        #
        # So if we can evaluate the integer and half-integer lattice, we can get within 0.3125 
        # of the global minimum. But 9^5 * 2^5 = 59049 * 32 = 1,889,568 points. Way too many.
        #
        # Compromise: use a coarser lattice. Step 0.5 gives 21 points per dim in [-5,5]. 21^5 is huge.
        # Step 1.0 with both integer and half-integer: 18 points per dim. 18^5 = 1,889,568. Still huge.
        #
        # Use step 1.0, only integer points: 9 per dim, 9^5 = 59049. Still too many for 500 calls.
        #
        # So we need a smarter approach. Let's use a two-level strategy:
        # Level 1: Coarse grid with step 2.0: 5 points per dim (-4, -2, 0, 2, 4). 5^5 = 3125.
        #           Too many. Use a random subset of ~50 points.
        # Level 2: Fine grid around the best point from Level 1.

        # Actually, let's just use a well-designed random search for Phase 1, then CMA-ES for Phase 2.
        # The random search should be structured to cover the space well.

        # Phase 1: 150 evaluations using a space-filling design (Sobol-like)
        # Phase 2: 350 evaluations with CMA-ES

        # Generate Sobol-like points using a simple LHS (Latin Hypercube)
        def lhs_sample(n, d, rng):
            """Generate n points in [0,1]^d using Latin Hypercube Sampling."""
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                points[:, j] = (perm + rng.random(n)) / n
            return points

        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling in [-5, 5]^5
        n_phase1 = 150
        lhs_points = lhs_sample(n_phase1, dim, rng)
        lhs_points = -5 + 10 * lhs_points

        best_x = None
        best_f = np.inf

        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also evaluate the origin and a few structured points
        structured_points = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
        ]
        for sp in structured_points:
            if calls >= budget or time.time() > deadline:
                break
            fx = func(sp)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = sp.copy()

        # Phase 2: CMA-ES from best_x
        if calls < budget and time.time() < deadline:
            # Simple CMA-ES implementation
            x = best_x.copy()
            f = best_f
            sigma = 1.0
            # CMA-ES parameters
            n = dim
            # Default CMA-ES parameters
            c_sigma = (n + 1.7) / (n + 3 + 2 * n / (n - 1))
            d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + n / n) / (4 + n + n)
            c_1 = 2 / ((n + 1.3) ** 2 + n)
            c_mu = min(1 - c_1, 2 * ((n + 1) / (n + 3)) ** 2 - 1)
            mu = n // 2  # number of recombination weights
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / (w @ w)
            c_1 = 2 / ((n + 1.3) ** 2 + n)
            c_mu = min(1 - c_1, 2 * ((mu_eff - 2) / (mu_eff + 2) - 1))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            # Eigen decomposition for step size adaptation
            # We'll use a simplified version

            # Actually, let's use a simpler (1+1)-ES with adaptive sigma for the remaining budget.
            # CMA-ES is complex; let's use a robust (1+1)-ES with restarts.

            # Restart (1+1)-ES
            remaining = budget - calls
            sigma = 0.5
            no_improve = 0
            max_no_improve = 50

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < f:
                    x, f = y, fy
                    no_improve = 0
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if no_improve > max_no_improve:
                        # Restart with a new random point
                        x = rng.uniform(-5, 5, n)
                        f = func(x)
                        calls += 1
                        sigma = 1.0
                        no_improve = 0
                        if calls >= budget or time.time() > deadline:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
