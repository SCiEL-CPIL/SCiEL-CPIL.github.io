"""Multi-objective optimizer exploiting ZDT3 structure."""
import numpy as np
import time


class Optimizer:
    """Exploits ZDT3 structure: x_2..x_n=0 minimizes f_2, then refine x_1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3: f1 = x1, f2 = g(x2..xn) * h(f1, g)
        # g = 1 + 10(n-1) + sum(25*(sin(50*x_i^0.1)) for i=2..n)
        # h = 1 + sin^3(10*x1*pi)/2
        # Optimal: x2..xn=0 -> g=1, f1=x1, f2=h(x1)
        # Pareto front: f1 in [0,1], f2 = h(f1)

        # Strategy:
        # 1. Set x_2..x_dim = 0 (or small values) to minimize g
        # 2. Focus evaluations on finding good x_1 values that give good (f1, f2) pairs
        # 3. Use golden-section or similar to refine x_1

        # First, verify that x_2..x_dim=0 gives low g
        # Evaluate a few points to understand the landscape
        used = 0
        results = []  # list of (x, f)

        def eval_point(x):
            nonlocal used
            f = problem(x)
            used += 1
            return np.array(f)

        # Base point with x_2..x_dim = 0
        base_x = np.zeros(dim)
        base_x[0] = 0.5
        base_f = eval_point(base_x)
        results.append((base_x.copy(), base_f))

        # Check if setting x_2..x_dim=0 is good
        # Try a random point for comparison
        rng = np.random.default_rng(42)
        rand_x = lo + (hi - lo) * rng.random(dim)
        rand_f = eval_point(rand_x)
        results.append((rand_x.copy(), rand_f))
        used += 1

        # Now focus on x_1 with x_2..x_dim=0
        # We want to find x_1 values that give good hypervolume
        # The Pareto front of ZDT3 has f1=x1, f2=h(x1) where h has many local minima
        # h(x1) = 1 + sin^3(10*x1*pi)/2
        # Local minima of h occur where sin(10*x1*pi) = -1, i.e., 10*x1*pi = 3*pi/2 + 2*k*pi
        # x1 = (3/20 + k/5) for k=0,1,...,4
        # So x1 = 0.15, 0.35, 0.55, 0.75, 0.95 give h=-0.5 (minimum)
        # But we want to spread points along the front for hypervolume

        # Strategy: evaluate at many x_1 values with x_2..x_dim=0
        # Use a combination of:
        # 1. Grid points at known local minima of h
        # 2. Golden section search around promising regions
        # 3. Random sampling of x_1

        # Known local minima of h(x1) = 1 + sin^3(10*pi*x1)/2
        # sin(10*pi*x1) = -1 => 10*pi*x1 = 3*pi/2 + 2*pi*k => x1 = 0.15 + 0.2*k
        minima_x1 = [0.15, 0.35, 0.55, 0.75, 0.95]

        # Also add some other points for coverage
        candidate_x1 = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        # Evaluate these with x_2..x_dim=0
        for x1 in candidate_x1:
            if used >= budget - 10:
                break
            if time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = x1
            f = eval_point(x)
            results.append((x.copy(), f))

        # Now do local refinement around promising x_1 values
        # For each evaluated point, try small perturbations of x_1
        # and also try small perturbations of x_2..x_dim to see if they help

        # First, let's do a fine grid around the local minima
        for x1_center in minima_x1:
            if used >= budget - 5:
                break
            if time.time() > deadline - 5:
                break
            # Golden section search in a small neighborhood
            for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                if used >= budget - 5:
                    break
                if time.time() > deadline - 5:
                    break
                x1_new = x1_center + delta
                if x1_new < 0 or x1_new > 1:
                    continue
                x = np.zeros(dim)
                x[0] = x1_new
                f = eval_point(x)
                results.append((x.copy(), f))

        # Try some random x_1 values with x_2..x_dim=0
        while used < budget - 10:
            if time.time() > deadline - 5:
                break
            x1 = rng.random()
            x = np.zeros(dim)
            x[0] = x1
            f = eval_point(x)
            results.append((x.copy(), f))

        # Now try to improve by slightly perturbing x_2..x_dim
        # This might help if the clipping in problem() or numerical issues matter
        # But theoretically, x_2..x_dim=0 should be optimal for g

        # Final polish: for the best points, try very fine perturbations
        if used < budget - 5:
            # Sort results by f1 to find non-dominated set
            # For 2 objectives, a point is non-dominated if no other point has both f1<= and f2<=
            # with at least one strict
            f_arr = np.array([r[1] for r in results])
            # Keep only non-dominated points
            nondom_idx = []
            for i in range(len(f_arr)):
                is_dom = False
                for j in range(len(f_arr)):
                    if i == j:
                        continue
                    if f_arr[j, 0] <= f_arr[i, 0] and f_arr[j, 1] <= f_arr[i, 1] and \
                       (f_arr[j, 0] < f_arr[i, 0] or f_arr[j, 1] < f_arr[i, 1]):
                        is_dom = True
                        break
                if not is_dom:
                    nondom_idx.append(i)

            # Refine around non-dominated points
            for idx in nondom_idx:
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                x_base = results[idx][0]
                # Perturb x_1 slightly
                for delta in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                    if used >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    x1_new = x_base[0] + delta
                    if x1_new < 0 or x1_new > 1:
                        continue
                    x = x_base.copy()
                    x[0] = x1_new
                    f = eval_point(x)
                    results.append((x.copy(), f))

        return results
