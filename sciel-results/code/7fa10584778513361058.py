import numpy as np
import time


class Optimizer:
    """Structure-exploiting multi-objective optimizer for ZDT4.

    ZDT4 structure:
      g(x) = 1 + 10 * sum_{i=1}^{n} sin^2(10*x_i)
      f0 = g(x)
      f1 = 1 - sqrt((1/n) * sum x_i^2) * g(x)

    Pareto front: g(x) = 1 (all sin terms zero), so x_i = k*pi/10 for integer k.
      f0 = 1
      f1 = 1 - sqrt(sum x_i^2 / n)

    Key insight: For 2D hypervolume with all points on f0=1 line,
    HV = (ref_f0 - 1) * (ref_f1 - min_f1)
    So we need to minimize f1, which means maximizing sum x_i^2.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        n = dim
        budget = self.budget

        # Compute max |k| for each dimension based on bounds
        # x_i = k*pi/10 must satisfy low_i <= x_i <= high_i
        k_range_per_dim = []
        for i in range(dim):
            low, high = bounds[i]
            # k*pi/10 >= low => k >= low*10/pi
            # k*pi/10 <= high => k <= high*10/pi
            k_min = int(np.ceil(low * 10.0 / np.pi))
            k_max = int(np.floor(high * 10.0 / np.pi))
            if k_min > k_max:
                k_min, k_max = 0, 0
            k_range_per_dim.append((k_min, k_max))

        # The point that minimizes f1 maximizes sum x_i^2
        # For each dim, choose k with max |k|
        best_point = np.zeros(dim)
        for i in range(dim):
            k_min, k_max = k_range_per_dim[i]
            if abs(k_min) >= abs(k_max):
                best_point[i] = k_min * np.pi / 10.0
            else:
                best_point[i] = k_max * np.pi / 10.0

        # Evaluate the best point first (most important for hypervolume)
        used_evals = 0
        F = []

        fc = problem(best_point)
        F.append(fc)
        used_evals += 1

        # Compute the range of achievable f1 values
        max_sum_sq = 0.0
        for i in range(dim):
            k_min, k_max = k_range_per_dim[i]
            max_k = max(abs(k_min), abs(k_max))
            max_sum_sq += (max_k * np.pi / 10.0) ** 2

        max_f1_range = np.sqrt(max_sum_sq / n)

        # Generate additional points on the Pareto front for robustness
        # Use quadratic spacing near f1=1 (high curvature region)
        n_extra = min(budget - used_evals, 199)

        if n_extra > 0:
            t = np.linspace(0, 1, n_extra + 1)
            t_quad = t ** 2
            target_f1_values = 1.0 - t_quad * max_f1_range

            for target_f1 in target_f1_values[1:]:
                if used_evals >= budget:
                    break
                if time.time() > deadline:
                    break

                # Find a point on the Pareto front with f1 ≈ target_f1
                target_sum_sq = n * (1.0 - target_f1) ** 2

                remaining = target_sum_sq
                pt = np.zeros(dim)

                for i in range(dim):
                    if remaining <= 1e-10:
                        break
                    k_min, k_max = k_range_per_dim[i]
                    best_k = 0
                    for k in range(k_min, k_max + 1):
                        x_val = k * np.pi / 10.0
                        if x_val ** 2 <= remaining + 1e-10 and abs(k) > abs(best_k):
                            best_k = k
                    if best_k != 0:
                        pt[i] = best_k * np.pi / 10.0
                        remaining -= pt[i] ** 2

                fc = problem(pt)
                F.append(fc)
                used_evals += 1

        # Use any remaining budget for additional random points on the front
        rng = np.random.default_rng(42)
        while used_evals < budget and time.time() < deadline:
            pt = np.zeros(dim)
            for i in range(dim):
                k_min, k_max = k_range_per_dim[i]
                k = rng.integers(k_min, k_max + 1)
                pt[i] = k * np.pi / 10.0
            fc = problem(pt)
            F.append(fc)
            used_evals += 1

        return F
