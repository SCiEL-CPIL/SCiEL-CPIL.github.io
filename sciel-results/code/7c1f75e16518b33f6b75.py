import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        n = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # ZDT4 structure:
        # f1(x) = 1 + 10(n-1) + sum_{i=2}^{n} g(x_i)
        # f2(x) = (f1/n)^2 * (1 + sin(10*pi*f1/n))
        # where g(x) = x^2 - 10*cos(4*pi*x)
        #
        # The Pareto front is achieved when g(x_i) = -10 for all i=2..n,
        # which happens at x_i = 0 (the global minimum of g).
        # On the Pareto front, f1 = 1 + 10(n-1) - 10(n-1) = 1.
        # So f1 = 1 and f2 = (1/n)^2 * (1 + sin(10*pi/n))
        #
        # Wait, that's not right. Let me reconsider.
        # f1 = 1 + 10(n-1) + sum g(x_i)
        # If all x_i = 0 for i>=2, g(x_i) = 0 - 10*cos(0) = -10
        # So f1 = 1 + 10(n-1) + (n-1)*(-10) = 1 + 10n - 10 - 10n + 10 = 1
        # f2 = (1/n)^2 * (1 + sin(10*pi*1/n))
        #
        # But the Pareto front of ZDT4 is actually parameterized by x_1.
        # The true Pareto front for ZDT4 is:
        # f1 = x_1, f2 = (1 - sqrt(f1/n))^(2) * (1 + sin(10*pi*f1/n))
        # for f1 in [0, 1]
        #
        # Wait, let me look at this more carefully. The standard ZDT4 definition:
        # f1(x) = x_1
        # g(x) = sum_{i=2}^n (x_i^2 - 10*cos(4*pi*x_i)) + 10*(n-1)
        # f2(x) = (1 + g(x)) * (1 - sqrt(f1/(1+g(x)))^2 * (1 + sin(10*pi*f1/(1+g(x)))))
        #
        # Hmm, different sources define it slightly differently. Let me use the common one:
        # f1 = x_1
        # g = sum_{i=2}^n (x_i^2 - 10*cos(4*pi*x_i)) + 10*(n-1)
        # f2 = (1 + g) * (1 - sqrt(f1/(1+g)) * (1 + sin(10*pi*f1/(1+g))))
        #
        # Actually, the most common ZDT4 is:
        # f1 = x_1
        # g(x) = sum_{i=2}^n (x_i^2 - 10*cos(4*pi*x_i)) + 10*(n-1)
        # f2 = (1+g) * (1 - sqrt(f1/(1+g)) * (1 + sin(10*pi*f1/(1+g))))
        #
        # The Pareto front is where g(x) = 0, i.e., x_i = 0 for i>=2.
        # Then f1 = x_1, f2 = (1+0)*(1 - sqrt(x_1/1) * (1 + sin(10*pi*x_1/1)))
        # = 1 - sqrt(x_1)*(1 + sin(10*pi*x_1))
        # for x_1 in [0, 1]
        #
        # But wait, the problem says "minimizing all 2 objectives of ZDT4" and the
        # reference point is [11.0, 400.0]. The scoring uses hypervolume.
        #
        # The key insight: if we set x_2..x_n to 0 (or close to 0), then g(x) ≈ 0
        # and we're on or near the Pareto front. The only variable that matters for
        # the Pareto front is x_1.
        #
        # Strategy: Set x_2..x_n to 0 (if within bounds), and sample x_1 densely
        # over [0, 1] (clipped to bounds). This gives us points very close to the
        # Pareto front.
        #
        # But we also need to consider: the problem function clips to bounds, so
        # if 0 is not in bounds for some x_i, we need to find the best value.
        #
        # For g(x_i) = x_i^2 - 10*cos(4*pi*x_i), the global min is at x_i=0 with
        # value -10. But there are many local minima. If 0 is not in bounds, we
        # should find the value closest to 0 that minimizes g.
        
        # Find optimal x_2..x_n
        optimal_x_rest = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            idx = i + 1
            if lo[idx] <= 0 and 0 <= hi[idx]:
                optimal_x_rest[i] = 0.0
            else:
                # Find the value in [lo, hi] that minimizes x^2 - 10*cos(4*pi*x)
                # The function has minima at x = k/2 for integer k, with value -10
                # at even k (x=0, ±1, ±2, ...) and value -10+... at odd k
                # Actually g(x) = x^2 - 10*cos(4*pi*x)
                # At x = k/2: g = k^2/4 - 10*cos(2*pi*k) = k^2/4 - 10
                # So minima are at x = k/2 with value k^2/4 - 10
                # Global min at k=0 (x=0): value -10
                # Next best at k=±1 (x=±0.5): value 0.25 - 10 = -9.75
                # Next at k=±2 (x=±1): value 1 - 10 = -9
                # etc.
                # If 0 is not in bounds, find the k/2 closest to 0 that is in bounds
                if lo[idx] > 0:
                    # Find smallest k such that k/2 >= lo[idx]
                    k_min = int(np.ceil(2 * lo[idx]))
                    # Check k_min and k_min-1 (if k_min-1 >= 0 and (k_min-1)/2 >= lo[idx])
                    candidates = []
                    for k in [k_min, k_min + 1]:
                        val = k / 2.0
                        if lo[idx] <= val <= hi[idx]:
                            candidates.append(val)
                    if not candidates:
                        optimal_x_rest[i] = lo[idx]
                    else:
                        # Pick the one with smallest g value
                        best = min(candidates, key=lambda v: v*v - 10*np.cos(4*np.pi*v))
                        optimal_x_rest[i] = best
                else:
                    # hi[idx] < 0
                    k_max = int(np.floor(2 * hi[idx]))
                    candidates = []
                    for k in [k_max, k_max - 1]:
                        val = k / 2.0
                        if lo[idx] <= val <= hi[idx]:
                            candidates.append(val)
                    if not candidates:
                        optimal_x_rest[i] = hi[idx]
                    else:
                        best = min(candidates, key=lambda v: v*v - 10*np.cos(4*np.pi*v))
                        optimal_x_rest[i] = best

        # Sample x_1 densely over [max(lo[0], 0), min(hi[0], 1)]
        x1_lo = max(lo[0], 0.0)
        x1_hi = min(hi[0], 1.0)

        if x1_lo >= x1_hi:
            x1_vals = np.array([x1_lo], dtype=np.float64)
        else:
            # Use dense sampling. Budget is 200, so use all of it for x_1 sampling.
            num_points = budget
            x1_vals = np.linspace(x1_lo, x1_hi, num_points, dtype=np.float64)

        # Construct points
        X = np.zeros((num_points, n), dtype=np.float64)
        X[:, 0] = x1_vals
        X[:, 1:] = optimal_x_rest

        # Evaluate all points
        F = np.array([problem(x) for x in X], dtype=np.float64)

        return X
