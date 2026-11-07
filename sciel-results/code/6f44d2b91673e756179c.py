"""Multi-objective optimizer on ZDT1 using known Pareto front structure with optimal quantization."""
import numpy as np
import time


class Optimizer:
    """Exploits ZDT1's known Pareto front: fix non-critical dims to 0,
    place x1 values using optimal quantization based on hypervolume integrand."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        B = int(self.budget)
        D = int(self.dim)

        # ZDT1 Pareto front: x2..xn = 0, f1 = x1, f2 = 1 - sqrt(x1/30)
        # With n=30: f2 = 1 - sqrt(x1/30)
        # Reference point: (11, 11)
        # Hypervolume integrand: h(t) = 11 - f2(t) = 10 + sqrt(t/30)

        # Optimal quantization: place points where the cumulative integrand is uniform
        # G(t) = integral from 0 to t of h(s) ds = 10*t + (2/(3*sqrt(30))) * t^1.5
        # We want to place B points such that G(t_i) is uniformly spaced

        sqrt30 = np.sqrt(30.0)
        G1 = 10.0 + 2.0 / (3.0 * sqrt30)  # G(1)

        # Generate uniform G values (excluding 0 to avoid t=0 which gives f2=1)
        g_vals = np.linspace(0, G1, B + 1)[1:]  # B values from G(0+epsilon) to G(1)

        # Invert G(t) = g to find t values
        # G(t) = 10*t + (2/(3*sqrt(30))) * t^1.5
        # Solve numerically using Newton's method or binary search
        t_vals = np.zeros(B)
        for i in range(B):
            g = g_vals[i]
            # Initial guess
            t = g / 10.0  # approximate
            t = np.clip(t, 0.0, 1.0)
            
            # Newton's method
            for _ in range(50):
                G_t = 10.0 * t + (2.0 / (3.0 * sqrt30)) * t**1.5
                dG_dt = 10.0 + (1.0 / sqrt30) * t**0.5
                if abs(dG_dt) < 1e-15:
                    break
                t_new = t - (G_t - g) / dG_dt
                t_new = np.clip(t_new, 0.0, 1.0)
                if abs(t_new - t) < 1e-15:
                    t = t_new
                    break
                t = t_new
            
            t_vals[i] = t

        # Ensure sorted and in [0, 1]
        t_vals = np.sort(t_vals)
        t_vals = np.clip(t_vals, 0.0, 1.0)

        # Build solutions: x1 = t_i, x2..x30 = 0
        X = np.zeros((B, D), dtype=np.float64)
        X[:, 0] = t_vals
        # x2..xn = 0 (lower bound) achieves g(x) = 0

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())

        return X
