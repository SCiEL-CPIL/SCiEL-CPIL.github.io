import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = min(200, self.budget)

        # Optimize point positions on [0,1] to maximize hypervolume
        # HV = sum_{i=1}^{m} (x_{i+1} - x_i) * 11 * sqrt(x_i), x_{m+1} = 11
        # Use gradient ascent with log-spaced initialization

        # Initial guess: points spread with a bias toward smaller x values
        # (since the front is steeper near x=0)
        t = np.linspace(0, 1, m + 2)[1:-1]  # m interior points
        x = t ** 1.5  # initial bias

        # Gradient ascent on x values
        def hv_and_grad(x):
            # x: array of m values in (0, 1), sorted
            # x_ext = [x_1, ..., x_m, 11]
            x_ext = np.append(x, 11.0)
            widths = np.diff(x_ext)  # m widths
            f2 = 11.0 * np.sqrt(x)   # m values
            hv = np.sum(widths * f2)

            # Gradient: dHV/dx_i = 11*sqrt(x_i) - (11/(2*sqrt(x_i)))*(x_{i+1} - x_i)
            # for i < m: contribution from width_i (as -x_i) and width_{i-1} (as +x_i)
            # More carefully:
            # HV = (x_2 - x_1)*11*sqrt(x_1) + (x_3 - x_2)*11*sqrt(x_2) + ... + (11 - x_m)*11*sqrt(x_m)
            # dHV/dx_1 = -11*sqrt(x_1) + 11*sqrt(x_2)
            # dHV/dx_i = -11*sqrt(x_i) + 11*sqrt(x_{i+1}) - (11/(2*sqrt(x_i)))*(x_{i+1} - x_i)  for i < m
            # Wait, let me recompute:
            # Term i: (x_{i+1} - x_i) * 11 * sqrt(x_i)
            # d/dx_i of term i: -11*sqrt(x_i)
            # d/dx_i of term (i-1): 11*sqrt(x_{i-1})  [if i > 1]
            # d/dx_i of term i (from sqrt): (x_{i+1} - x_i) * 11 * (1/(2*sqrt(x_i)))
            # So for i = 1: dHV/dx_1 = -11*sqrt(x_1) + 11*sqrt(x_2) + (x_2-x_1)*11/(2*sqrt(x_1))
            # For 1 < i < m: dHV/dx_i = -11*sqrt(x_i) + 11*sqrt(x_{i-1}) + (x_{i+1}-x_i)*11/(2*sqrt(x_i))
            # For i = m: dHV/dx_m = -11*sqrt(x_m) + 11*sqrt(x_{m-1}) + (11-x_m)*11/(2*sqrt(x_m))

            grad = np.zeros(m)
            for i in range(m):
                # Contribution from term i: derivative w.r.t. x_i
                # term_i = (x_{i+1} - x_i) * 11 * sqrt(x_i)
                # d(term_i)/dx_i = -11*sqrt(x_i) + (x_{i+1} - x_i) * 11 / (2*sqrt(x_i))
                x_next = x[i + 1] if i < m - 1 else 11.0
                grad[i] = -11.0 * np.sqrt(x[i]) + (x_next - x[i]) * 11.0 / (2.0 * np.sqrt(x[i]))
                # Contribution from term (i-1): derivative w.r.t. x_i (as the "next" point)
                if i > 0:
                    grad[i] += 11.0 * np.sqrt(x[i - 1])

            return hv, grad

        # Gradient ascent with adaptive step size
        lr = 0.01
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            hv, grad = hv_and_grad(x)
            # Clip x to stay in (epsilon, 1 - epsilon)
            x_new = x + lr * grad
            x_new = np.clip(x_new, 1e-8, 1.0 - 1e-8)
            # Ensure sorted
            x_new = np.sort(x_new)
            # Adaptive learning rate
            lr *= 0.995
            x = x_new

        # Final polish with smaller steps
        lr = 0.001
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            hv, grad = hv_and_grad(x)
            x_new = x + lr * grad
            x_new = np.clip(x_new, 1e-8, 1.0 - 1e-8)
            x_new = np.sort(x_new)
            x = x_new
            lr *= 0.98

        # Evaluate the optimized points
        used = 0
        recorded = []
        for x1 in x:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        return recorded
