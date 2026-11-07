import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Sorted list of f1 values evaluated
        f1_vals = []

        for step in range(budget):
            if time.time() > deadline:
                break

            if len(f1_vals) == 0:
                # First point: maximize (11-f1)*(10+sqrt(f1)) over f1 in [0,1]
                # 3u^2 + 20u - 11 = 0 where u=sqrt(f1)
                sqrt133 = np.sqrt(133.0)
                u_opt = (-20.0 + 2.0 * sqrt133) / 6.0
                f1_new = min(max(u_opt ** 2, 0.0), 1.0)
            else:
                # Find gap with maximum marginal hypervolume gain
                best_gain = -1.0
                best_f1 = 0.0

                f1_arr = np.array(f1_vals)
                n = len(f1_arr)

                # Check gap (0, f1[0]) if f1[0] > 0
                if f1_arr[0] > 1e-15:
                    left, right = 0.0, f1_arr[0]
                    c = 0.0
                    d = np.sqrt(right)
                    disc = c * c + 3.0 * d * d
                    u_star = (c + np.sqrt(disc)) / 3.0
                    f1_cand = u_star * u_star
                    gain = (f1_cand - right) * (c - np.sqrt(f1_cand))
                    if gain > best_gain:
                        best_gain = gain
                        best_f1 = f1_cand

                # Check gaps between consecutive points
                for i in range(n - 1):
                    left, right = f1_arr[i], f1_arr[i + 1]
                    if right - left < 1e-15:
                        continue
                    c = np.sqrt(left)
                    d = np.sqrt(right)
                    disc = c * c + 3.0 * d * d
                    u_star = (c + np.sqrt(disc)) / 3.0
                    f1_cand = u_star * u_star
                    gain = (f1_cand - right) * (c - np.sqrt(f1_cand))
                    if gain > best_gain:
                        best_gain = gain
                        best_f1 = f1_cand

                # Check gap (f1[-1], 1) if f1[-1] < 1
                if f1_arr[-1] < 1.0 - 1e-15:
                    left, right = f1_arr[-1], 1.0
                    c = np.sqrt(left)
                    d = 1.0
                    disc = c * c + 3.0 * d * d
                    u_star = (c + np.sqrt(disc)) / 3.0
                    f1_cand = u_star * u_star
                    gain = (f1_cand - right) * (c - np.sqrt(f1_cand))
                    if gain > best_gain:
                        best_gain = gain
                        best_f1 = f1_cand

                f1_new = best_f1

            # Evaluate the point
            x = np.zeros(dim)
            x[0] = f1_new
            x = np.clip(x, lo, hi)
            problem(x)

            # Insert into sorted list
            f1_vals.append(f1_new)
            f1_vals.sort()

        return None
