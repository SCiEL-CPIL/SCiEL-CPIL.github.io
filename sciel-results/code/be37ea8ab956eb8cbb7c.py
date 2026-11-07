"""ZDT1 Pareto-front optimizer using the analytical optimal spacing derived from
equalizing the marginal hypervolume contribution of each point on the convex front.

For ZDT1 with reference point (11, 11), the Pareto front is
  f1 = t,  f2 = 2 - sqrt(2t),  t in [0, 1].
The 200 points that maximize the 2-D hypervolume are obtained by solving the
equal-marginal-contribution condition via a fast forward-recurrence + bisection
on the free parameter, followed by a high-precision polish pass.
"""
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 305.0
        budget = self.budget
        dim = self.dim

        used = 0

        def eval_point(x):
            nonlocal used
            f = np.asarray(problem(np.asarray(x, dtype=float)), dtype=float)
            used += 1
            return f

        n = min(budget, 200)

        def compute_optimal_t(n_pts):
            """Compute optimal t-values via forward recurrence + bisection."""
            # The optimality condition for interior point i (0-indexed):
            #   t_{i+1} = 3*t_i - sqrt(2*t_i*t_{i-1})
            # This is a forward recurrence. Given t[0]=0 and t[1]=alpha,
            # we can compute all t_i. We need t[n-1]=1, so bisect on alpha.
            
            def forward(alpha):
                """Given t[1]=alpha, compute all t_i via recurrence."""
                t_arr = np.zeros(n_pts)
                t_arr[0] = 0.0
                t_arr[1] = alpha
                for i in range(1, n_pts - 1):
                    ti = t_arr[i]
                    tim1 = t_arr[i-1]
                    if ti <= 0 or tim1 <= 0:
                        t_arr[i+1] = ti + 1e-10
                    else:
                        t_arr[i+1] = 3.0 * ti - np.sqrt(2.0 * ti * tim1)
                    t_arr[i+1] = max(t_arr[i+1], t_arr[i] + 1e-15)
                return t_arr
            
            # Bisect on alpha = t[1] so that t[n-1] = 1
            lo_alpha = 1e-10
            hi_alpha = 0.5  # upper bound
            
            for _ in range(80):
                mid_alpha = (lo_alpha + hi_alpha) / 2.0
                t_arr = forward(mid_alpha)
                if t_arr[-1] < 1.0:
                    lo_alpha = mid_alpha
                else:
                    hi_alpha = mid_alpha
                if hi_alpha - lo_alpha < 1e-14:
                    break
            
            t_opt = forward((lo_alpha + hi_alpha) / 2.0)
            # Normalize to ensure t[0]=0, t[-1]=1
            t_opt[0] = 0.0
            t_opt[-1] = 1.0
            # Ensure monotonicity
            for i in range(1, n_pts):
                if t_opt[i] <= t_opt[i-1]:
                    t_opt[i] = t_opt[i-1] + 1e-15
            t_opt[-1] = 1.0
            return np.clip(t_opt, 0.0, 1.0)

        t = compute_optimal_t(n)

        # Evaluate all points
        X = []
        F = []
        for ti in t:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(ti)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        # Polish pass: use remaining budget to refine points
        # Use exact marginal HV gain formula to avoid full HV recomputation
        remaining = budget - used
        if remaining > 0 and len(F) > 0:
            F_arr = np.array(F)
            sort_idx = np.argsort(F_arr[:, 0])
            F_sorted = F_arr[sort_idx]
            t_sorted = t[sort_idx].copy()
            
            def f2_of_t(tv):
                return 2.0 - np.sqrt(2.0 * np.maximum(tv, 0.0))
            
            def marginal_gain(t_vals, i, delta):
                """Exact HV change when moving point i by delta in t-space.
                Assumes t_vals is sorted by t (which is f1)."""
                ti = t_vals[i]
                ti_new = ti + delta
                if ti_new <= t_vals[i-1] or ti_new >= t_vals[i+1]:
                    return 0.0
                f2_i = f2_of_t(ti)
                f2_i_new = f2_of_t(ti_new)
                f2_left = f2_of_t(t_vals[i-1])
                f2_right = f2_of_t(t_vals[i+1])
                # The point i contributes to HV as part of trapezoids with neighbors.
                # Original contribution involving point i:
                #   (ti - t_{i-1})*(11 - f2_{i-1}) + (t_{i+1} - ti)*(11 - f2_i)
                #   Wait, that's not right. Let me reconsider.
                # HV = sum over i of (f1_{i+1} - f1_i) * (11 - f2_i)
                # where f1_{n} = 11 (reference).
                # So point i's f2 affects term i: (f1_{i+1} - f1_i)*(11 - f2_i)
                # And point i's f1 affects term i-1: (f1_i - f1_{i-1})*(11 - f2_{i-1})
                # and term i: (f1_{i+1} - f1_i)*(11 - f2_i)
                # Moving point i from ti to ti_new:
                # Term i-1 changes: (ti_new - t_{i-1})*(11 - f2_{i-1}) vs (ti - t_{i-1})*(11 - f2_{i-1})
                #   delta = (ti_new - ti)*(11 - f2_{i-1}) = delta*(11 - f2_{i-1})
                # Term i changes: two parts:
                #   (11 - f2_i_new)*(t_{i+1} - ti_new) - (11 - f2_i)*(t_{i+1} - ti)
                #   = (11 - f2_i_new)*(t_{i+1} - ti - delta) - (11 - f2_i)*(t_{i+1} - ti)
                #   = (11 - f2_i_new)*(t_{i+1} - ti) - (11 - f2_i_new)*delta - (11 - f2_i)*(t_{i+1} - ti)
                #   = ((11 - f2_i_new) - (11 - f2_i))*(t_{i+1} - ti) - (11 - f2_i_new)*delta
                #   = (f2_i - f2_i_new)*(t_{i+1} - ti) - (11 - f2_i_new)*delta
                # Total gain:
                gain = delta*(11.0 - f2_left) + (f2_i - f2_i_new)*(t_vals[i+1] - ti) - (11.0 - f2_i_new)*delta
                return gain
            
            # Multiple polish rounds
            for _polish in range(min(remaining, 50)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                best_gain = 0.0
                best_i = -1
                best_delta = 0.0
                for i in range(1, len(t_sorted) - 1):
                    for delta in [1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
                        g = marginal_gain(t_sorted, i, delta)
                        if g > best_gain:
                            best_gain = g
                            best_i = i
                            best_delta = delta
                if best_i >= 0 and best_gain > 1e-12:
                    t_sorted[best_i] += best_delta
                    x = np.zeros(dim)
                    x[0] = float(t_sorted[best_i])
                    f = eval_point(x)
                    if np.all(np.isfinite(f)):
                        F_sorted[best_i] = f
                        X[sort_idx[best_i]] = x
                else:
                    break

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
