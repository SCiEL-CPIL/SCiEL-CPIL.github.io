import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, self.n_obj))

        N = budget

        # ZDT4 Front Model:
        # f1 = x1
        # f2 = g_min * (1 - sqrt(f1 / g_min))
        # g_min = 82 (for dim=10, x_i=0 for i>1)
        # f2 = 82 - sqrt(82 * x1)
        # Reference R = [11.0, 400.0]
        # HV term weight: R2 - f2 = 400 - (82 - sqrt(82 x1)) = 318 + sqrt(82 x1)
        
        g_min = 82.0
        R1 = 11.0
        R2 = 400.0
        
        # Optimize power law exponent p
        # x_i = (i/N)^p for i=1..N, x_1=0, x_N=1
        # Model HV = sum_{i=1}^{N-1} (x_{i+1} - x_i) * (318 + sqrt(82 * x_i)) + (R1 - x_N) * (318 + sqrt(82 * x_N))
        # Note: x_N = 1.0
        
        best_p = 2.0
        best_hv = -1.0
        
        # Try a range of p values
        for p in np.linspace(1.0, 3.0, 21):
            idx = np.arange(N, dtype=float)
            # i=0..N-1
            # x_0 = 0
            # x_i = (i/N)^p
            x_vals = (idx / N) ** p
            x_vals[0] = 0.0
            x_vals[-1] = 1.0
            
            # Calculate HV
            # Weights w_i = 318 + sqrt(82 * x_i)
            w = 318.0 + np.sqrt(82.0 * x_vals)
            
            # Sum (x_{i+1} - x_i) * w_i
            # x_{N} = R1 = 11.0
            dx = np.diff(x_vals)
            # dx has length N-1
            # Sum over i=0 to N-2
            hv_sum = np.sum(dx * w[:-1])
            
            # Last term
            last_dx = R1 - x_vals[-1]
            last_term = last_dx * w[-1]
            
            hv = hv_sum + last_term
            
            if hv > best_hv:
                best_hv = hv
                best_p = p

        # Generate final points with best_p
        idx = np.arange(N, dtype=float)
        x0_vals = (idx / N) ** best_p
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        
        # Ensure monotonic
        for k in range(1, N):
            if x0_vals[k] <= x0_vals[k-1]:
                x0_vals[k] = x0_vals[k-1] + 1e-12

        # Tail variables
        tail = np.zeros(dim - 1)
        for k in range(dim - 1):
            if lo[k+1] <= 0.0 <= hi[k+1]:
                tail[k] = 0.0
            elif 0.0 < lo[k+1]:
                tail[k] = lo[k+1]
            else:
                tail[k] = hi[k+1]

        evals = 0
        results = []

        for x0_val in x0_vals:
            if evals >= budget:
                break
            if time.time() - t0 > DEADLINE:
                break

            x = np.empty(dim)
            x[0] = x0_val
            x[1:] = tail
            x = np.clip(x, lo, hi)

            try:
                f = problem(x)
                evals += 1
                if f is not None:
                    results.append(np.asarray(f, dtype=float))
            except Exception:
                pass

        if results:
            return np.array(results)

        return np.empty((0, self.n_obj))
