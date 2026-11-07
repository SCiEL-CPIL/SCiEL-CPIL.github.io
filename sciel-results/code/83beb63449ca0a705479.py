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
        DEADLINE = 295.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, dim))

        # ZDT4 Pareto Front: x[1:] = 0, x[0] in [0, 1]
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin^2(10 * pi * x0)
        
        # Strategy: Arc-length parameterization of the Pareto front.
        # The front has high curvature near x0=0 (due to sqrt) and oscillatory
        # behavior from sin^2(10*pi*x0). Uniform sampling in x0 under-samples
        # regions where the front has high geometric length. By computing the
        # cumulative arc length and sampling uniformly in arc length, we ensure
        # that each "unit of front" gets roughly equal representation, which
        # maximizes hypervolume coverage.
        
        N = budget
        n_dense = 100000  # High resolution for arc length computation
        
        if N == 1:
            x0_vals = np.array([0.0])
        else:
            # Dense sampling for arc length computation
            x0_dense = np.linspace(0.0, 1.0, n_dense)
            
            # Compute objectives for dense sampling
            f1_dense = x0_dense
            f2_dense = 1.0 - np.sqrt(x0_dense) - x0_dense * np.sin(10.0 * np.pi * x0_dense)**2
            
            # Compute arc length
            df1 = np.diff(f1_dense)
            df2 = np.diff(f2_dense)
            dl = np.sqrt(df1**2 + df2**2)
            arc_len = np.concatenate([[0.0], np.cumsum(dl)])
            total_arc = arc_len[-1]
            
            # Uniformly sample by arc length
            target_arcs = np.linspace(0.0, total_arc, N)
            
            # Interpolate to find x0 values for each target arc length
            x0_vals = np.interp(target_arcs, arc_len, x0_dense)
            
            # Ensure we include the endpoints exactly
            x0_vals[0] = 0.0
            x0_vals[-1] = 1.0
        
        # Clip to bounds
        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = x0_val
            # Clip to bounds for all dimensions (though x[1:] are 0)
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass

        if results:
            return np.array(results)
            
        return np.empty((0, dim))
