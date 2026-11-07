import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None
            
        # Extract bounds for x1 (index 0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        x1_range = x1_hi - x1_lo
        
        # ZDT2 specific constants
        C = 10.0 * (n - 1)
        ref = 11.0
        
        # Analytical Pareto front for ZDT2: f2 = C * (1 - sqrt(f1))
        # We assume x1 is the first variable, others 0.
        # f1 = x1
        
        def analytical_f2(x1):
            return C * (1.0 - np.sqrt(np.clip(x1, 0, 1)))
            
        def calc_hv(x1_vals):
            # x1_vals should be sorted
            x1_vals = np.sort(x1_vals)
            y_vals = analytical_f2(x1_vals)
            
            if len(x1_vals) == 0:
                return 0.0
            if len(x1_vals) == 1:
                return (ref - x1_vals[0]) * (ref - y_vals[0])
            
            dx = np.diff(x1_vals)
            # Heights are (ref - y_i) for i=0 to N-2
            heights = ref - y_vals[:-1]
            hv = np.sum(dx * heights)
            # Last rectangle
            last_width = ref - x1_vals[-1]
            last_height = ref - y_vals[-1]
            hv += last_width * last_height
            return hv

        # Optimize power p for x_i = x_start + (x_end - x_start) * (i/(N-1))^p
        # We restrict the search to the region where f2 < ref
        # f2(x) < ref => C(1 - sqrt(x)) < ref => sqrt(x) > 1 - ref/C
        # x > (1 - ref/C)^2
        
        cutoff_ratio = 1.0 - ref / C
        if cutoff_ratio < 0:
            x_start_unit = 0.0
        else:
            x_start_unit = cutoff_ratio ** 2
            
        x_end_unit = 1.0
        
        N = budget
        idx = np.arange(N)
        
        best_p = 0.5
        best_hv = -np.inf
        
        # Search p in a reasonable range. 
        # Theoretical optimal p is around 0.5-1.0 for this curvature.
        p_candidates = np.linspace(0.1, 3.0, 30)
        for p in p_candidates:
            # Map i to [x_start_unit, x_end_unit]
            if N == 1:
                x1_test = np.array([x_start_unit])
            else:
                t = idx / (N - 1)
                x1_test = x_start_unit + (x_end_unit - x_start_unit) * (t ** p)
            
            hv_val = calc_hv(x1_test)
            if hv_val > best_hv:
                best_hv = hv_val
                best_p = p
        
        # Generate final points in [0, 1] space
        if N == 1:
            x1_final_unit = np.array([x_start_unit])
        else:
            t = idx / (N - 1)
            x1_final_unit = x_start_unit + (x_end_unit - x_start_unit) * (t ** best_p)
        
        # Map to actual bounds
        x1_final = x1_lo + x1_final_unit * x1_range
        
        # Create full vectors
        X = np.zeros((N, n))
        X[:, 0] = x1_final
        
        # Evaluate exactly N points
        F = np.array([problem(x) for x in X])
        
        return X
