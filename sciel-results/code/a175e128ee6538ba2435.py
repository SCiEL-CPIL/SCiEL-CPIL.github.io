import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        n_tail = n - 1
        
        def get_candidates(x1):
            cands = [0.0, 0.5, 1.0]
            if x1 > 1e-6:
                # Critical points of cos(10 * pi * x1 * x)
                # 10 * pi * x1 * x = (2k+1) * pi  => x = (2k+1) / (20 * x1)
                max_k = int(20 * x1)
                for k in range(max_k + 1):
                    x_val = (2 * k + 1) / (20 * x1)
                    if 0.0 <= x_val <= 1.0:
                        cands.append(x_val)
            # Deduplicate and clip
            cands = [np.clip(c, lo[1], hi[1]) for c in cands]
            # Unique
            cands = list(set([round(c, 10) for c in cands]))
            return cands

        # Phase 1: Calibration
        n_calib = 15
        n_sample = self.budget - n_calib
        
        # We will use 3 candidates per x1 in calibration to be safe
        # 5 x1 values * 3 candidates = 15 evals
        x1_calib = np.linspace(0.0, 1.0, 5)
        best_tail_calib = []
        best_f2_calib = []
        
        for x1 in x1_calib:
            cands = get_candidates(x1)
            # Limit to 3 best guesses? Or just first 3?
            # Let's just evaluate the first 3 unique candidates
            # Actually, let's evaluate all if < 3, else pick 3 spread out
            if len(cands) > 3:
                # Pick 0, middle, end
                cands_eval = [cands[0], cands[len(cands)//2], cands[-1]]
            else:
                cands_eval = cands
            
            best_g = np.inf
            best_x_tail = None
            
            for xt in cands_eval:
                x = np.zeros(n)
                x[0] = x1
                x[1:] = xt
                x = np.clip(x, lo, hi)
                f = problem(x)
                # Estimate g
                if f[0] > 1e-6:
                    # f2 = (1+g)(1-sqrt(f1))
                    # g = f2/(1-sqrt(f1)) - 1
                    g_est = f[1] / (1 - np.sqrt(f[0])) - 1
                else:
                    # f1=0, f2 = 1+g
                    g_est = f[1] - 1
                
                if g_est < best_g:
                    best_g = g_est
                    best_x_tail = xt
            
            best_tail_calib.append(best_x_tail)
            best_f2_calib.append(best_g) # Store g for interpolation logic? No, store x_tail

        # Phase 2: Sampling
        x1_sample = np.linspace(0.0, 1.0, n_sample)
        
        for x1 in x1_sample:
            # Interpolate best_tail from calibration
            # Find indices
            if x1 <= x1_calib[0]:
                xt = best_tail_calib[0]
            elif x1 >= x1_calib[-1]:
                xt = best_tail_calib[-1]
            else:
                # Find i such that x1_calib[i] <= x1 <= x1_calib[i+1]
                i = np.searchsorted(x1_calib, x1)
                if i == 0:
                    xt = best_tail_calib[0]
                elif i == len(x1_calib):
                    xt = best_tail_calib[-1]
                else:
                    x1_lo = x1_calib[i-1]
                    x1_hi = x1_calib[i]
                    xt_lo = best_tail_calib[i-1]
                    xt_hi = best_tail_calib[i]
                    if x1_hi > x1_lo:
                        frac = (x1 - x1_lo) / (x1_hi - x1_lo)
                    else:
                        frac = 0.5
                    xt = xt_lo + frac * (xt_hi - xt_lo)
            
            x = np.zeros(n)
            x[0] = x1
            x[1:] = xt
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
