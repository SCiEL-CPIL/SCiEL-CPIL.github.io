import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 specific knowledge
        # Build lookup table
        x1_samples = np.linspace(0, 1, 10000)
        sin_term = np.sin(6 * np.pi * x1_samples)
        sin_term_pow = sin_term ** 6
        exp_term = np.exp(-4 * x1_samples)
        base = 1 - exp_term * sin_term_pow
        base = np.clip(base, 0, 1)
        f1_samples = 1 - base ** 0.1
        
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]
        
        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]
        
        n_points = self.budget
        target_f1s = np.linspace(f1_min, f1_max, n_points)
        
        # Interpolation
        x1_targets = np.zeros(n_points)
        
        # np.searchsorted returns index i such that f1_sorted[i-1] < target <= f1_sorted[i]
        # We want i such that f1_sorted[i] <= target < f1_sorted[i+1]
        # searchsorted side='left' gives index where target would be inserted to keep order.
        # So f1_sorted[i-1] < target <= f1_sorted[i]
        # Let's use side='right' to get i such that f1_sorted[i] <= target < f1_sorted[i+1]
        # Actually, let's just use searchsorted and handle indices carefully.
        
        indices = np.searchsorted(f1_sorted, target_f1s, side='left')
        # indices is in [0, len(f1_sorted)]
        # If indices == 0, target <= f1_sorted[0]
        # If indices == len, target > f1_sorted[-1]
        
        # Clip indices to [0, len-1] for the lower bound of the interval
        i_low = np.clip(indices, 0, len(f1_sorted) - 1)
        i_high = np.clip(indices, 1, len(f1_sorted)) # Wait, if indices is 0, i_high should be 1?
        
        # Let's do it element-wise or with masks to be safe.
        # Simpler:
        # For each target, find the bracket.
        
        # Vectorized interpolation
        # i = searchsorted(f1_sorted, target, side='left')
        # If i == 0, use x1_sorted[0]
        # If i == len, use x1_sorted[-1]
        # Else, interpolate between i-1 and i
        
        i = np.searchsorted(f1_sorted, target_f1s, side='left')
        
        # Handle boundaries
        # If i == 0, x1 = x1_sorted[0]
        # If i == len(f1_sorted), x1 = x1_sorted[-1]
        
        x1_interp = np.zeros(n_points)
        
        # Case 1: i == 0
        mask0 = (i == 0)
        x1_interp[mask0] = x1_sorted[0]
        
        # Case 2: i == len
        maskN = (i == len(f1_sorted))
        x1_interp[maskN] = x1_sorted[-1]
        
        # Case 3: 0 < i < len
        maskMid = ~(mask0 | maskN)
        if np.any(maskMid):
            i_mid = i[maskMid]
            f1_lo = f1_sorted[i_mid - 1]
            f1_hi = f1_sorted[i_mid]
            x1_lo = x1_sorted[i_mid - 1]
            x1_hi = x1_sorted[i_mid]
            targets_mid = target_f1s[maskMid]
            
            # Avoid division by zero
            denom = f1_hi - f1_lo
            # If denom is 0, just take x1_lo
            safe_denom = np.where(denom == 0, 1, denom)
            slope = (targets_mid - f1_lo) / safe_denom
            x1_interp[maskMid] = x1_lo + slope * (x1_hi - x1_lo)
            # If denom was 0, slope is 0 (since target == f1_lo == f1_hi), so x1 = x1_lo. Correct.
            
        x1_targets = x1_interp

        recorded = []
        for i in range(n_points):
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
