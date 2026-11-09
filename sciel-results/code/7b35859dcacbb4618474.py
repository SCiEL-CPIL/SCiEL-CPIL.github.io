import numpy as np

def heuristics(demands, capacity):
        n = len(demands)
        d = np.array(demands, dtype=float)
        c = float(capacity)
        
        # Sum of pairs
        pair_sums = d[:, None] + d[None, :]
        
        # Feasibility mask
        feasible = pair_sums <= c
        
        # Residual space
        residuals = c - pair_sums
        
        # Smallest item size
        w_min = np.min(d)
        
        # Avoid division by zero if w_min is 0 (though demands are 20-99, so w_min >= 20)
        if w_min == 0:
             w_min = 1.0
             
        # Modulo of residual by w_min
        # We want R % w_min to be small.
        # remainder = residuals % w_min
        # fit_score = 1 - (remainder / w_min)
        
        remainder = residuals % w_min
        fit_score = 1.0 - (remainder / w_min)
        
        # Tightness score: prefer larger sums
        # tightness = pair_sums / c
        # But if pair_sums is 0, this is 0.
        # We want to avoid 0 heuristic values if possible, or let the ACO handle them.
        # The contract says non-positive becomes 1e-9.
        
        # Combined heuristic
        # h = tightness * fit_score
        # This rewards pairs that are both tight and have clean residuals.
        
        h = (pair_sums / c) * fit_score
        
        # Apply feasibility
        h = np.where(feasible, h, 1e-9)
        
        # Ensure no negative values (shouldn't happen, but safe)
        h = np.maximum(h, 1e-9)
        
        return h
