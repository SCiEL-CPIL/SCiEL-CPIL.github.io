import numpy as np

def heuristics(prizes, weights):
        """
        Compute attractiveness scores using a combined Sum+Max weight penalty.
        """
        eps = 1e-9
        
        # Aggregate prizes to shape (100,)
        # Assuming prizes are replicated or should be averaged
        item_prizes = prizes.mean(axis=0)
        
        # Aggregate weights to shape (100,)
        # Sum captures total resource, Max captures bottleneck
        sum_weights = weights.sum(axis=0)
        max_weights = weights.max(axis=0)
        
        # Combined denominator
        denom = sum_weights + max_weights + eps
        
        # Heuristic
        heuristic = item_prizes / denom
        
        # Ensure positivity and finiteness
        heuristic = np.maximum(heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        
        return heuristic
