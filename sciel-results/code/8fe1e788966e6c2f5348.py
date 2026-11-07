import numpy as np

def heuristics(demands, capacity):
        n = len(demands)
        # Ensure demands are floats
        d = np.array(demands, dtype=float)
        
        # Pairwise sums
        sums = d[:, None] + d[None, :]
        
        # Mask for valid pairs
        valid = sums <= capacity
        
        # Product
        prods = d[:, None] * d[None, :]
        
        # Normalize
        # Option 1: Pure Product
        # eta = prods / (capacity * capacity)
        
        # Option 2: Product + Sum
        # eta = prods / (capacity * capacity) + 0.5 * (sums / capacity)
        
        # Let's try Option 2
        eta = (prods / (capacity * capacity)) + 0.5 * (sums / capacity)
        
        # Apply mask
        eta = np.where(valid, eta, 1e-9)
        
        return eta
