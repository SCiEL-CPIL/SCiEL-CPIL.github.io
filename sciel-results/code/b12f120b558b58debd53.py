import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    denominators = list(range(2, 49))
    denominators.sort()
    
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Suffix sums for pruning
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + reciprocals[i]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Use a list to hold state to avoid passing lists recursively (optimization)
    # But for N=47, passing lists is fine.
    
    def search(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
        
        # Pruning
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum and current_sum < 1:
                best_sum = current_sum
                best_indices = current_indices[:]
            return
        
        # Include
        next_sum = current_sum + reciprocals[idx]
        if next_sum < 1:
            current_indices.append(idx)
            search(idx + 1, next_sum, current_indices)
            current_indices.pop()
            
        # Exclude
        search(idx + 1, current_sum, current_indices)
        
    search(0, Fraction(0), [])
    
    return best_indices
