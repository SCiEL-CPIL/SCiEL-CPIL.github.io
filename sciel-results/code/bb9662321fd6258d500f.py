import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Reciprocals as Fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Suffix sums: suffix[i] = sum of recips[i:]
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips[i]
    
    # Greedy initial solution: add in order while sum < 1
    best_indices = []
    best_sum = Fraction(0)
    cur = Fraction(0)
    for i in range(n):
        if cur + recips[i] < 1:
            cur += recips[i]
            best_indices.append(i)
            best_sum = cur
    best_deficit = 1 - best_sum
    
    # Branch and Bound
    def bnb(i, current_sum, current_indices):
        nonlocal best_deficit, best_indices, best_sum
        
        if time.time() > DEADLINE:
            return
        
        # Pruning 1: If even taking all remaining can't beat best_deficit
        # Max possible sum = current_sum + suffix[i]
        if current_sum + suffix[i] < 1 - best_deficit:
            return
            
        # If we've considered all items
        if i == n:
            if current_sum < 1:
                deficit = 1 - current_sum
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = list(current_indices)
                    best_sum = current_sum
            return
        
        # Option 1: Skip item i
        bnb(i + 1, current_sum, current_indices)
        
        # Option 2: Take item i (if it doesn't make sum >= 1)
        new_sum = current_sum + recips[i]
        if new_sum < 1:
            current_indices.append(i)
            bnb(i + 1, new_sum, current_indices)
            current_indices.pop()
        
        # Additional pruning: If current_sum is already very close to 1, 
        # and adding any single item would exceed or not improve, we might stop?
        # But B&B handles this via the suffix check mostly.
        
    bnb(0, Fraction(0), [])
    
    return best_indices
