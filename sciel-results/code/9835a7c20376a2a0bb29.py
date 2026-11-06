"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter to only divisors <= target
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)  # Sort descending for better pruning
    
    start_time = time.time()
    deadline = start_time + 45  # Safety margin before 60s limit
    
    # Try increasing subset sizes k = 1, 2, 3, ...
    # For each k, search for a subset of size k that sums to target
    num_divs = len(divisors)
    
    for k in range(1, min(num_divs, 256) + 1):
        if time.time() > deadline:
            break
            
        # Generate combinations of size k
        # Use pruning: if the sum of the k largest remaining divisors is less than target, break
        # We iterate through combinations in lexicographic order (on indices)
        
        found = None
        for combo in combinations(range(num_divs), k):
            if time.time() > deadline:
                break
                
            # Calculate sum
            s = sum(divisors[i] for i in combo)
            if s == target:
                found = [divisors[i] for i in combo]
                break
            # Pruning: if we've gone too far in the lexicographic order,
            # and the current sum is already too high or too low, we can potentially break,
            # but it's complex. We rely on the fact that k is small.
            
        if found is not None:
            return found
    
    # Fallback: try a greedy approach if search didn't find solution in time
    remainder = target
    chosen = []
    for d in divisors:  # Already sorted descending
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
            
    if remainder == 0:
        return chosen
    else:
        # Last resort: return empty (should not happen for this instance)
        return []

# EVOLVE-BLOCK-END
