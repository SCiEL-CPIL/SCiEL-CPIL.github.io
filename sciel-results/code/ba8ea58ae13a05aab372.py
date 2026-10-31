import math
import time

# Cache to speed up repeated calls within a single evaluation (if any)
# We'll use a module-level dict to store computed priorities.
_priority_cache = {}

def priority(vector, n):
    # Deadline guard: track elapsed time and return a safe default if needed
    # This is a heuristic; the actual deadline enforcement is done by the evaluator.
    # We'll just ensure we don't do anything expensive.
    
    # Check cache
    key = (vector, n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    # Count occurrences of 0, 1, 2
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Ideal count per symbol for balanced vector
    ideal = n / 3.0
    
    # Deviation from ideal balance (L1 distance)
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # We want to MINIMIZE deviation, so negate for "higher is better"
    # Scale to dominate tie-breakers
    balance_score = -deviation * 10000.0
    
    # Deterministic polynomial hash for tie-breaking
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    
    # Map hash to small range [0, 1) for tie-breaking
    hash_score = h / (10**9 + 7)
    
    result = float(balance_score + hash_score)
    
    # Store in cache
    _priority_cache[key] = result
    
    return result
