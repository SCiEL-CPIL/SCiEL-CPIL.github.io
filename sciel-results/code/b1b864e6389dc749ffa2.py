import time
import random

START_TIME = time.time()
DEADLINE = 50.0  # Safety margin: evaluator kills at 60s

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def fast_validity_check(A):
    """
    Fast validity check: build cardinality-indexed sum bitsets and check
    for cross-cardinality collisions. Returns (valid, S) where S is the
    list of bitsets if valid, or (False, None) if invalid.
    """
    n = len(A)
    S = [0] * (n + 1)
    S[0] = 1  # Bit 0 set: sum 0 with 0 elements
    
    for x in A:
        # Update in reverse to avoid using x multiple times
        for k in range(n - 1, -1, -1):
            if S[k]:
                S[k+1] |= (S[k] << x)
    
    # Check for cross-cardinality collisions
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return (False, None)
    
    return (True, S)

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two
    subsets have different cardinalities, their sums differ.
    Uses a greedy approach with fast validity checks.
    """
    # Start with a known valid base: powers of 2 (guaranteed valid)
    # Powers of 2 have unique subset sums, so cardinality-separated sums are also unique
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Validate base
    valid, S = fast_validity_check(base)
    if not valid:
        # Fallback: return just [1]
        return [1]
    
    best = list(base)
    best_S = S
    
    all_values = set(range(1, 257))
    current_set = set(best)
    
    # Greedy: try to add elements one by one
    # Sort candidates by value (smaller values first might be easier to fit)
    candidates = sorted(all_values - current_set)
    
    for x in candidates:
        if is_done():
            break
        if len(best) >= 60:
            break
            
        new_set = best + [x]
        valid, new_S = fast_validity_check(new_set)
        if valid:
            best = new_set
            best_S = new_S
            current_set.add(x)
    
    # If we have time, try local search with swaps
    if not is_done() and len(best) < 60:
        # Try random swaps to improve
        start_ls = time.time()
        max_ls_time = min(10.0, DEADLINE - elapsed() - 2.0)
        
        if max_ls_time > 1.0:
            improved = True
            while improved and not is_done() and (time.time() - start_ls) < max_ls_time:
                improved = False
                # Try random swaps
                for _ in range(100):
                    if is_done() or (time.time() - start_ls) >= max_ls_time:
                        break
                    
                    # Pick a random element to remove
                    if not best:
                        break
                    idx_out = random.randint(0, len(best) - 1)
                    x_out = best[idx_out]
                    remaining = [x for i, x in enumerate(best) if i != idx_out]
                    
                    # Try to add a random element
                    available = list(all_values - set(remaining))
                    if not available:
                        break
                    
                    x_in = random.choice(available)
                    new_set = remaining + [x_in]
                    
                    valid, new_S = fast_validity_check(new_set)
                    if valid:
                        best = new_set
                        best_S = new_S
                        current_set = set(best)
                        improved = True
                        break
    
    # Final validation
    valid, _ = fast_validity_check(best)
    if valid:
        return list(best)
    
    # Fallback to base
    return list(base)

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
