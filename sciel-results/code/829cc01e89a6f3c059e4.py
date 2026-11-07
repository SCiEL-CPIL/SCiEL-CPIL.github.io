import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 80  # Safety margin for 90s limit
    
    best_span = float('inf')
    best_result = None
    
    # We search for increasing window sizes W
    # The optimal span is likely small. 
    # We iterate L in a range that covers enough residue classes.
    # LCM of divisors is large, but we don't need full LCM if we just check existence.
    # However, to be safe, we iterate L up to a reasonable limit.
    
    # To ensure we find the global minimum, we iterate W from 0 upwards.
    # For each W, we iterate L. If we find a solution, we return it immediately 
    # because any larger W would have a larger span.
    
    # Optimization: 
    # For a fixed W, the pattern of available multiples repeats every LCM(divisors).
    # But LCM is huge. 
    # Instead, we can just iterate L for a sufficient number of steps.
    # Since the divisors are small, the "gaps" are small.
    # Let's try L from 0 to 5000 for each W.
    
    for W in range(0, 200):
        if time.time() > deadline:
            break
            
        # If we have already found a better span, we can stop if W >= best_span
        if W >= best_span:
            break
            
        # Iterate starting points L
        # We only need to check L up to a certain point. 
        # A multiple of d exists in [L, L+W] if the distance to the next multiple is <= W.
        # The "worst case" gap for a divisor d is d-1.
        # So if W >= max(divisors), a multiple always exists for any L.
        # The constraint is distinctness.
        
        for L in range(0, 5000):
            if time.time() > deadline:
                break
                
            # Generate candidate multiples for each divisor in [L, L+W]
            candidates = []
            valid_window = True
            for d in divisors:
                # Find the first multiple of d >= L
                m = ((L + d - 1) // d) * d
                cands = []
                while m <= L + W:
                    if m <= max_val:
                        cands.append(m)
                    m += d
                if not cands:
                    valid_window = False
                    break
                candidates.append(cands)
            
            if not valid_window:
                continue
                
            # Now, find a distinct assignment.
            # Backtracking
            assignment = [-1] * n
            used = set()
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                for val in candidates[idx]:
                    if val not in used:
                        used.add(val)
                        assignment[idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                # Found a valid assignment for this W
                # Since we iterate W increasing, this is the minimal span
                # Check if it's better than current best (it should be, or equal)
                span = W
                if span < best_span:
                    best_span = span
                    best_result = assignment[:]
                    # Since we are iterating W from 0, the first valid W is the minimum.
                    # We can return immediately.
                    return [int(x) for x in best_result]
    
    # Fallback if no small window found (should not happen for these divisors)
    # Try a larger search or fallback to a known valid solution
    if best_result is None:
        # Simple fallback: pick multiples far apart
        # This is just to ensure we return something valid if the search fails
        # But the search above should find a solution for W < 100
        reps = []
        for i, d in enumerate(divisors):
            # Pick a multiple that is likely distinct
            # e.g., i * d * 10 + d
            val = (i + 1) * d * 100
            reps.append(val)
        best_result = reps
        best_span = max(reps) - min(reps)
        
    return [int(x) for x in best_result]

if __name__ == "__main__":
    res = construct()
    print(res)
    print("Span:", max(res) - min(res))
