import numpy as np

def construct():
    N = 10_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    best_error = float('inf')
    best_a, best_b = None, None
    
    # Iterate b from N down to 1
    # We can stop early if the minimum possible error for b is >= best_error
    # Min error for b:
    # if b is odd: 1/(2*b)
    # if b is even: 1/b (since a=1.5b is invalid)
    
    for b in range(N, 0, -1):
        # Calculate theoretical minimum error for this b
        if b % 2 == 1:
            min_err = 1.0 / (2.0 * b)
        else:
            min_err = 1.0 / b
            
        if min_err >= best_error:
            # Even if we find a solution, it won't be better than current best
            # But wait, is it possible that a solution with LARGER error is found first?
            # Yes, but we are iterating b downwards.
            # If min_err >= best_error, no solution for this b (or smaller b) can beat best_error?
            # No, smaller b has LARGER min_err.
            # So if min_err >= best_error, we can break?
            # Let's check:
            # b decreases -> min_err increases.
            # So if for current b, min_err >= best_error, then for all b' < b, min_err(b') > min_err(b) >= best_error.
            # So we can break.
            break
        
        # Determine candidates for a
        # a must be <= N and != 1.5b (if 1.5b is int)
        target = 1.5 * b
        
        candidates = []
        
        if b % 2 == 0:
            # 1.5 * b is integer
            a_exact = int(target)
            # a_exact is invalid (error 0)
            # Nearest valid a: a_exact - 1, a_exact + 1
            if a_exact - 1 >= 1:
                candidates.append(a_exact - 1)
            if a_exact + 1 <= N:
                candidates.append(a_exact + 1)
        else:
            # 1.5 * b is k + 0.5
            # Nearest integers: floor(target), ceil(target)
            a_floor = int(target)
            a_ceil = a_floor + 1
            if a_floor >= 1:
                candidates.append(a_floor)
            if a_ceil <= N:
                candidates.append(a_ceil)
        
        # Check candidates
        for a in candidates:
            if a == b:
                continue # Just in case, though ratio would be 1
            if sigma[a] == sigma[b]:
                # Calculate actual error
                err = abs(a / b - 1.5)
                if err < best_error:
                    best_error = err
                    best_a = int(a)
                    best_b = int(b)
                # Since we found a solution for this b, and min_err was the lower bound,
                # and for odd b the error is exactly min_err, for even b it is exactly min_err.
                # So we don't need to check other candidates for this b?
                # For odd b, both candidates have same error.
                # For even b, both candidates have same error.
                # So we can break candidate loop?
                break
        
    if best_a is None:
        # Fallback: return any valid pair, e.g. a=1, b=1? No, sigma(1)=1, sigma(1)=1.
        # But |1/1 - 1.5| = 0.5.
        # We need to find at least one pair.
        # If the loop didn't find one, it means no pair with error < initial inf?
        # This shouldn't happen as there are many pairs.
        # But if the break condition was too aggressive?
        # min_err starts at 1e-7. best_error is inf.
        # So it will check b=N, N-1, ... until it finds one.
        # If it finds one, best_error is updated.
        # Then it continues until min_err >= best_error.
        
        # If best_a is still None, something is wrong.
        # Let's just return a known pair if we must.
        # But the loop should find one.
        return [1, 1] # Invalid error 0.5? No, 1/1-1.5 = -0.5. Abs is 0.5.
        
    return [best_a, best_b]
