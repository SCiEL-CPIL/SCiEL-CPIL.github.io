import time

def generate_smooth_numbers(primes, limit):
    """
    Generates all numbers <= limit whose prime factors are all in 'primes'.
    """
    smooths = {1}
    for p in primes:
        new_smooths = set()
        for x in smooths:
            y = x
            while y * p <= limit:
                y *= p
                new_smooths.add(y)
        smooths.update(new_smooths)
    return smooths

def construct():
    N = 123457
    start_time = time.time()
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We will check increasing sets of primes
    # S_k contains numbers with prime factors in primes[0:k+1]
    
    # Optimization: Pre-compute smooth numbers for the largest set we might need?
    # Or just iterate.
    
    # The knowledge says 11 is likely. Let's start from small sets.
    
    current_smooths = {1}
    
    for i, p in enumerate(primes):
        # Check deadline
        if time.time() - start_time > 80:
            break
            
        # Update smooths to include p
        # This is effectively computing the set for primes[0:i+1]
        new_vals = set()
        for x in current_smooths:
            y = x
            while y * p <= N:
                y *= p
                if y <= N:
                    new_vals.add(y)
                else:
                    break
        current_smooths.update(new_vals)
        
        # Check for pair a, b in current_smooths such that a+b=N
        # We can iterate over a in current_smooths
        # If N-a is in current_smooths, we found one.
        
        # To be deterministic and perhaps find smaller a? 
        # The problem doesn't specify which pair to return if multiple exist with same max P.
        # "Return two positive integers a,b".
        # Any valid pair with minimal max P is fine.
        
        found = False
        for a in current_smooths:
            if time.time() - start_time > 85:
                return [1, N-1] # Fallback, though unlikely to be reached if 11-smooth works
            
            b = N - a
            if b in current_smooths:
                # Found a pair
                # Verify P(a) and P(b) are <= p?
                # Yes, because they are in the smooth set for primes up to p.
                # So max(P(a), P(b)) <= p.
                # And since we didn't find it in previous steps (smaller primes), 
                # it's likely the max is exactly p (unless a=1 or b=1, but 1 is in all sets).
                # If a=1, b=N. N must be in current_smooths.
                # If N is in current_smooths, then P(N) <= p.
                # So this is correct.
                return [a, b]
                
        # If not found, continue to next prime

    # If no pair found with these primes, fallback
    # This should not happen for N=123457 within the first few primes.
    # But just in case, return a valid decomposition.
    # We can try to find any decomposition with small prime factors using a simple search
    # or just return [1, N-1] which is valid but has high score.
    # Given the knowledge, we expect to find a solution early.
    
    # As a last resort, use a simple factorization check for a few candidates
    # But [1, N-1] is a safe bet for validity.
    return [1, N-1]
