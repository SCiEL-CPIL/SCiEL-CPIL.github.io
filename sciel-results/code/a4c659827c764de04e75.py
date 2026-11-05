"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

TARGET = 123457

def linear_sieve(n):
    """Generate all primes up to n using linear sieve."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes

def generate_smooth_numbers(primes, limit):
    """Generate all numbers composed only of the given primes, up to limit.
    Returns a sorted list of such numbers."""
    # Use a set to avoid duplicates, then sort
    smooth = set()
    smooth.add(1)
    changed = True
    # Iteratively multiply by primes
    while changed:
        changed = False
        new_vals = set()
        for s in smooth:
            for p in primes:
                v = s * p
                if v <= limit:
                    if v not in smooth:
                        new_vals.add(v)
                        changed = True
                else:
                    break
        smooth.update(new_vals)
    return sorted(smooth)

def construct():
    # Precompute all primes up to TARGET
    primes = linear_sieve(TARGET)
    
    # Binary search on the index in primes for the smallest K (prime)
    # such that there exist two K-smooth numbers summing to TARGET
    lo, hi = 0, len(primes) - 1
    best_pair = None
    best_K = None
    
    # We need to find the smallest prime K such that the set of K-smooth
    # numbers contains a pair summing to TARGET.
    # For each candidate K (primes[0..mid]), generate smooth numbers and check.
    
    # To make this efficient, we'll binary search on the prime index.
    # But generating smooth numbers for each mid is expensive.
    # Alternative: just iterate from small primes upward until we find a valid pair.
    # Since the optimal K is likely small, this should be fast.
    
    # Let's try incrementally adding primes and checking
    # Start with just prime 2
    current_primes = []
    smooth_set = {1}
    
    for idx, p in enumerate(primes):
        current_primes.append(p)
        # Update smooth numbers by multiplying existing smooth numbers by p
        changed = True
        while changed:
            changed = False
            new_vals = set()
            for s in list(smooth_set):
                v = s * p
                if v <= TARGET:
                    if v not in smooth_set:
                        new_vals.add(v)
                        changed = True
                # Also need to handle powers: s * p^k
                # Actually the while loop with updating smooth_set handles this
            smooth_set.update(new_vals)
        
        # Check if any two numbers in smooth_set sum to TARGET
        # For each a in smooth_set, check if TARGET - a is in smooth_set
        found = False
        for a in smooth_set:
            b = TARGET - a
            if b in smooth_set and a <= b:  # a <= b to avoid duplicates, but we just need any pair
                # We found a valid pair
                return [a, b]
            if b in smooth_set:
                # Double check: both should be positive
                if a >= 1 and b >= 1:
                    return [a, b]
    
    # Fallback: should not reach here for this target
    return [1, TARGET - 1]

# EVOLVE-BLOCK-END
