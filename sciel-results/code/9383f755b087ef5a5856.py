"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0  # seconds
_start_time = time.time()

# Precompute smallest prime factor (SPF) up to a reasonable limit
# Weibull items with capacity 100 usually result in remainders < 100.
# We can handle larger remainders by falling back to trial division or just capping.
# Let's use a limit of 1000 to be safe for any weird cases, though capacity is 100.
MAX_N = 1000
spf = np.zeros(MAX_N + 1, dtype=np.int64)
for i in range(2, MAX_N + 1):
    if spf[i] == 0:
        spf[i] = i
        for j in range(i * i, MAX_N + 1, i):
            if spf[j] == 0:
                spf[j] = i

def count_prime_factors(n):
    """Count total number of prime factors with multiplicity (Omega(n))."""
    if n < 2:
        return 0
    count = 0
    while n > 1:
        if n <= MAX_N:
            p = spf[n]
            while n % p == 0:
                n //= p
                count += 1
        else:
            # Fallback for large n (should rarely happen given capacity 100)
            # Simple trial division for small primes
            d = 2
            while d * d <= n:
                while n % d == 0:
                    n //= d
                    count += 1
                d += 1
            if n > 1:
                count += 1
    return count

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety: fall back to simple best fit
        return -(bins - item)
    
    remaining = bins - item
    feasible = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(feasible):
        return scores
        
    feasible_indices = np.where(feasible)[0]
    feasible_remaining = remaining[feasible_indices]
    
    # The strategy: Minimize the number of prime factors of the remainder.
    # Hypothesis: Remainders with fewer prime factors (e.g., primes, 1, squares of primes)
    # are "simpler" numbers. While this is a heuristic based on number theory,
    # it serves as a structural alternative to standard fit-based heuristics.
    # 
    # Wait, the hypothesis in the prompt is: "remainders with fewer prime factors are more likely 
    # to be exactly filled by future integer-sized Weibull items".
    # Actually, if a remainder is a prime, it can only be filled by that prime or 1.
    # If a remainder is highly composite, it can be filled by many combinations.
    # The prompt's hypothesis might be counter-intuitive or specific to a certain interpretation.
    # Let's re-read carefully: "minimization of the total count of prime factors... 
    # leveraging the hypothesis that remainders with fewer prime factors are more likely 
    # to be exactly filled by future integer-sized Weibull items".
    # 
    # This implies we should PREFER remainders with FEWER prime factors.
    # So, score = -count_prime_factors(remainder).
    # Higher score (less negative) is better.
    
    # We need to handle float remainders. The prompt mentions "integer-sized Weibull items".
    # Weibull distribution generates floats. The "integer-sized" might refer to the 
    # nature of the items in the specific test generation or a simplification.
    # However, `item` and `bins` are floats. `remaining` is float.
    # We should round the remaining to the nearest integer for the prime factorization,
    # as prime factorization is defined for integers.
    
    int_remaining = np.round(feasible_remaining).astype(np.int64)
    
    # Clip to valid range for SPF lookup
    int_remaining_clipped = np.clip(int_remaining, 0, MAX_N)
    
    # Vectorized calculation of Omega(n) is tricky with variable loop lengths.
    # Given N=1000, a simple loop in Python might be slow if called 1000 times per item.
    # But we only have 1000 items total in the sequence. 1000 * 1000 = 1,000,000 operations.
    # In Python, this might be too slow (seconds).
    # We need a faster way or a precomputed lookup table for Omega(n) up to MAX_N.
    
    # Let's precompute Omega(n) for all n in [0, MAX_N]
    # This is a one-time cost.
    
    # We will compute this on the fly if not done, but since we can't easily store state 
    # in the function without global vars, let's use a global cache.
    
    global _omega_cache
    try:
        _omega_cache
    except NameError:
        _omega_cache = np.zeros(MAX_N + 1, dtype=np.int64)
        for i in range(2, MAX_N + 1):
            if spf[i] == i: # i is prime
                for j in range(i, MAX_N + 1, i):
                    # Count how many times i divides j
                    temp = j
                    while temp % i == 0:
                        _omega_cache[j] += 1
                        temp //= i
        # _omega_cache[0] and _omega_cache[1] are 0 by default, which is correct.
        
    omega_counts = _omega_cache[int_remaining_clipped]
    
    # Score: We want to minimize omega_counts.
    # So score = -omega_counts.
    # To break ties and ensure some basic packing logic (like not wasting huge space 
    # if two remainders have same omega), we might need a tie-breaker.
    # However, the directive says "Make a STRUCTURAL change... different design".
    # Let's stick to the pure number-theoretic score first, perhaps with a tiny 
    # best-fit tie-breaker to avoid pathological cases where a huge bin with 
    # remainder 97 (prime, omega=1) is chosen over a small bin with remainder 1 (omega=0).
    # Wait, remainder 1 has omega 0. Remainder 97 has omega 1. 
    # So remainder 1 is preferred. That's good.
    # What about remainder 100 (2^2 * 5^2, omega=4) vs remainder 99 (3^2 * 11, omega=3)?
    # 99 is preferred.
    
    # Let's add a very small best-fit component to stabilize the choice when omega counts are equal.
    # This ensures we don't arbitrarily pick a huge bin if a smaller bin has the same "arithmetic simplicity".
    tie_breaker = -feasible_remaining / 10000.0
    
    scores[feasible_indices] = -omega_counts.astype(np.float64) + tie_breaker
    
    return scores
# EVOLVE-BLOCK-END
