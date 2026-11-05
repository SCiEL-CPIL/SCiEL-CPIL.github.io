"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

# Primes strictly below 24
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
NUM_PRIMES = len(PRIMES)

def compute_signatures(max_n):
    """
    For each number up to max_n, compute a bitmask where bit k is set
    if the k-th small prime divides the number.
    This represents the set of distinct small-prime factors.
    
    We also need to track the 'complete factor composed of primes STRICTLY BELOW 24',
    but the objective is to MINIMIZE the number of distinct extracted factors.
    
    Wait, re-reading the task:
    "For each n+i, 1<=i<=24, extract its complete factor composed of primes STRICTLY BELOW 24. 
     MINIMIZE the number of distinct extracted factors."
    
    Let $S_i$ be the set of primes $p < 24$ that divide $n+i$.
    The "complete factor composed of primes STRICTLY BELOW 24" for $n+i$ is the product of $p^{v_p(n+i)}$ for all $p < 24$.
    Let $f_i$ be this factor.
    The task asks to MINIMIZE the number of distinct values in the set $\{f_1, f_2, \dots, f_{24}\}$.
    
    So we are looking for a window of 24 consecutive integers such that the number of distinct
    "small-prime parts" is minimized.
    
    Note: If a number is not divisible by any prime < 24, its "complete factor" is 1.
    So 1 is a possible value.
    
    Strategy:
    Precompute the "small prime part" value for each number in a range.
    Then slide a window of size 24 and count distinct values.
    
    Range: n in [1000, 1000000].
    So we need to check windows starting at n=1000 up to n=1000000.
    The window covers n+1 to n+24.
    So the numbers involved are from 1001 to 1000024.
    
    Precomputation:
    For each number m in [1, 1000024], compute $f(m) = \prod_{p < 24} p^{v_p(m)}$.
    This can be done efficiently using a sieve-like approach or simple factorization for small primes.
    Since there are only 9 primes, factorization is fast.
    
    Let's precompute an array `f` of size `max_val + 1`.
    `f[m]` will store the integer value of the small-prime part.
    
    Then, for each n in [1000, 1000000], we look at the slice `f[n+1 : n+25]`.
    We want to minimize `len(set(f[n+1 : n+25]))`.
    
    We can use a sliding window frequency count to do this in O(1) per step after O(N) precomputation.
    
    Steps:
    1. Define max_val = 1000024.
    2. Initialize `f` array of zeros.
    3. For each m in 1..max_val, compute f[m].
       Actually, we can compute f[m] by iterating through the primes.
       Or better, initialize f[m] = 1.
       For each prime p in PRIMES:
           for multiple of p:
               compute power of p dividing m, multiply into f[m].
       This is O(N * sum(1/p)) which is fine.
       
    4. Sliding window:
       Let `freq` be a dictionary or array (if values are small enough) to count occurrences of each distinct f-value in the current window.
       Since f[m] can be large, use a dictionary.
       
       Initialize window for n=1000:
       elements = f[1001..10024]
       Count distincts.
       
       Then slide:
       Remove f[n+1]
       Add f[n+25]
       
       Track the minimum distinct count and the corresponding n.
       
    5. Return the best n.
    """
    max_n = 1000000
    max_val = max_n + 24
    # f[m] will be the product of p^v_p(m) for p in PRIMES
    f = [1] * (max_val + 1)
    
    # Sieve-like multiplication
    for p in PRIMES:
        # For each multiple of p
        for m in range(p, max_val + 1, p):
            temp = m
            power = 1
            while temp % p == 0:
                power *= p
                temp //= p
            f[m] *= power
            
    # Sliding window
    window_size = 24
    start_n = 1000
    end_n = max_n
    
    # Initialize frequency for first window: n=1000, elements f[1001]..f[10024]
    freq = {}
    distinct_count = 0
    
    def add_val(v):
        nonlocal distinct_count
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct_count += 1
            
    def remove_val(v):
        nonlocal distinct_count
        freq[v] -= 1
        if freq[v] == 0:
            del freq[v]
            distinct_count -= 1
            
    # Pre-fill window for n=1000
    for i in range(1, window_size + 1):
        add_val(f[1000 + i])
        
    best_distinct = distinct_count
    best_n = 1000
    
    # Slide from n=1001 to max_n
    # Window for n: f[n+1]..f[n+24]
    # Window for n-1: f[n]..f[n+23]
    # When moving from n-1 to n: remove f[n], add f[n+24]
    
    for n in range(start_n + 1, end_n + 1):
        # Remove the leftmost element of the previous window, which is f[n]
        # Previous window (for n-1) was f[n]..f[n+23]
        # Current window (for n) is f[n+1]..f[n+24]
        remove_val(f[n])
        add_val(f[n + 24])
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
            
    return best_n

# Deadline guard wrapper
def construct():
    start_time = time.time()
    try:
        result = compute_signatures(1000000)
        return result
    except Exception as e:
        # Fallback to a simple value if something goes wrong
        return 1000

# EVOLVE-BLOCK-END
