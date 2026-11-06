"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 1000000
    LO = 10000
    
    # powerful_part[i] will hold the product of p^e for each prime power p^e (e>=2) dividing i
    # Initialize to 1 for all
    pp = np.ones(N + 1, dtype=np.int64)
    
    # Sieve: for each prime p, compute p^2, p^3, ... and mark multiples
    # For a number n = p^e * m where p does not divide m, the powerful part contribution is p^e
    # We need to find the highest power of p dividing n, and if e >= 2, multiply pp[n] by p^e
    
    # Efficient approach:
    # For each prime p, for each power q = p^2, p^3, ...:
    #   For each multiple m of q:
    #     Determine the exact exponent of p in m
    #     If exponent >= 2, multiply pp[m] by p^(exponent)
    
    # Better approach using the sieve structure:
    # For each prime p:
    #   p2 = p*p
    #   For each multiple m of p2:
    #     Count how many times p divides m (i.e., find e such that p^e | m but p^(e+1) ∤ m)
    #     If e >= 2, multiply pp[m] by p^e
    
    # Even better: use the fact that if p^e || m (exact power), we can process by iterating
    # through powers of p.
    
    # Let's use a more direct sieve:
    # For each prime p, let p2 = p*p.
    # For k from p2 to N step p:
    #   if k is divisible by p but we want the FULL power...
    
    # Most efficient: for each prime p, compute all powers p^2, p^3, ..., p^j <= N
    # For each such power q = p^i (i >= 2):
    #   For each multiple m of q that is NOT a multiple of p*q (i.e., p^(i+1) ∤ m):
    #     pp[m] *= p^i
    #   Wait, that's not quite right either.
    
    # Let me think differently. For each number n, its powerful part is the product of p^e 
    # where p^e || n and e >= 2.
    # 
    # Sieve approach:
    # pp = ones(N+1)
    # For each prime p:
    #   p2 = p*p
    #   For m in range(p2, N+1, p):
    #     # m is divisible by p
    #     # Count the exact power of p in m
    #     temp = m
    #     e = 0
    #     while temp % p == 0:
    #       temp //= p
    #       e += 1
    #     if e >= 2:
    #       pp[m] *= p**e
    # 
    # But this is O(N log log N * avg_factor_count) which might be slow in Python.
    # Let's use NumPy vectorization where possible.
    
    # Alternative vectorized approach:
    # For each prime p:
    #   For each power q = p^2, p^3, ... <= N:
    #     indices = multiples of q
    #     For those indices, check if they are NOT divisible by p*q (to get exact power)
    #     Actually, we want: for m divisible by p^i but not p^(i+1), multiply pp[m] by p^i
    #     This is equivalent to: for m divisible by p^i, multiply pp[m] by p, 
    #     but only if i >= 2... no that's not right either.
    
    # Let me reconsider. The cleanest approach:
    # For each prime p:
    #   p2 = p*p
    #   For m = p2, 2*p2, 3*p2, ... <= N:
    #     e = 0
    #     temp = m
    #     while temp % p == 0:
    #       temp //= p
    #       e += 1
    #     if e >= 2:
    #       pp[m] *= p**e
    #
    # This is correct but potentially slow. Let's estimate:
    # Sum over primes p of N/p^2 ≈ N * sum(1/p^2) ≈ N * 0.45 ≈ 450,000 iterations for the outer loop
    # Each iteration does a while loop of length ~log_p(m), average maybe 2-3.
    # Total ~1-2 million operations, should be fine in Python within 85s.
    
    # First, get all primes up to N using a simple sieve
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # Now compute powerful parts
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For each multiple of p2
        for m in range(p2, N + 1, p2):
            # Count exact power of p in m
            temp = m
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                pp[m] *= (p ** e)
        
        # Time check
        if time.time() > deadline - 5:
            break
    
    if time.time() > deadline:
        return 10000
    
    # Compute product of consecutive powerful parts
    # prod[i] = pp[i] * pp[i+1] for i in [LO, N-1]
    prod = pp[LO:N] * pp[LO+1:N+1]
    
    # Find the index of the maximum product
    max_idx = np.argmax(prod)
    best_n = LO + max_idx
    
    return int(best_n)

# EVOLVE-BLOCK-END
