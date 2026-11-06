"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Generate powerful numbers up to a limit L
    # A powerful number is one where every prime factor appears with exponent >= 2
    # 1 is powerful by convention
    
    # Strategy: 
    # 1. Generate powerful numbers up to some limit L
    # 2. Compute all pairwise sums a+b where a <= b and a+b <= 10^18
    # 3. Count frequencies to find N with most representations
    # 4. Return N and its pairs
    
    # First, generate powerful numbers up to L
    # A powerful number has the form p1^e1 * p2^e2 * ... where each ei >= 2
    # We can generate them by iterating over possible "squarefree kernels" and raising to powers
    
    def generate_powerful(limit):
        """Generate all powerful numbers <= limit."""
        powerful = set([1])
        
        # For each possible squarefree part s (product of distinct primes),
        # and for each exponent pattern, we get s^2 * k^2 where k is any integer
        # Actually, powerful numbers are of the form a^2 * b^3 where b is squarefree
        # Wait, that's not quite right either.
        
        # Better approach: powerful number n = product of p_i^(e_i) where e_i >= 2
        # We can generate by considering that n = m^2 * r where r is squarefree
        # and each prime in r appears to an odd power >= 3 in n
        # Actually: n is powerful iff n = a^2 * b^3 where b is squarefree
        # Let me verify: if n = product p_i^(e_i) with e_i >= 2
        # Write e_i = 2*f_i + g_i where g_i in {0, 1}
        # Then n = (product p_i^(f_i))^2 * (product p_i^(g_i))^3
        # The second factor is a cube of a squarefree number
        # So yes, n = a^2 * b^3 where b is squarefree
        
        # Generate all b (squarefree) such that b^3 <= limit, i.e., b <= limit^(1/3)
        b_max = int(limit ** (1/3)) + 1
        
        # For each squarefree b, generate a such that a^2 * b^3 <= limit
        # i.e., a <= sqrt(limit / b^3)
        
        # First, find all squarefree numbers up to b_max
        # Use sieve to find primes, then generate squarefree numbers
        
        # Sieve of Eratosthenes up to b_max
        if b_max < 2:
            squarefree = [1]
        else:
            is_prime = [True] * (b_max + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, isqrt(b_max) + 1):
                if is_prime[i]:
                    for j in range(i*i, b_max + 1, i):
                        is_prime[j] = False
            
            # Generate squarefree numbers using inclusion-exclusion or direct enumeration
            # A number is squarefree if it's not divisible by p^2 for any prime p
            squarefree = []
            for n in range(1, b_max + 1):
                is_sqfree = True
                temp = n
                for p in range(2, isqrt(n) + 1):
                    if temp % (p*p) == 0:
                        is_sqfree = False
                        break
                    while temp % p == 0:
                        temp //= p
                if is_sqfree:
                    squarefree.append(n)
        
        # For each squarefree b, generate a values
        for b in squarefree:
            b_cubed = b ** 3
            if b_cubed > limit:
                continue
            a_max = isqrt(limit // b_cubed)
            for a in range(1, a_max + 1):
                val = a * a * b_cubed
                if val <= limit:
                    powerful.add(val)
        
        return sorted(powerful)
    
    # Start with a reasonable limit and increase if time permits
    # Let's start with L = 10^7 which should give ~8K powerful numbers
    L = 10000000
    
    powerful = generate_powerful(L)
    powerful_set = set(powerful)
    
    # Now compute all pairwise sums and count them
    # For a <= b, a + b = N, both powerful
    # We want to find N with maximum count
    
    # Number of powerful numbers is ~8K, so ~34M pairs - might be too slow in pure Python
    # Let's use a smarter approach or increase L carefully
    
    # Actually, let's try with the current set and see how long it takes
    # If too slow, we'll use a smaller L
    
    n_pow = len(powerful)
    
    # Count sums using a dictionary
    sum_counts = Counter()
    
    # To speed up, we can use the fact that we only need a <= b
    # and a + b <= 10^18 (which is always true since a, b <= L = 10^7)
    
    # But 8K^2 / 2 = 32M iterations might be slow in Python
    # Let's try and check time
    
    for i in range(n_pow):
        a = powerful[i]
        for j in range(i, n_pow):
            b = powerful[j]
            s = a + b
            if s > 10**18:
                break
            sum_counts[s] += 1
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 100:
        # We're running out of time, return best found so far
        pass
    
    if sum_counts:
        best_N = max(sum_counts, key=sum_counts.get)
        best_count = sum_counts[best_N]
        
        # Collect pairs for best_N
        pairs = []
        for a in powerful:
            b = best_N - a
            if a <= b and b in powerful_set:
                pairs.append([a, b])
        
        return {'N': best_N, 'pairs': pairs}
    
    # Fallback: return a simple case
    return {'N': 2, 'pairs': [[1, 1]]}
# EVOLVE-BLOCK-END
