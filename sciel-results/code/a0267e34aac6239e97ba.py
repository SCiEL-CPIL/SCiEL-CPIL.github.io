"""Rewritten program: search for long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_result = {'start': 1, 'length': 1}
    
    # Strategy: Use CRT to construct runs where each number has many small prime factors
    # This makes phi(n)/n small, so phi values stay small and distinct
    
    # Small primes to include in each number
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try different subsets of small primes and run lengths
    # For a run of length k starting at s, we want s+i to have many small prime factors
    
    # Approach 1: Construct via CRT
    # For run length k, find s such that s+i is divisible by many small primes
    # We can force divisibility by specific primes for specific positions
    
    def compute_phi(n):
        """Compute Euler's totient function for n."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    
    def verify_run(s, k):
        """Verify that phi(s), phi(s+1), ..., phi(s+k-1) are all distinct."""
        seen = set()
        for i in range(k):
            v = compute_phi(s + i)
            if v in seen:
                return False
            seen.add(v)
        return True
    
    # CRT-based construction
    # For a run of length k, assign each position a distinct set of small primes
    # to be divisible by, then solve CRT
    
    def crt_solve(residues, moduli):
        """Solve system of congruences x ≡ residues[i] (mod moduli[i]) using CRT."""
        M = 1
        for m in moduli:
            M *= m
        x = 0
        for r, m in zip(residues, moduli):
            Mi = M // m
            # Find inverse of Mi mod m
            inv = pow(Mi, -1, m)
            x += r * Mi * inv
        return x % M
    
    # Try various constructions
    # For run length k, we want each s+i to be divisible by at least some small primes
    
    # Method: For each position i in [0, k-1], choose a set of small primes
    # that s+i should be divisible by. Then use CRT to find s.
    
    # Simple approach: make s+i divisible by (i+1)-th smallest prime or similar
    # But better: make all s+i divisible by 2, 3, 5, 7, 11, 13 (common factors)
    # Then vary additional factors
    
    # Let's try: for run length k, make s+i divisible by 2*3*5*7*11*13 = 30030
    # for all i, plus additional primes specific to each position
    
    base_product = 1
    for p in [2, 3, 5, 7, 11, 13, 17, 19]:
        base_product *= p
    # base_product = 9699690
    
    # If s is divisible by base_product, then s+i is NOT necessarily divisible by base_product
    # We need a different approach.
    
    # Better approach: Use the fact that we can make s+i ≡ 0 (mod p) for specific p
    # For each small prime p, we can make at most one s+i ≡ 0 (mod p) in a short run
    # (unless the run is longer than p)
    
    # For a run of length k, for each prime p > k, we can make exactly one s+i ≡ 0 (mod p)
    # For primes p <= k, we can make multiple s+i ≡ 0 (mod p)
    
    # Strategy: For run length k, use primes up to some limit
    # Assign each position i a unique "signature" of primes it's divisible by
    
    # Let's try a simpler approach: just search in regions where numbers have many small factors
    
    # Precompute: for a range, count how many small prime factors each number has
    # Then find the longest run where all numbers have at least T small prime factors
    
    # But this requires scanning a large range...
    
    # Alternative: Use the CRT construction more carefully
    # For run length k, we want s+i to each have at least 6-8 small prime factors
    # Let's assign primes to positions
    
    # For position i (0-indexed), make s+i divisible by primes at indices i, i+k, i+2k, ...
    # No, that's not quite right either.
    
    # Let me try a direct approach:
    # 1. For various run lengths k (from large to small), try to construct s
    # 2. Use CRT to make s+i divisible by specific small primes
    # 3. Verify the totients are distinct
    
    # For a run of length k, let's use the first k primes (or a subset)
    # and assign them to positions
    
    # Actually, let's try the simplest effective approach:
    # Make s ≡ -i (mod p_i) for i = 0, 1, ..., k-1 where p_i are distinct primes
    # This makes s+i ≡ 0 (mod p_i)
    # Additionally, make all s+i divisible by 2, 3, 5, 7 (if possible)
    
    # For simplicity, let's just use distinct primes for each position
    # and add common small factors
    
    # Try run lengths from 200 down to 50
    for k in range(200, 49, -1):
        if time.time() > deadline - 5:
            break
        
        # Use primes for each position
        # We need k distinct primes, plus we want common small factors
        # Let's use primes starting from 2
        primes_needed = []
        p = 2
        while len(primes_needed) < k:
            # Check if p is prime
            is_prime = True
            for q in primes_needed:
                if q * q > p:
                    break
                if p % q == 0:
                    is_prime = False
                    break
            if is_prime:
                primes_needed.append(p)
            p += 1
        
        # Try to make s+i ≡ 0 (mod primes_needed[i]) for each i
        # Also try to add common factors: make s ≡ 0 (mod 2), (mod 3), etc.
        # But s ≡ 0 (mod 2) and s+i ≡ 0 (mod primes_needed[i]) may conflict
        
        # Let's just use the distinct primes first
        residues = []
        moduli = []
        for i in range(k):
            residues.append(-i)  # s + i ≡ 0 (mod primes_needed[i])
            moduli.append(primes_needed[i])
        
        try:
            s = crt_solve(residues, moduli)
        except:
            continue
        
        # s might be 0, so ensure s >= 1
        if s < 1:
            M = 1
            for m in moduli:
                M *= m
            s += M
        
        # Check if s + k - 1 <= 10^12
        if s + k - 1 > 10**12:
            continue
        
        # Verify the run
        if verify_run(s, k):
            best_result = {'start': s, 'length': k}
            break  # Found a valid run of length k, which is good
    
    # If CRT construction didn't work well, try a search-based approach
    if best_result['length'] < 50:
        # Search in regions with many small prime factors
        # Focus on numbers that are multiples of 30030 = 2*3*5*7*11*13
        # or 9699690 = 2*3*5*7*11*13*17*19
        
        # Try around multiples of large primorials
        primorial = 9699690  # 2*3*5*7*11*13*17*19
        
        # Check runs around multiples of primorial
        # For a multiple m of primorial, m is divisible by all these primes
        # m+1, m+2, ... may not be, but some will have small factors
        
        # Let's check a window around each multiple
        window = 200
        max_multiple = min(10**12 // primorial, 10**6)  # Limit search
        
        for mult in range(1, min(max_multiple, 100000)):
            if time.time() > deadline - 5:
                break
            base = mult * primorial
            # Check run starting at base - 100 to base + 100
            for offset in range(-50, 51):
                s = base + offset
                if s < 1 or s + 200 - 1 > 10**12:
                    continue
                # Try to find the longest valid run starting at s
                seen = set()
                length = 0
                for i in range(200):
                    v = compute_phi(s + i)
                    if v in seen:
                        break
                    seen.add(v)
                    length += 1
                if length > best_result['length']:
                    best_result = {'start': s, 'length': length}
    
    return best_result
# EVOLVE-BLOCK-END
