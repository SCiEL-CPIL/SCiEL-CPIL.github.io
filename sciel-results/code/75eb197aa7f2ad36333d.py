"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute smallest prime factor for fast factorization
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def evaluate(arr):
        """Return the number of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)
    
    # Strategy: Use modular arithmetic to constrain sums
    # If all a_i ≡ r (mod m), then a_i + a_j ≡ 2r (mod m)
    # We want 2r ≡ 0 (mod m) so all sums are divisible by m
    # This means r ≡ 0 (mod m/gcd(2,m))
    
    # For m = 2: r ≡ 0 (mod 1) → any r works, sums even
    # For m = 6: r ≡ 0 (mod 3) → r ∈ {0,3,6,9,...} mod 6, sums ≡ 0 mod 6
    # For m = 30: r ≡ 0 (mod 15) → r ∈ {0,15,30,...} mod 30, sums ≡ 0 mod 30
    
    # Key insight: If we pick numbers in arithmetic progression with difference d,
    # and all ≡ r (mod m) where 2r ≡ 0 (mod m), then sums are ≡ 2r (mod m)
    # But sums also have structure: a_i + a_j = r + id + r + jd = 2r + (i+j)d
    # So sums are all ≡ 2r (mod gcd(d, m))
    
    # Better approach: Pick 12 numbers that are all ≡ r (mod m) for small m
    # where 2r ≡ 0 (mod m), and choose them to minimize additional prime factors
    
    # Let's try systematic search over modulus m and residue r
    best = None
    best_score = float('inf')
    
    # Try various moduli that are products of small primes
    small_primes = [2, 3, 5, 7, 11, 13]
    
    # Generate candidate moduli as products of subsets of small primes
    candidate_moduli = []
    for size in range(1, 4):  # products of 1, 2, or 3 small primes
        for combo in combinations(small_primes, size):
            m = 1
            for p in combo:
                m *= p
            if m <= 10000:  # must be able to fit 12 numbers in [1, 10000]
                candidate_moduli.append(m)
    
    # Also include some larger moduli
    candidate_moduli.extend([30, 42, 60, 66, 70, 78, 90, 102, 110, 114, 120, 130, 138, 150, 165, 174, 180, 190, 198, 210])
    candidate_moduli = list(set(candidate_moduli))
    
    for m in sorted(candidate_moduli):
        if time.time() > deadline - 10:
            break
        
        # Find residues r such that 2r ≡ 0 (mod m)
        # This means r ≡ 0 (mod m/gcd(2,m))
        g = math.gcd(2, m)
        step = m // g  # r must be multiple of step
        
        for r in range(0, m, step):
            if r == 0:
                r_actual = m  # use m instead of 0 to keep positive
            else:
                r_actual = r
            
            # Generate 12 distinct numbers ≡ r (mod m) in [1, 10000]
            # Numbers: r_actual, r_actual + m, r_actual + 2m, ...
            # Need at least 12 such numbers
            count = (10000 - r_actual) // m + 1 if r_actual <= 10000 else 0
            if count < 12:
                continue
            
            # Take first 12: r_actual, r_actual+m, ..., r_actual+11m
            candidate = [r_actual + k * m for k in range(12)]
            
            # Check all <= 10000
            if candidate[-1] > 10000:
                continue
            
            score = evaluate(candidate)
            if score < best_score:
                best_score = score
                best = candidate[:]
    
    # If no good candidate found, use fallback
    if best is None:
        best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
        best_score = evaluate(best)
    
    # Local search to improve
    random.seed(42)
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        no_improve_count = 0
        
        # Try each position
        for i in range(12):
            if time.time() > deadline - 5:
                break
            old_val = best[i]
            current_set = set(best)
            
            # Try various replacement strategies
            # Strategy 1: Try values with same residue mod small moduli
            # Strategy 2: Random values
            attempts = 0
            while attempts < 500 and time.time() < deadline - 5:
                attempts += 1
                
                # Try to pick a value that maintains good modular properties
                # Pick random modulus from candidates
                m_choice = random.choice([2, 3, 5, 6, 10, 15, 30])
                # Pick random residue
                r_choice = random.randint(0, m_choice - 1)
                # Find values ≡ r_choice (mod m_choice)
                min_v = max(1, r_choice - (r_choice % m_choice) + m_choice) if r_choice != 0 else m_choice
                # Actually, values ≡ r (mod m) are: r, r+m, r+2m, ...
                # If r == 0, values are m, 2m, 3m, ...
                if r_choice == 0:
                    candidates_r = [k * m_choice for k in range(1, 10000 // m_choice + 1)]
                else:
                    candidates_r = [r_choice + k * m_choice for k in range(0, (10000 - r_choice) // m_choice + 1) if r_choice + k * m_choice >= 1]
                
                if not candidates_r:
                    continue
                
                new_val = random.choice(candidates_r)
                
                if new_val in current_set:
                    continue
                
                candidate = best[:]
                candidate[i] = new_val
                score = evaluate(candidate)
                
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    no_improve_count = 0
                    break
                elif score == best_score:
                    # Accept neutral moves to explore
                    if random.random() < 0.3:
                        best = candidate
                        improved = True
                        no_improve_count = 0
                        break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 5:
                break
    
    return best
# EVOLVE-BLOCK-END
