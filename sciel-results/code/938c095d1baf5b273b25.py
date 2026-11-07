"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Strategy: 
    # 1. Generate a set of candidate k values that are likely to produce collisions in k*sigma(k).
    #    Numbers with many small prime factors (highly abundant/smooth numbers) tend to have 
    #    large sigma(k)/k ratios. Collisions are more likely among numbers with similar "shape"
    #    in terms of prime factorization.
    # 2. Compute M = k * sigma(k) for these candidates.
    # 3. Group M values. Return the M with the most distinct k's.
    
    # Candidate generation:
    # We will generate numbers of the form 2^a * 3^b * 5^c * 7^d * 11^e * 13^f ...
    # To keep k <= sqrt(M) and M <= 10^18, we need k * sigma(k) <= 10^18.
    # sigma(k) >= k, so k^2 <= 10^18 => k <= 10^9.
    # Actually, sigma(k) can be much larger than k. For highly abundant numbers, sigma(k)/k ~ log(log(k)).
    # Let's restrict k to be composed of the first few primes to maximize the chance of collisions 
    # in the multiplicative structure, or simply to have a dense set of candidates in the "smooth" region.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # We'll do a breadth-first or depth-first search over exponents.
    # A simple approach: iterate through combinations of exponents for the first few primes.
    # To ensure we cover a good range and find collisions, we'll generate numbers up to a certain limit.
    
    candidates = set()
    
    # Add some known good candidates from the seed and small multiples
    # 336 = 2^4 * 3 * 7
    # 372 = 2^2 * 3 * 31
    # 434 = 2 * 7 * 31
    
    # Let's generate all numbers of the form 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # with a,b,c,d,e,f >= 0 such that the number is not too large.
    # If k is around 10^6, sigma(k) is around 10^7, M is 10^13.
    # If k is around 10^8, sigma(k) is around 10^9, M is 10^17.
    # We want M <= 10^18.
    
    # We'll use a recursive generator to build smooth numbers.
    def gen_smooth(primes_list, max_val, current=1, idx=0):
        if time.time() > deadline:
            return
        if idx == len(primes_list):
            if current > 1:
                candidates.add(current)
            return
        p = primes_list[idx]
        val = current
        while val <= max_val:
            gen_smooth(primes_list, max_val, val, idx + 1)
            if val > max_val // p:
                break
            val *= p

    # We need to choose max_val carefully. 
    # If max_val is too high, the number of candidates explodes.
    # Let's start with a moderate limit for the "core" smooth numbers.
    # We can also include numbers with larger primes as long as the total product is small.
    
    # Let's try generating 7-smooth numbers (primes 2,3,5,7) up to 10^7
    # and 13-smooth numbers up to 10^5
    # This might be too many or too few. Let's be smarter.
    
    # Alternative: Randomized search + Local optimization?
    # No, let's stick to structured generation but expand the prime set slightly.
    
    # Let's generate numbers composed of primes [2, 3, 5, 7, 11, 13]
    # We'll limit the product to 10^8 to keep the count manageable, but check M <= 10^18.
    # Actually, if k is small, M is small. Collisions are rare for small M.
    # We want M to be large enough to have many divisors/factors.
    
    # Let's use a priority queue or just iterate over exponents in a bounded box.
    
    # Box search for exponents of 2, 3, 5, 7
    # 2^a * 3^b * 5^c * 7^d
    # a in [0, 30], b in [0, 20], c in [0, 15], d in [0, 12]
    # This covers a lot of smooth numbers.
    
    # To increase the chance of collisions, we should look for k values that are "close" in value 
    # but have different factorizations? No, k*sigma(k) is monotonic-ish? 
    # No, it's not strictly monotonic.
    
    # Let's just generate a large set of smooth numbers and compute M.
    
    # We will generate numbers of the form 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # such that the number <= 10^8.
    
    # To avoid explosion, we'll iterate a, b, c, d and compute the max possible e, f.
    
    p_list = [2, 3, 5, 7, 11, 13]
    
    # We'll use a recursive function that tracks the current value and index.
    # To prevent too many calls, we'll limit the depth or the number of additions.
    
    candidates.clear()
    
    def add_candidate(k):
        if k > 1:
            candidates.add(k)
            
    # We'll generate all 13-smooth numbers up to 10^7.
    # Count estimate:
    # Number of 7-smooth numbers <= N is ~ (log N)^4 / (4! * prod log p)
    # For N=10^7, log N ~ 16.
    # 16^4 / 24 ~ 40000.
    # For 13-smooth, it's a bit more, maybe 100k-200k.
    # This is computationally feasible to compute sigma for.
    
    gen_smooth(p_list, 10**7)
    
    # Also add some numbers with larger primes to diversify?
    # Maybe just 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g ...
    # Let's add a second pass with primes up to 31, but limit the product to 10^6 to keep it fast.
    
    p_list_2 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    gen_smooth(p_list_2, 10**6)
    
    # Now compute M = k * sigma(k) for all candidates.
    # We need an efficient sigma function.
    # Since k <= 10^7, we can precompute smallest prime factor (SPF) up to 10^7.
    
    max_k = max(candidates) if candidates else 1
    # Sieve for SPF
    spf = list(range(max_k + 1))
    for i in range(2, int(math.isqrt(max_k)) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, max_k + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            power = 0
            sum_powers = 1
            while n % p == 0:
                n //= p
                power += 1
                sum_powers *= p
                sum_powers += 1
            # sum_powers is now 1 + p + ... + p^power
            result *= sum_powers
        return result

    # Map M to list of k's
    m_to_k = {}
    
    for k in candidates:
        if time.time() > deadline:
            break
        s = sigma(k)
        m = k * s
        if m <= 10**18:
            if m not in m_to_k:
                m_to_k[m] = []
            m_to_k[m].append(k)
    
    # Find the M with the most k's
    best_m = None
    best_k_list = []
    
    for m, ks in m_to_k.items():
        if len(ks) > len(best_k_list):
            best_m = m
            best_k_list = ks
            # Check if we have enough time to keep searching?
            # If we find a very good one, we might stop early?
            # But we want to maximize. Let's finish the loop if it's fast.
    
    if best_m is None:
        # Fallback to seed
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify and format
    # Ensure k's are distinct (they are, from set)
    # Ensure k <= isqrt(M)?
    # Wait, the constraint is k_i <= isqrt(M).
    # Let's check this constraint.
    # k * sigma(k) = M => sigma(k) = M/k.
    # Since sigma(k) >= k + 1 (for k>1), M/k > k => k^2 < M => k < sqrt(M).
    # So the condition k <= isqrt(M) is naturally satisfied for k > 1.
    # For k=1, sigma(1)=1, M=1. isqrt(1)=1. 1<=1. OK.
    
    # Sort k's for consistency
    best_k_list.sort()
    
    return {'M': best_m, 'k': best_k_list}

# EVOLVE-BLOCK-END
