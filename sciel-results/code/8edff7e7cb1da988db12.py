"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations, product

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _all_pairwise_coprime(vals):
    vs = list(set(vals))
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _score_pair(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _all_pairwise_coprime(sums):
        return 0
    return len(sums)

def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 60s limit
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    primes = [n for n in range(2, 257) if _is_prime(n)]
    
    # Strategy: search for A, B where all 9 sums are distinct primes
    # This guarantees pairwise coprimality and max cardinality 9
    
    # Try: fix B, search for A
    # For efficiency, try small structured sets first
    
    best_score = 0
    best_pair = ([1, 7, 13], [4, 10, 16])  # fallback (score 1.0 equivalent)
    
    # Quick check: the seed gives sums {5,11,17,11,17,23,17,23,29} -> distinct {5,11,17,23,29}
    # That's 5 distinct, all primes -> pairwise coprime. Score = 5/5 = 1.0
    
    # Let's search more systematically
    # Try all pairs where A and B are 3-element subsets of [1,128]
    # But that's C(128,3)^2 which is huge. Need smarter approach.
    
    # Smarter: iterate over B (small sets), then for each B, try A values
    # Key constraint: for all a in A, b in B: a+b must be prime (for max score 9)
    
    # Even smarter: pick B first from small primes-related sets
    # If B = {b1, b2, b3}, then for a to work: a+b1, a+b2, a+b3 all prime
    # And the 9 sums a_i + b_j must all be distinct primes
    
    # Let's try B sets and find valid A
    # Limit B to reasonable range for speed
    b_range = list(range(1, 129))
    
    # To keep it fast, try structured B sets
    # B = {1, p, q} where p, q are small primes or composites
    
    # Actually, let's just do a targeted search:
    # For each candidate B (sampled), find A such that all sums are distinct primes
    
    found = False
    
    # Try B = {1, 2, 3} type small sets and larger structured ones
    # Key: a+b must be prime for all 9 pairs, and all 9 must be distinct
    
    # Let's iterate B over a curated set of candidates
    b_candidates = []
    # Small sets
    for b1 in range(1, 30):
        for b2 in range(b1 + 1, 30):
            for b3 in range(b2 + 1, 30):
                b_candidates.append([b1, b2, b3])
    
    # Also add some larger structured sets
    for b1 in [2, 4, 6, 8, 10, 12]:
        for b2 in range(b1 + 2, 60):
            for b3 in range(b2 + 2, 60):
                b_candidates.append([b1, b2, b3])
    
    for B in b_candidates:
        if time.time() > deadline:
            break
        b1, b2, b3 = B[0], B[1], B[2]
        
        # Find a values where a+b1, a+b2, a+b3 are all prime
        valid_a = []
        for a in range(1, 129):
            if _is_prime(a + b1) and _is_prime(a + b2) and _is_prime(a + b3):
                valid_a.append(a)
        
        if len(valid_a) < 3:
            continue
        
        # Try all combinations of 3 from valid_a
        for A in combinations(valid_a, 3):
            A = list(A)
            sums = set()
            all_prime = True
            for a in A:
                for b in B:
                    s = a + b
                    if not _is_prime(s):
                        all_prime = False
                        break
                    sums.add(s)
                if not all_prime:
                    break
            if all_prime and len(sums) == 9:
                return A, B
            elif all_prime and len(sums) > best_score:
                best_score = len(sums)
                best_pair = (A, B)
        
        # Also check non-prime pairwise coprime sums for this B
        for A in combinations(valid_a, 3):
            A = list(A)
            sc = _score_pair(A, B)
            if sc > best_score:
                best_score = sc
                best_pair = (A, B)
    
    if best_score >= 9:
        return best_pair
    
    # Broader search: not requiring all primes, just pairwise coprime distinct sums
    # Try random sampling
    import random
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline and attempts < 200000:
        attempts += 1
        A = rng.sample(range(1, 129), 3)
        B = rng.sample(range(1, 129), 3)
        sc = _score_pair(A, B)
        if sc > best_score:
            best_score = sc
            best_pair = (A, B)
            if best_score == 9:
                break
    
    return best_pair

# EVOLVE-BLOCK-END
