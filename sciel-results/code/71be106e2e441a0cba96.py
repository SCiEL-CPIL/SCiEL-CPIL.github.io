"""Rewritten program with smarter prime-based search."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    distinct = sorted(set(vals))
    n = len(distinct)
    if n <= 1:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _sumset(A, B):
    return set(a + b for a in A for b in B)

def _is_valid(A, B):
    if len(A) != 3 or len(set(A)) != 3:
        return False
    if len(B) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    sums = _sumset(A, B)
    return _pairwise_coprime(list(sums))

def _score(A, B):
    if not _is_valid(A, B):
        return 0
    return len(_sumset(A, B))

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    # Sieve primes up to 256 (max sum = 128+128=256)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(256)
    prime_set = set(primes)
    
    # Best known
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _score(best_A, best_B)
    
    # Key insight: For fixed A = [a1, a2, a3], valid b values are those where
    # a1+b, a2+b, a3+b are ALL prime. Precompute this for each A.
    # Then pick 3 distinct b's from this set and check all 9 sums are distinct primes.
    
    # Generate candidate A values. Focus on small values for efficiency.
    # A elements in [1, 128], but small values give more room for B.
    
    # Precompute: for each a in [1,128], the set of b in [1,128] such that a+b is prime
    # a+b in [2, 256], so b in [1,128] means a+b in [a+1, a+128]
    # We need a+b to be prime.
    
    # For efficiency, precompute for each a the set of valid b's
    valid_b_sets = {}
    for a in range(1, 129):
        valid_b_sets[a] = set(b for b in range(1, 129) if (a + b) in prime_set)
    
    # For a given A = [a1, a2, a3], valid B values are intersection of valid_b_for_a[a_i]
    # We want to find A where this intersection has at least 3 elements,
    # and among those 3, all 9 sums a_i+b_j are distinct primes.
    
    # Iterate over candidate A's. To be systematic, try all A with small elements first.
    # A elements: a1 < a2 < a3, each in [1, 128]
    # To keep it manageable, focus on a1, a2, a3 <= 60 initially, then expand.
    
    random.seed(42)
    
    # Phase 1: Systematic search over A with small elements
    # For each A, compute intersection of valid b's, then check triples
    max_a_val = 50
    
    for a1 in range(1, max_a_val + 1):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, max_a_val + 1):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, max_a_val + 1):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                
                # Intersection of valid b's
                common_b = valid_b_sets[a1] & valid_b_sets[a2] & valid_b_sets[a3]
                
                if len(common_b) < 3:
                    continue
                
                common_b_list = sorted(common_b)
                
                # Check all triples of b's from common_b_list
                # For each triple, check if all 9 sums are distinct primes
                # This is O(n^3) where n = len(common_b_list)
                # To speed up, we can check incrementally
                
                n_b = len(common_b_list)
                if n_b <= 3:
                    triples = [common_b_list]
                else:
                    triples = combinations(common_b_list, 3)
                
                found_better = False
                for b1, b2, b3 in triples:
                    B = [b1, b2, b3]
                    # Check all 9 sums are distinct primes
                    all_distinct_primes = True
                    seen = set()
                    for a in A:
                        for b in B:
                            s = a + b
                            if s in seen or s not in prime_set:
                                all_distinct_primes = False
                                break
                            seen.add(s)
                        if not all_distinct_primes:
                            break
                    if all_distinct_primes:
                        s = 9
                        if s > best_score:
                            best_score = s
                            best_A = A[:]
                            best_B = B[:]
                            found_better = True
                        break  # Found a valid B for this A, move to next A
                
                if found_better and best_score == 9:
                    # Can't do better than 9, return immediately
                    return [best_A, best_B]
    
    if best_score == 9:
        return [best_A, best_B]
    
    if time.time() > deadline:
        return [best_A, best_B]
    
    # Phase 2: Expand search to larger A values
    max_a_val2 = 128
    for a1 in range(1, max_a_val2 + 1):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, max_a_val2 + 1):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, max_a_val2 + 1):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                
                common_b = valid_b_sets[a1] & valid_b_sets[a2] & valid_b_sets[a3]
                
                if len(common_b) < 3:
                    continue
                
                common_b_list = sorted(common_b)
                n_b = len(common_b_list)
                
                # Limit triples to check for speed
                if n_b > 20:
                    common_b_list = common_b_list[:20]
                    n_b = 20
                
                triples = combinations(common_b_list, 3)
                
                found_better = False
                for b1, b2, b3 in triples:
                    B = [b1, b2, b3]
                    all_distinct_primes = True
                    seen = set()
                    for a in A:
                        for b in B:
                            s = a + b
                            if s in seen or s not in prime_set:
                                all_distinct_primes = False
                                break
                            seen.add(s)
                        if not all_distinct_primes:
                            break
                    if all_distinct_primes:
                        s = 9
                        if s > best_score:
                            best_score = s
                            best_A = A[:]
                            best_B = B[:]
                            found_better = True
                        break
                
                if found_better and best_score == 9:
                    return [best_A, best_B]
    
    if best_score == 9:
        return [best_A, best_B]
    
    if time.time() > deadline:
        return [best_A, best_B]
    
    # Phase 3: Try to find solutions with 8 distinct sums (some collisions but still pairwise coprime)
    # This is harder, use random search with local improvement
    random.seed(42)
    
    def random_set(size, low, high):
        return random.sample(range(low, high + 1), size)
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        A = random_set(3, 1, 80)
        B = random_set(3, 1, 80)
        s = _score(A, B)
        if s > best_score:
            best_score = s
            best_A = A[:]
            best_B = B[:]
            if best_score == 9:
                return [best_A, best_B]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
