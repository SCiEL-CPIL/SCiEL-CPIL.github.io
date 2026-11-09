import random
import math
import time
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(sums):
    n = len(sums)
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(sums[i], sums[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return count of distinct sums if all distinct sums are pairwise coprime, else 0."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) == 0:
        return 0
    slist = list(sums)
    if _pairwise_coprime(slist):
        return len(slist)
    return 0

def construct():
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    best = [[1, 7, 13], [4, 10, 16]]
    best_score = 5  # seed score
    
    # Precompute: for each value in [1,128], its prime factors (for quick gcd checks)
    # Actually gcd is fast enough
    
    # Phase 1: Random search
    rng = random.Random(42)
    
    # Try many random pairs
    for _ in range(500000):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = _score(A, B)
        if s > best_score:
            best_score = s
            best = [A, B]
            if best_score >= 9:
                break
    
    # Phase 2: Local search from best
    if best_score < 9 and time.time() < deadline:
        A, B = list(best[0]), list(best[1])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try changing each element of A
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val == A[i]:
                        continue
                    if len(set(A)) != 3 and new_val not in set(A):
                        # Check distinctness
                        pass
                    new_A = A[:]
                    new_A[i] = new_val
                    if len(set(new_A)) != 3:
                        continue
                    s = _score(new_A, B)
                    if s > best_score:
                        best_score = s
                        best = [sorted(new_A), sorted(B)]
                        A = sorted(new_A)
                        improved = True
                        if best_score >= 9:
                            break
                if best_score >= 9:
                    break
            if best_score >= 9:
                break
            # Try changing each element of B
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val == B[i]:
                        continue
                    new_B = B[:]
                    new_B[i] = new_val
                    if len(set(new_B)) != 3:
                        continue
                    s = _score(A, new_B)
                    if s > best_score:
                        best_score = s
                        best = [sorted(A), sorted(new_B)]
                        B = sorted(new_B)
                        improved = True
                        if best_score >= 9:
                            break
                if best_score >= 9:
                    break
    
    # Phase 3: More targeted search - try to find 9 distinct pairwise coprime sums
    if best_score < 9 and time.time() < deadline:
        # Strategy: pick sums that are all prime or pairwise coprime, then work backwards
        # For 9 sums all pairwise coprime, many should be prime
        # Let's try: fix B = [1, p, q] where p,q are primes, then A should give coprime sums
        # Or: use the structure that if all sums are prime, they're automatically pairwise coprime (except 2)
        
        # Try: make all 9 sums prime
        # a_i + b_j must all be prime for i,j in {1,2,3}
        # This is a 3x3 magic-like constraint
        
        # Random search specifically targeting prime sums
        prime_set = set()
        for n in range(2, 257):
            if all(n % d != 0 for d in range(2, int(n**0.5) + 1)):
                prime_set.add(n)
        
        for _ in range(200000):
            if time.time() > deadline:
                break
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))
            sums = set()
            all_prime = True
            for a in A:
                for b in B:
                    s = a + b
                    sums.add(s)
                    if s not in prime_set:
                        all_prime = False
                        break
                if not all_prime:
                    break
            if all_prime and len(sums) > best_score:
                best_score = len(sums)
                best = [A, B]
                if best_score >= 9:
                    break
    
    return best
