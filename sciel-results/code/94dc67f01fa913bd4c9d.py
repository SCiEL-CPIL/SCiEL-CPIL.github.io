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

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    best = [[1, 7, 13], [4, 10, 16]]
    best_score = 5  # seed score
    
    # Precompute primes up to 256 (max possible sum is 128+128=256)
    primes_up_to_256 = set()
    for n in range(2, 257):
        if _is_prime(n):
            primes_up_to_256.add(n)
    
    rng = random.Random(42)
    
    # Phase 1: Random search with early termination
    for _ in range(200000):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) > best_score:
            if _pairwise_coprime(list(sums)):
                best_score = len(sums)
                best = [A, B]
                if best_score >= 9:
                    return best
    
    # Phase 2: Systematic search - fix A, find B
    # Try small structured A values
    candidate_As = []
    for a1 in range(1, 30):
        for a2 in range(a1+1, 50):
            for a3 in range(a2+1, 70):
                candidate_As.append([a1, a2, a3])
    
    # Shuffle for diversity
    rng.shuffle(candidate_As)
    
    for A in candidate_As[:50000]:
        if time.time() > deadline:
            break
        # For this A, find B such that all sums are pairwise coprime
        # Try random B values
        for _ in range(200):
            if time.time() > deadline:
                break
            B = sorted(rng.sample(range(1, 129), 3))
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) > best_score:
                if _pairwise_coprime(list(sums)):
                    best_score = len(sums)
                    best = [sorted(A), sorted(B)]
                    if best_score >= 9:
                        return best
    
    # Phase 3: Local search from best
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
                    new_A = A[:]
                    new_A[i] = new_val
                    if len(set(new_A)) != 3:
                        continue
                    sums = set()
                    for a in new_A:
                        for b in B:
                            sums.add(a + b)
                    if len(sums) > best_score:
                        if _pairwise_coprime(list(sums)):
                            best_score = len(sums)
                            best = [sorted(new_A), sorted(B)]
                            A = sorted(new_A)
                            improved = True
                            if best_score >= 9:
                                return best
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
                    sums = set()
                    for a in A:
                        for b in new_B:
                            sums.add(a + b)
                    if len(sums) > best_score:
                        if _pairwise_coprime(list(sums)):
                            best_score = len(sums)
                            best = [sorted(A), sorted(new_B)]
                            B = sorted(new_B)
                            improved = True
                            if best_score >= 9:
                                return best
                if best_score >= 9:
                    break
    
    # Phase 4: Targeted search for all-prime-sum configurations
    if best_score < 9 and time.time() < deadline:
        # If all 9 sums are prime, they're automatically pairwise coprime (except possibly 2)
        # We need a_i + b_j to be prime for all i,j
        # Try structured approaches
        
        # Approach: Use B = [1, 2, 3] and find A such that a+1, a+2, a+3 are all prime
        # These are "prime triplets" - rare but possible
        # Actually, a+1 and a+2 can't both be prime unless one is 2
        # So try B = [1, 3, 5] or other odd spacings
        
        # Better: search for A where A+B gives all primes
        # Use the fact that if B has even spread, A values should be chosen carefully
        
        for _ in range(100000):
            if time.time() > deadline:
                break
            # Random A
            A = sorted(rng.sample(range(1, 129), 3))
            # Try to find B such that all sums are prime
            # For each candidate B value, check if a+b is prime for all a in A
            valid_b = []
            for b in range(1, 129):
                if all((a + b) in primes_up_to_256 for a in A):
                    valid_b.append(b)
            
            if len(valid_b) >= 3:
                # Try all combinations of 3 from valid_b
                for B in combinations(valid_b, 3):
                    B = list(B)
                    if len(set(B)) != 3:
                        continue
                    sums = set()
                    for a in A:
                        for b in B:
                            sums.add(a + b)
                    if len(sums) > best_score:
                        if _pairwise_coprime(list(sums)):
                            best_score = len(sums)
                            best = [sorted(A), sorted(B)]
                            if best_score >= 9:
                                return best
    
    return best
