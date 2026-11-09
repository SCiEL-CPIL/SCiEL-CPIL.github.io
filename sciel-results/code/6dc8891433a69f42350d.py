"""Search for 3x3 integer sets with pairwise-coprime distinct sumsets."""
import math
import random
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_pairwise_coprime(vals):
        """Check if all distinct values in vals are pairwise coprime."""
        distinct = sorted(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    def score(A, B):
        """Return (num_distinct, pairwise_coprime) for sumset A+B."""
        sums = [a + b for a in A for b in B]
        distinct = sorted(set(sums))
        pc = is_pairwise_coprime(distinct)
        return len(distinct), pc

    def evaluate(A, B):
        """Score: if pairwise coprime, return count; else negative penalty."""
        n, pc = score(A, B)
        if pc:
            return n
        else:
            return -n  # penalty: prefer fewer sums when not coprime

    # Precompute some useful data
    # Primes up to 256 for potential use
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return [i for i in range(2, n+1) if is_p[i]]
    
    primes = sieve(256)
    
    # Strategy: We want 9 distinct sums that are pairwise coprime.
    # Sums range from 2 to 256.
    # Pairwise coprime sets of size 9 in [2,256]:
    # We can use primes, or 1, or products of distinct primes.
    
    # Let's try a systematic + random search approach.
    
    best_score = 0
    best_A = None
    best_B = None
    
    # Phase 1: Try to find 9 distinct pairwise coprime sums
    # Key insight: if we can make all 9 sums distinct AND pairwise coprime, score = 9.
    # The sums are a_i + b_j for i,j in {0,1,2}.
    
    # Approach: pick 9 pairwise coprime targets, then try to factor them as A+B.
    # For 3x3, sums form a "matrix" S where S[i][j] = A[i] + B[j].
    # Constraint: S[i][j] - S[i][0] = B[j] - B[0] (same for all i)
    #           S[i][j] - S[0][j] = A[i] - A[0] (same for all j)
    # So S must be of the form S[i][j] = c + A[i] + B[j] where c=0 offset... 
    # Actually S[i][j] = A[i] + B[j]. The constraint is:
    # S[i][j] + S[0][0] == S[i][0] + S[0][j] for all i,j (additive structure)
    
    # So we need a 3x3 matrix of pairwise coprime values satisfying:
    # M[i][j] + M[0][0] = M[i][0] + M[0][j]
    # And M[i][j] = A[i] + B[j] with A[i], B[j] in [1,128], all A distinct, all B distinct.
    # Given M, A[i] = M[i][0] - B[0], B[j] = M[0][j] - B[0]... 
    # Let's parameterize: pick A[0], A[1], A[2], B[0], B[1], B[2] all in [1,128].
    # M[i][j] = A[i] + B[j].
    
    # For random search, just pick random A and B and check.
    
    # Phase 1: Random search with restarts
    random.seed(42)
    
    for iteration in range(500000):
        if time.time() > DEADLINE:
            break
        
        # Random A and B
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        s = evaluate(A, B)
        if s > best_score:
            best_score = s
            best_A = A[:]
            best_B = B[:]
        
        if best_score >= 9:
            break
    
    # Phase 2: Local search / hill climbing from best found
    if best_A is not None and time.time() < DEADLINE:
        current_A = best_A[:]
        current_B = best_B[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 5000
        
        while no_improve < max_no_improve and time.time() < DEADLINE:
            # Try random perturbation
            A2 = current_A[:]
            B2 = current_B[:]
            
            # Mutate one element
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                A2[idx] = random.randint(1, 128)
            else:
                idx = random.randint(0, 2)
                B2[idx] = random.randint(1, 128)
            
            # Ensure distinctness
            if len(set(A2)) < 3 or len(set(B2)) < 3:
                no_improve += 1
                continue
            
            s2 = evaluate(A2, B2)
            if s2 > current_score:
                current_A = A2
                current_B = B2
                current_score = s2
                no_improve = 0
                if current_score >= 9:
                    break
            else:
                no_improve += 1
    
    # Phase 3: Structured search - try to build 9 pairwise coprime sums
    # Pick 9 pairwise coprime numbers in [2, 256], then solve for A, B.
    if time.time() < DEADLINE - 5:
        # Generate candidate sets of 9 pairwise coprime numbers
        # Use primes and 1
        # Actually, let's try: pick 9 primes, arrange in 3x3 additive matrix
        
        # For a 3x3 additive matrix M[i][j] = A[i] + B[j]:
        # M[0][0] = A0+B0, M[0][1] = A0+B1, M[0][2] = A0+B2
        # M[1][0] = A1+B0, M[1][1] = A1+B1, M[1][2] = A1+B2
        # M[2][0] = A2+B0, M[2][1] = A2+B1, M[2][2] = A2+B2
        # 
        # Given M, we can recover: A[i] = M[i][0] - B[0], B[j] = M[0][j] - B[0]
        # Let B[0] = t. Then A[i] = M[i][0] - t, B[j] = M[0][j] - t.
        # Need all A[i], B[j] in [1, 128] and distinct.
        
        # Try: pick 9 pairwise coprime targets, check if they form additive matrix.
        # That's hard to search directly. Instead, let's do targeted random search
        # biased toward making sums prime or pairwise coprime.
        
        # Better: use the structure. Pick A0, B0 first, then choose A1, A2, B1, B2
        # to make all 9 sums pairwise coprime.
        
        for attempt in range(200000):
            if time.time() > DEADLINE:
                break
            
            A0 = random.randint(1, 128)
            B0 = random.randint(1, 128)
            base = A0 + B0
            
            # We need 8 more sums: A0+B1, A0+B2, A1+B0, A1+B1, A1+B2, A2+B0, A2+B1, A2+B2
            # All 9 must be pairwise coprime.
            
            # Pick A1, A2, B1, B2
            A1 = random.randint(1, 128)
            A2 = random.randint(1, 128)
            B1 = random.randint(1, 128)
            B2 = random.randint(1, 128)
            
            A = [A0, A1, A2]
            B = [B0, B1, B2]
            
            if len(set(A)) < 3 or len(set(B)) < 3:
                continue
            
            s = evaluate(A, B)
            if s > best_score:
                best_score = s
                best_A = A[:]
                best_B = B[:]
            
            if best_score >= 9:
                break
    
    # Phase 4: Final targeted search for 9
    # Try specific constructions
    if best_score < 9 and time.time() < DEADLINE - 2:
        # Known approach: use primes
        # Try A = [p1, p2, p3], B = [q1, q2, q3] where all p_i + q_j are pairwise coprime
        # This is a strong constraint. Let's try small primes.
        
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
        
        for A in combinations(small_primes, 3):
            if time.time() > DEADLINE:
                break
            for B in combinations(small_primes, 3):
                if len(set(A)) < 3 or len(set(B)) < 3:
                    continue
                s = evaluate(list(A), list(B))
                if s > best_score:
                    best_score = s
                    best_A = list(A)
                    best_B = list(B)
                if best_score >= 9:
                    break
            if best_score >= 9:
                break
    
    # Return best found
    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]
