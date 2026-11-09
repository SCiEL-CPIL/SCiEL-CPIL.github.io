"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time
from math import gcd
from itertools import combinations

def _pairwise_coprime_check(values):
    """Check if all distinct values in the list are pairwise coprime."""
    vals = sorted(set(values))
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """
    Evaluate a candidate (A, B).
    Returns (score, valid, distinct_count) where:
      - valid: True if all distinct sums are pairwise coprime
      - distinct_count: number of distinct values in A+B
      - score: distinct_count if valid, else 0
    """
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if not _pairwise_coprime_check(distinct):
        return 0, False, len(distinct)
    return len(distinct), True, len(distinct)

def _is_valid_candidate(A, B):
    """Check if A and B are valid inputs: 3 distinct integers in [1, 128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A):
        return False
    if any(x < 1 or x > 128 for x in B):
        return False
    return True

def construct():
    """
    Return two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5-second safety margin
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    # For pairwise coprime, each distinct sum should ideally be prime or a product of unique primes
    # But primes are the safest bet
    
    # Strategy: systematic search with parity constraints
    # Key insight: if all sums are odd primes, they are pairwise coprime
    # For sums to be odd, we need a+b odd for all pairs
    # This means one set is all even and the other is all odd (or vice versa)
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # First, try a systematic approach:
    # A = all even, B = all odd (or A odd, B even)
    # Sums will all be odd, so if they're all prime, they're pairwise coprime
    
    even_nums = list(range(2, 129, 2))  # 2, 4, 6, ..., 128
    odd_nums = list(range(1, 129, 2))   # 1, 3, 5, ..., 127
    
    # Precompute primality up to 256
    is_prime = [True] * 257
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(256**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, 257, i):
                is_prime[j] = False
    
    # Try combinations of 3 evens and 3 odds where all 9 sums are prime
    # This gives 9 distinct sums if all are distinct primes
    
    found = False
    
    # Iterate over combinations of 3 evens
    # To keep it fast, limit the search
    evens_sample = even_nums  # All evens, 64 elements
    odds_sample = odd_nums    # All odds, 64 elements
    
    # For speed, use a smarter approach:
    # Pick 3 evens and 3 odds, check if all 9 sums are prime
    # Use random sampling with systematic bias
    
    random.seed(42)
    
    # First pass: try some structured patterns
    # Pattern 1: consecutive evens and odds
    for base_e in range(2, 100, 2):
        if time.time() > deadline:
            break
        for base_o in range(1, 100, 2):
            A = [base_e, base_e + 2, base_e + 4]
            B = [base_o, base_o + 2, base_o + 4]
            if not _is_valid_candidate(A, B):
                continue
            sums = [a + b for a in A for b in B]
            distinct = list(set(sums))
            if len(distinct) == 9 and all(is_prime[s] for s in distinct):
                score, valid, dc = _evaluate(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
                    found = True
                    if best_score == 9:
                        return best_A, best_B
    
    # Second pass: random search with parity constraint
    if not found or best_score < 9:
        iterations = 0
        while time.time() < deadline:
            iterations += 1
            
            # Random 3 evens and 3 odds
            A = random.sample(even_nums, 3)
            B = random.sample(odd_nums, 3)
            
            sums = [a + b for a in A for b in B]
            distinct = list(set(sums))
            
            # Quick check: are all distinct sums prime?
            if all(is_prime[s] for s in distinct):
                score, valid, dc = _evaluate(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return best_A, best_B
            else:
                # Check if they're pairwise coprime (not necessarily prime)
                score, valid, dc = _evaluate(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return best_A, best_B
    
    # If we haven't found 9, try the reverse parity (A odd, B even)
    if best_score < 9:
        while time.time() < deadline:
            A = random.sample(odd_nums, 3)
            B = random.sample(even_nums, 3)
            
            sums = [a + b for a in A for b in B]
            distinct = list(set(sums))
            
            if all(is_prime[s] for s in distinct):
                score, valid, dc = _evaluate(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return best_A, best_B
            else:
                score, valid, dc = _evaluate(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return best_A, best_B
    
    # If still not 9, try mixed parity but check pairwise coprime
    if best_score < 9:
        while time.time() < deadline:
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            
            sums = [a + b for a in A for b in B]
            distinct = list(set(sums))
            
            score, valid, dc = _evaluate(A, B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    return best_A, best_B
    
    return best_A, best_B
# EVOLVE-BLOCK-END
