import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock
    
    def distinct_sums(A, B):
        return {a + b for a in A for b in B}
    
    def is_pairwise_coprime(vals):
        lst = list(vals)
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)
    
    # Start with known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Systematic search: try various B sets, then search for A
    # Focus on B sets that might yield prime sums
    
    # Strategy: Try B = {b1, b2, b3} where elements are chosen to spread sums
    # Then for each A candidate, check validity
    
    # Phase 1: Try structured B sets
    # B sets with small spread to keep sums in a manageable range
    candidate_Bs = []
    
    # Generate diverse B candidates
    for b1 in range(1, 20):
        for b2 in range(b1 + 1, 30):
            for b3 in range(b2 + 1, 40):
                candidate_Bs.append([b1, b2, b3])
    
    # Also try some larger B values
    for b1 in range(5, 25, 2):
        for b2 in range(b1 + 3, 50, 3):
            for b3 in range(b2 + 5, 80, 5):
                candidate_Bs.append([b1, b2, b3])
    
    # Phase 2: For each B, search for A
    # A elements must be in [1, 128]
    # To maximize distinct sums, we want A elements well-separated
    
    # For efficiency, limit A search
    # Key insight: if B has small elements, A can range widely
    
    for B in candidate_Bs:
        if time.time() > deadline - 10:
            break
        b_min = min(B)
        b_max = max(B)
        
        # A elements should be in [1, 128], sums in [b_min+1, b_max+128]
        # To get 9 distinct sums, A elements should be sufficiently separated
        # Minimum separation: if A = {a1, a2, a3} with a1 < a2 < a3
        # Sums are a_i + b_j. For all to be distinct, need a2-a1 > max(B)-min(B) 
        # or careful arrangement. Actually not strictly, but helps.
        
        spread_B = b_max - b_min
        
        # Try A with good separation
        # a1 in [1, 10], a2 = a1 + spread_B + 1 to a1 + 50, a3 = a2 + spread_B + 1 to a2 + 50
        for a1 in range(1, 15):
            if time.time() > deadline - 5:
                break
            for a2 in range(a1 + spread_B + 1, min(a1 + spread_B + 60, 128)):
                if time.time() > deadline - 5:
                    break
                for a3 in range(a2 + spread_B + 1, min(a2 + spread_B + 60, 129)):
                    A = [a1, a2, a3]
                    sc = score_of(A, B)
                    if sc > best_score:
                        best_score = sc
                        best_A = A
                        best_B = B
                        if best_score == 9:
                            return best_A, best_B
    
    # Phase 3: Random search with remaining time
    import random
    random.seed(42)
    
    while time.time() < deadline - 2:
        # Generate random A and B
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        
        sc = score_of(A, B)
        if sc > best_score:
            best_score = sc
            best_A = A
            best_B = B
            if best_score == 9:
                break
    
    return best_A, best_B
