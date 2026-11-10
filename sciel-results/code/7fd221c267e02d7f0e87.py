import itertools
import time
from math import gcd

def score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    
    if len(sums) == 0:
        return 0
        
    # Check pairwise coprime
    s_list = list(sums)
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            if gcd(s_list[i], s_list[j]) != 1:
                return 0
                
    return len(s_list)

def construct():
    start_time = time.time()
    timeout = 55.0
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    # Try a small range first
    # If we find 9, we can stop early if we are confident, or just keep searching for better (impossible)
    # But 9 is max. So if we find 9, we can return immediately.
    
    range_limit = 30
    
    # Generate all combinations
    # To avoid O(N^2) full check, we can just iterate.
    # Number of combos for N=30 is 4060.
    # 4060 * 4060 = 16M.
    # 16M * (9 sums + gcds) might be slow in Python.
    # Let's optimize the inner loop.
    
    # Precompute all A combos
    A_combos = list(itertools.combinations(range(1, range_limit + 1), 3))
    B_combos = list(itertools.combinations(range(1, range_limit + 1), 3))
    
    # Sort by potential? No.
    
    for A in A_combos:
        if time.time() - start_time > timeout:
            break
        for B in B_combos:
            if time.time() - start_time > timeout:
                break
            
            # Fast check
            sums = []
            valid = True
            # Check distinctness and coprime
            # We can use a set for distinctness
            s_set = set()
            for a in A:
                for b in B:
                    s = a + b
                    if s in s_set:
                        # Duplicates reduce cardinality, but don't invalidate coprime condition of *distinct* values.
                        # But we want max cardinality. If duplicate, cardinality < 9.
                        # We can still check coprime for the distinct ones.
                        pass
                    s_set.add(s)
            
            if len(s_set) < best_score:
                continue # Can't beat best if cardinality is lower
                
            s_list = list(s_set)
            # Check pairwise coprime
            for i in range(len(s_list)):
                for j in range(i + 1, len(s_list)):
                    if gcd(s_list[i], s_list[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                if len(s_set) > best_score:
                    best_score = len(s_set)
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        # Max possible
                        return [list(best_A), list(best_B)]

    return [list(best_A), list(best_B)]
