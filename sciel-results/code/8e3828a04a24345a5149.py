import itertools
import random
import numpy as np

def count_violations(S):
    """
    Counts the number of pair-sum collisions in S.
    A collision is a pair of distinct pairs (i,j) and (k,l) with i<=j, k<=l
    such that S[i]+S[j] == S[k]+S[l].
    We return the number of such colliding pairs of pairs.
    """
    sums = {}
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    
    violations = 0
    for count in sums.values():
        # Number of pairs of pairs that share this sum
        # If count is c, there are c pairs. The number of ways to pick 2 distinct pairs is C(c, 2)
        violations += count * (count - 1) // 2
    return violations

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S with all unordered pair sums unique.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Check sizes from 12 down to 1
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(list(subset)):
                return k
    return 1

def search_best_set(max_iters=5000, seed=42):
    random.seed(seed)
    # Initialize with a dense set, e.g., 0..11
    current_set = list(range(12))
    current_violations = count_violations(current_set)
    
    best_set = current_set.copy()
    best_violations = current_violations
    
    # Pool of available numbers
    # We want to swap elements to increase violations.
    # To increase violations, we generally want to make the set more "dense" or structured.
    # Swapping a large number with a small number might increase density.
    
    for _ in range(max_iters):
        # Pick a random element to remove
        idx_remove = random.randint(0, 11)
        u = current_set[idx_remove]
        
        # Pick a random element to add from [0, 128] not in current_set
        # To be efficient, we can pick from a smaller range if we believe dense is better.
        # But let's try a range that allows for densification.
        # If current set is sparse, we want to add small numbers.
        # Let's try picking from [0, 30] initially, or just random [0, 128].
        # Given the hypothesis that dense is better, we should bias towards smaller numbers.
        # However, the "structure" matters too.
        
        # Let's try a move: pick a random v in [0, 128] not in S.
        v = random.randint(0, 128)
        if v in current_set:
            continue
            
        # Perform swap
        new_set = current_set.copy()
        new_set[idx_remove] = v
        new_violations = count_violations(new_set)
        
        if new_violations > current_violations:
            current_set = new_set
            current_violations = new_violations
            
            if current_violations > best_violations:
                best_violations = current_violations
                best_set = current_set.copy()
                
            # Optional: restart or continue. Let's continue.
        else:
            # Sometimes accept worse moves to escape local maxima?
            # For now, greedy.
            pass
            
    return best_set

def construct():
    # Run the search
    best_set = search_best_set()
    
    # Verify constraints
    if len(set(best_set)) != 12:
        return list(range(12))
    if min(best_set) < 0 or max(best_set) > 128:
        return list(range(12))
        
    return best_set
