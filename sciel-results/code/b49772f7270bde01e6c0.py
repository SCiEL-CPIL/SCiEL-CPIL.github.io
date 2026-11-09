"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement, combinations

def construct():
    n = 256
    max_size = 64
    
    # We want to maximize |A| + |B| where A and B are Sidon sets in [1, n]
    # with disjoint nonzero difference sets.
    
    # Strategy: Use a greedy approach but with better ordering and backtracking hints.
    # Key insight: The constraint is on differences being disjoint. 
    # If we pick elements carefully, we can maximize the total size.
    
    # Let's try a different approach: 
    # 1. Generate a large Sidon set first (say for A)
    # 2. Then greedily add elements to B that don't conflict with A's differences
    
    # Actually, let's think about the structure. 
    # A Sidon set in [1, n] has at most ~sqrt(n) elements typically, but can be larger.
    # For n=256, sqrt(256)=16, so we can have up to ~16-20 elements in a Sidon set.
    # But the task says "at most 64 elements each", which is a loose upper bound.
    # The real constraint is the Sidon property (unique pair sums).
    
    # Let's use a randomized greedy approach with multiple restarts to find a good solution.
    
    best_total = 0
    best_A = []
    best_B = []
    
    import random
    random.seed(42)
    
    for trial in range(50):
        A = []
        B = []
        
        # Alternate adding elements to A and B to balance sizes
        # Start with a random ordering of candidates
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # For each candidate, try adding to A or B (or both? No, each element goes to one set)
        # Actually, elements can be in both sets? "A and B need not themselves be disjoint"
        # So an element can be in both A and B.
        
        # Let's try: for each element, decide if it goes to A, B, or both.
        # But this is complex. Let's simplify: 
        # Greedily build A first, then B, but allow overlap.
        
        # Better: Alternate between adding to A and B.
        
        diff_A = set()
        diff_B = set()
        
        for x in candidates:
            # Try adding x to A
            can_add_A = True
            new_diffs_A = set()
            for a in A:
                d = abs(x - a)
                if d in diff_A or d in diff_B:
                    can_add_A = False
                    break
                new_diffs_A.add(d)
            
            if can_add_A:
                # Check Sidon property for A
                temp_A = A + [x]
                sums = []
                for i in range(len(temp_A)):
                    for j in range(i, len(temp_A)):
                        sums.append(temp_A[i] + temp_A[j])
                if len(sums) == len(set(sums)):
                    A.append(x)
                    diff_A.update(new_diffs_A)
            
            # Try adding x to B
            can_add_B = True
            new_diffs_B = set()
            for b in B:
                d = abs(x - b)
                if d in diff_A or d in diff_B:
                    can_add_B = False
                    break
                new_diffs_B.add(d)
            
            if can_add_B:
                # Check Sidon property for B
                temp_B = B + [x]
                sums = []
                for i in range(len(temp_B)):
                    for j in range(i, len(temp_B)):
                        sums.append(temp_B[i] + temp_B[j])
                if len(sums) == len(set(sums)):
                    B.append(x)
                    diff_B.update(new_diffs_B)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
