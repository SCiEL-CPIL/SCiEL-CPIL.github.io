import random
import sys
from itertools import combinations, combinations_with_replacement
import heapq

def construct():
    n = 256
    max_size = 64
    random.seed(42)

    # We'll build two Sidon sets A and B with disjoint nonzero differences.
    # Use a greedy + local search approach.
    
    # First, generate a good initial solution using the original greedy
    # but in a smarter order, then improve via local swaps.
    
    def is_sidon(S):
        if len(S) <= 1:
            return True
        sums = set()
        for a in S:
            for b in S:
                if a <= b:
                    if (a + b) in sums:
                        return False
                    sums.add(a + b)
        return True

    def nonzero_diffs(S):
        if len(S) <= 1:
            return set()
        diffs = set()
        for a in S:
            for b in S:
                if a != b:
                    diffs.add(b - a)
        return diffs

    def can_add(S, x, other_diffs):
        # Check if x can be added to S maintaining Sidon and disjoint diffs
        new_S = S + [x]
        if not is_sidon(new_S):
            return False
        new_diffs = nonzero_diffs(new_S)
        if new_diffs & other_diffs:
            return False
        return True

    # Greedy construction with multiple random orderings, keep best
    best_total = 0
    best_A = []
    best_B = []

    for trial in range(200):
        random.seed(trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A = []
        B = []
        
        for x in order:
            # Try adding to A
            if len(A) < max_size:
                other_diffs = nonzero_diffs(B)
                if can_add(A, x, other_diffs):
                    A.append(x)
                    continue
            # Try adding to B
            if len(B) < max_size:
                other_diffs = nonzero_diffs(A)
                if can_add(B, x, other_diffs):
                    B.append(x)
                    continue
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]

    A = best_A[:]
    B = best_B[:]

    # Local search: try swapping elements to improve
    # For simplicity, do a few rounds of trying to replace elements
    for iteration in range(50):
        improved = False
        
        # Try to add a new element by removing one that conflicts
        for side, S in [(0, A), (1, B)]:
            if len(S) >= max_size:
                continue
            other = B if side == 0 else A
            other_diffs = nonzero_diffs(other)
            
            # Find candidates not in S
            existing = set(S)
            for x in range(1, n + 1):
                if x in existing:
                    continue
                if can_add(S, x, other_diffs):
                    if side == 0:
                        A.append(x)
                    else:
                        B.append(x)
                    improved = True
                    break
        
        if not improved:
            break

    # Try swaps: replace an element in A with another that allows more in B
    # and vice versa
    for swap_round in range(20):
        improved = False
        for i, a_val in enumerate(A[:]):
            if a_val not in A:
                continue
            # Try removing a_val and adding a different element
            A_temp = A[:]
            A_temp.remove(a_val)
            B_diffs = nonzero_diffs(B)
            
            for x in range(1, n + 1):
                if x in A_temp or x in B:
                    continue
                if can_add(A_temp, x, B_diffs):
                    # Check if this improves total
                    A_new = A_temp + [x]
                    # Can we now add more to B?
                    A_diffs_new = nonzero_diffs(A_new)
                    B_new = B[:]
                    for y in range(1, n + 1):
                        if y in B_new or y in A_new:
                            continue
                        if len(B_new) >= max_size:
                            break
                        if can_add(B_new, y, A_diffs_new):
                            B_new.append(y)
                            break
                    if len(A_new) + len(B_new) > len(A) + len(B):
                        A = A_new
                        B = B_new
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    return [A, B]
