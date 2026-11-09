"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations_with_replacement, combinations

def construct():
    n = 256
    best_total = 0
    best_A = None
    best_B = None
    
    for seed in range(200):
        rng = random.Random(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Shuffle the order of elements
        elements = list(range(1, n+1))
        rng.shuffle(elements)
        
        for x in elements:
            # Try adding to A
            can_A = True
            # Check sums: x+x and x+s for s in A
            if x + x in sums_A:
                can_A = False
            else:
                for s in A:
                    if x + s in sums_A:
                        can_A = False
                        break
            # Check differences: x-s and s-x for s in A must not be in diffs_B
            if can_A:
                for s in A:
                    d1 = x - s
                    d2 = s - x
                    if d1 in diffs_B or d2 in diffs_B:
                        can_A = False
                        break
            
            # Try adding to B
            can_B = True
            if x + x in sums_B:
                can_B = False
            else:
                for s in B:
                    if x + s in sums_B:
                        can_B = False
                        break
            if can_B:
                for s in B:
                    d1 = x - s
                    d2 = s - x
                    if d1 in diffs_A or d2 in diffs_A:
                        can_B = False
                        break
            
            if can_A and can_B:
                # Randomly choose
                if rng.random() < 0.5:
                    A.append(x)
                    sums_A.add(x + x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                        diffs_A.add(x - s)
                        diffs_A.add(s - x)
                else:
                    B.append(x)
                    sums_B.add(x + x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                        diffs_B.add(x - s)
                        diffs_B.add(s - x)
            elif can_A:
                A.append(x)
                sums_A.add(x + x)
                for s in A[:-1]:
                    sums_A.add(x + s)
                    diffs_A.add(x - s)
                    diffs_A.add(s - x)
            elif can_B:
                B.append(x)
                sums_B.add(x + x)
                for s in B[:-1]:
                    sums_B.add(x + s)
                    diffs_B.add(x - s)
                    diffs_B.add(s - x)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
