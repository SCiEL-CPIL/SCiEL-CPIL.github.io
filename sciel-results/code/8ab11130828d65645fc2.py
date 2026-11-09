"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def is_sidon(S):
    """Check if set S is Sidon."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def get_diffs(S):
    """Get set of non-zero absolute differences."""
    diffs = set()
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            diffs.add(abs(S[i] - S[j]))
    return diffs

def construct():
    n = 256
    
    # Phase 1: Deterministic Greedy Construction
    A = []
    B = []
    pool = []
    diff_A = set()
    diff_B = set()
    
    for x in range(1, n + 1):
        in_A = False
        in_B = False
        
        # Try adding to A
        temp_A = A + [x]
        if is_sidon(temp_A):
            new_diffs_A = get_diffs(temp_A)
            if not (new_diffs_A & diff_B):
                A.append(x)
                diff_A = new_diffs_A
                in_A = True
        
        # Try adding to B
        temp_B = B + [x]
        if is_sidon(temp_B):
            new_diffs_B = get_diffs(temp_B)
            if not (new_diffs_B & diff_A):
                B.append(x)
                diff_B = new_diffs_B
                in_B = True
        
        if not in_A and not in_B:
            pool.append(x)
            
    # Phase 2: Local Search
    # We try to improve the size by adding from pool or swapping.
    
    improved = True
    while improved:
        improved = False
        pool_set = set(pool)
        
        # 1. Try simple additions
        for x in pool[:]:
            # Try add to A
            if x not in A:
                temp_A = A + [x]
                if is_sidon(temp_A):
                    new_diffs_A = get_diffs(temp_A)
                    if not (new_diffs_A & diff_B):
                        A.append(x)
                        diff_A = new_diffs_A
                        pool.remove(x)
                        improved = True
                        break
            
            # Try add to B
            if x not in B:
                temp_B = B + [x]
                if is_sidon(temp_B):
                    new_diffs_B = get_diffs(temp_B)
                    if not (new_diffs_B & diff_A):
                        B.append(x)
                        diff_B = new_diffs_B
                        pool.remove(x)
                        improved = True
                        break
        
        if improved:
            continue
        
        # 2. Try Swaps (Remove 1 from A, Add 1 or 2 from Pool)
        # We try to remove an element from A and see if we can add more from Pool.
        for a in A[:]:
            # Remove a
            A.remove(a)
            old_diff_A = diff_A
            diff_A = get_diffs(A)
            
            added_count = 0
            added_elements = []
            
            # Try to add candidates from pool to A
            for x in pool[:]:
                if x in A: continue
                temp_A = A + [x]
                if is_sidon(temp_A):
                    new_diffs_A = get_diffs(temp_A)
                    if not (new_diffs_A & diff_B):
                        A.append(x)
                        diff_A = new_diffs_A
                        pool.remove(x)
                        added_elements.append(x)
                        added_count += 1
                        break
                        
            # Try to add another from pool to A
            for x in pool[:]:
                if x in A: continue
                temp_A = A + [x]
                if is_sidon(temp_A):
                    new_diffs_A = get_diffs(temp_A)
                    if not (new_diffs_A & diff_B):
                        A.append(x)
                        diff_A = new_diffs_A
                        pool.remove(x)
                        added_elements.append(x)
                        added_count += 1
                        break
                        
            # Try to add one to B
            for x in pool[:]:
                if x in B: continue
                temp_B = B + [x]
                if is_sidon(temp_B):
                    new_diffs_B = get_diffs(temp_B)
                    if not (new_diffs_B & diff_A):
                        B.append(x)
                        diff_B = new_diffs_B
                        pool.remove(x)
                        added_elements.append(x)
                        added_count += 1
                        break
                        
            # Check if we gained size
            # We removed 1 (a). We added added_count.
            if added_count > 1:
                # Net gain
                improved = True
                break
            else:
                # Revert
                for x in added_elements:
                    if x in A:
                        A.remove(x)
                    if x in B:
                        B.remove(x)
                    pool.append(x)
                diff_A = old_diff_A
                A.append(a)
                
        # Similar for B swaps (optional, but let's add for symmetry)
        if not improved:
            for b in B[:]:
                B.remove(b)
                old_diff_B = diff_B
                diff_B = get_diffs(B)
                
                added_count = 0
                added_elements = []
                
                # Try to add candidates from pool to B
                for x in pool[:]:
                    if x in B: continue
                    temp_B = B + [x]
                    if is_sidon(temp_B):
                        new_diffs_B = get_diffs(temp_B)
                        if not (new_diffs_B & diff_A):
                            B.append(x)
                            diff_B = new_diffs_B
                            pool.remove(x)
                            added_elements.append(x)
                            added_count += 1
                            break
                            
                # Try to add another from pool to B
                for x in pool[:]:
                    if x in B: continue
                    temp_B = B + [x]
                    if is_sidon(temp_B):
                        new_diffs_B = get_diffs(temp_B)
                        if not (new_diffs_B & diff_A):
                            B.append(x)
                            diff_B = new_diffs_B
                            pool.remove(x)
                            added_elements.append(x)
                            added_count += 1
                            break
                            
                # Try to add one to A
                for x in pool[:]:
                    if x in A: continue
                    temp_A = A + [x]
                    if is_sidon(temp_A):
                        new_diffs_A = get_diffs(temp_A)
                        if not (new_diffs_A & diff_B):
                            A.append(x)
                            diff_A = new_diffs_A
                            pool.remove(x)
                            added_elements.append(x)
                            added_count += 1
                            break
                            
                if added_count > 1:
                    improved = True
                    break
                else:
                    # Revert
                    for x in added_elements:
                        if x in A:
                            A.remove(x)
                        if x in B:
                            B.remove(x)
                        pool.append(x)
                    diff_B = old_diff_B
                    B.append(b)

    return [A, B]

# EVOLVE-BLOCK-END
