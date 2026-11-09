import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # safety margin before 90s kill

def _pair_sums_unique(elements):
    """Check if all pair sums (including a+a) are unique."""
    sums = set()
    for i, a in enumerate(elements):
        for j in range(i, len(elements)):
            s = a + elements[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diffs(elements):
    """Get set of nonzero signed differences."""
    d = set()
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            diff = elements[j] - elements[i]
            d.add(diff)
            d.add(-diff)
    return d

def _can_add_to_set(x, elements, sum_set, diff_set, other_diff_set):
    """Check if x can be added to the set with given incremental data."""
    # Check pair sums uniqueness
    for a in elements:
        s = x + a
        if s in sum_set:
            return False
    s2 = x + x
    if s2 in sum_set:
        return False
    # Check differences
    new_diffs = set()
    for a in elements:
        d = x - a
        new_diffs.add(d)
        new_diffs.add(-d)
    # Check no conflict with own diff set (shouldn't happen if sums are unique, but be safe)
    if new_diffs & diff_set:
        return False
    # Check no conflict with other set's diff set
    if new_diffs & other_diff_set:
        return False
    return True

def _greedy_build():
    """Greedy construction with full range [1,256]."""
    A = []
    B = []
    sumA = set()
    sumB = set()
    diffA = set()
    diffB = set()
    
    for x in range(1, N + 1):
        # Try adding to A
        if len(A) < MAX_EACH and _can_add_to_set(x, A, sumA, diffA, diffB):
            # Update sumA
            for a in A:
                sumA.add(x + a)
            sumA.add(x + x)
            # Update diffA
            for a in A:
                d = x - a
                diffA.add(d)
                diffA.add(-d)
            A.append(x)
        
        # Try adding to B
        if len(B) < MAX_EACH and _can_add_to_set(x, B, sumB, diffB, diffA):
            # Update sumB
            for b in B:
                sumB.add(x + b)
            sumB.add(x + x)
            # Update diffB
            for b in B:
                d = x - b
                diffB.add(d)
                diffB.add(-d)
            B.append(x)
    
    return A, B, sumA, sumB, diffA, diffB

def _local_search(A, B, sumA, sumB, diffA, diffB, deadline):
    """Try to improve via local search: add elements, try swaps."""
    t0 = time.time()
    A = list(A)
    B = list(B)
    sumA = set(sumA)
    sumB = set(sumB)
    diffA = set(diffA)
    diffB = set(diffB)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements not in either set
        for x in range(1, N + 1):
            if x in A or x in B:
                continue
            # Try adding to A
            if len(A) < MAX_EACH and _can_add_to_set(x, A, sumA, diffA, diffB):
                for a in A:
                    sumA.add(x + a)
                sumA.add(x + x)
                for a in A:
                    d = x - a
                    diffA.add(d)
                    diffA.add(-d)
                A.append(x)
                improved = True
                break
            # Try adding to B
            if len(B) < MAX_EACH and _can_add_to_set(x, B, sumB, diffB, diffA):
                for b in B:
                    sumB.add(x + b)
                sumB.add(x + x)
                for b in B:
                    d = x - b
                    diffB.add(d)
                    diffB.add(-d)
                B.append(x)
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Try swaps: remove one element from A, try to add a different one
        if len(A) > 0:
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                a_remove = A[i]
                # Remove a_remove from A
                A_new = [a for j, a in enumerate(A) if j != i]
                # Recompute sumA and diffA for A_new
                sumA_new = set()
                for j, a in enumerate(A_new):
                    for k in range(j, len(A_new)):
                        sumA_new.add(a + A_new[k])
                diffA_new = set()
                for j in range(len(A_new)):
                    for k in range(j + 1, len(A_new)):
                        d = A_new[k] - A_new[j]
                        diffA_new.add(d)
                        diffA_new.add(-d)
                
                # Try to add a new element to A_new
                for x in range(1, N + 1):
                    if x in A_new or x in B:
                        continue
                    if len(A_new) < MAX_EACH and _can_add_to_set(x, A_new, sumA_new, diffA_new, diffB):
                        # Check if this improves: we removed a_remove and added x
                        # Net change: 0 (swap), but maybe x allows more additions later
                        # For now, just accept if it doesn't decrease size
                        # Actually, let's only do swaps if they enable further additions
                        # For simplicity, skip swaps for now
                        pass
                # Restore A
                A = list(A)
    
    return A, B, sumA, sumB, diffA, diffB

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    best_A, best_B = [], []
    best_size = 0
    
    # Phase 1: Greedy with full range
    A, B, sumA, sumB, diffA, diffB = _greedy_build()
    best_A, best_B = list(A), list(B)
    best_size = len(A) + len(B)
    
    # Phase 2: Local search
    if time.time() < deadline:
        A, B, sumA, sumB, diffA, diffB = _local_search(A, B, sumA, sumB, diffA, diffB, deadline)
        if len(A) + len(B) > best_size:
            best_A, best_B = list(A), list(B)
            best_size = len(A) + len(B)
    
    # Phase 3: Multiple random restarts
    if time.time() < deadline:
        for restart in range(50):
            if time.time() > deadline - 2.0:
                break
            
            # Randomized greedy: shuffle the order of x
            xs = list(range(1, N + 1))
            random.shuffle(xs)
            
            A = []
            B = []
            sumA = set()
            sumB = set()
            diffA = set()
            diffB = set()
            
            for x in xs:
                if time.time() > deadline - 2.0:
                    break
                
                # Try adding to A
                if len(A) < MAX_EACH and _can_add_to_set(x, A, sumA, diffA, diffB):
                    for a in A:
                        sumA.add(x + a)
                    sumA.add(x + x)
                    for a in A:
                        d = x - a
                        diffA.add(d)
                        diffA.add(-d)
                    A.append(x)
                
                # Try adding to B
                if len(B) < MAX_EACH and _can_add_to_set(x, B, sumB, diffB, diffA):
                    for b in B:
                        sumB.add(x + b)
                    sumB.add(x + x)
                    for b in B:
                        d = x - b
                        diffB.add(d)
                        diffB.add(-d)
                    B.append(x)
            
            if len(A) + len(B) > best_size:
                best_A, best_B = list(A), list(B)
                best_size = len(A) + len(B)
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
