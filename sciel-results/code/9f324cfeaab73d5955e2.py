"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin for 90s limit
    
    n = 256
    max_size = 64
    
    def is_sidon(s):
        """Check if s is a Sidon set (all pair sums including a+a are unique)."""
        if len(s) <= 1:
            return True
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def get_diffs(s):
        """Get set of nonzero differences from set s."""
        d = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d.add(s[j] - s[i])
        return d
    
    def can_add_to_set(x, s, sums_set, diffs_set, other_diffs):
        """Check if x can be added to set s maintaining Sidon and diff disjointness."""
        # Check Sidon: all pair sums x+s[i] and x+x must be new
        for elem in s:
            if x + elem in sums_set:
                return False
        if x + x in sums_set:
            return False
        # Check differences: x-s[i] and s[i]-x must not conflict with other_diffs
        for elem in s:
            d1 = x - elem
            d2 = elem - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        # Also check that new diffs don't conflict with existing diffs of s
        # (they can't since s is Sidon, but we need disjointness with other set)
        return True
    
    def greedy_mis_construct(order, max_size):
        """
        MIS-inspired greedy: maintain conflict info incrementally.
        At each step, pick the candidate with minimum conflict count.
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Precompute for each x what sums/diffs it would create with existing sets
        # We process in the given order but use a priority queue based on "conflict score"
        
        # Build initial conflict scores for all candidates
        # conflict_score(x, side) = number of existing elements that would create conflicts
        
        def get_conflict_score(x, side):
            """Number of conflicts x would create if added to side."""
            if side == 0:
                s = A
                diffs_other = diffs_B
            else:
                s = B
                diffs_other = diffs_A
            
            score = 0
            for elem in s:
                # Sum conflict: x + elem already in sums
                if side == 0:
                    if (x + elem) in sums_A:
                        score += 1
                else:
                    if (x + elem) in sums_B:
                        score += 1
                # Diff conflict
                d1 = x - elem
                d2 = elem - x
                if d1 in diffs_other:
                    score += 1
                if d2 in diffs_other:
                    score += 1
            # x+x conflict
            if side == 0:
                if (x + x) in sums_A:
                    score += 1
            else:
                if (x + x) in sums_B:
                    score += 1
            return score
        
        # Process candidates
        for x in order:
            if len(A) >= max_size and len(B) >= max_size:
                break
            
            # Determine which side is feasible and pick the one with lower conflict
            best_side = None
            best_score = float('inf')
            
            if len(A) < max_size:
                # Check if x can be added to A
                feasible = True
                for elem in A:
                    if (x + elem) in sums_A:
                        feasible = False
                        break
                    if (x - elem) in diffs_B or (elem - x) in diffs_B:
                        feasible = False
                        break
                if feasible and (x + x) in sums_A:
                    feasible = False
                if feasible:
                    cs = get_conflict_score(x, 0)
                    if cs < best_score:
                        best_score = cs
                        best_side = 0
            
            if len(B) < max_size:
                # Check if x can be added to B
                feasible = True
                for elem in B:
                    if (x + elem) in sums_B:
                        feasible = False
                        break
                    if (x - elem) in diffs_A or (elem - x) in diffs_A:
                        feasible = False
                        break
                if feasible and (x + x) in sums_B:
                    feasible = False
                if feasible:
                    cs = get_conflict_score(x, 1)
                    if cs < best_score:
                        best_score = cs
                        best_side = 1
            
            if best_side is not None:
                if best_side == 0:
                    A.append(x)
                    for elem in A[:-1]:
                        sums_A.add(x + elem)
                        diffs_A.add(x - elem)
                        diffs_A.add(elem - x)
                    sums_A.add(x + x)
                else:
                    B.append(x)
                    for elem in B[:-1]:
                        sums_B.add(x + elem)
                        diffs_B.add(x - elem)
                        diffs_B.add(elem - x)
                    sums_B.add(x + x)
        
        return A, B
    
    def polish(A, B, deadline):
        """Try to add more elements in multiple random orders."""
        A = A[:]
        B = B[:]
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Rebuild state
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])
        
        in_A = set(A)
        in_B = set(B)
        
        # Try adding remaining elements in several random orders
        remaining = [x for x in range(1, n+1) if x not in in_A and x not in in_B]
        
        for polish_trial in range(10):
            if time.time() > deadline:
                break
            random.shuffle(remaining)
            for x in remaining:
                if time.time() > deadline:
                    break
                if len(A) >= max_size and len(B) >= max_size:
                    break
                
                added = False
                # Try A first
                if len(A) < max_size:
                    ok = True
                    for elem in A:
                        if (x + elem) in sums_A:
                            ok = False
                            break
                        if (x - elem) in diffs_B or (elem - x) in diffs_B:
                            ok = False
                            break
                    if ok and (x + x) in sums_A:
                        ok = False
                    if ok:
                        A.append(x)
                        for elem in A[:-1]:
                            sums_A.add(x + elem)
                            diffs_A.add(x - elem)
                            diffs_A.add(elem - x)
                        sums_A.add(x + x)
                        in_A.add(x)
                        added = True
                
                # Try B
                if not added and len(B) < max_size:
                    ok = True
                    for elem in B:
                        if (x + elem) in sums_B:
                            ok = False
                            break
                        if (x - elem) in diffs_A or (elem - x) in diffs_A:
                            ok = False
                            break
                    if ok and (x + x) in sums_B:
                        ok = False
                    if ok:
                        B.append(x)
                        for elem in B[:-1]:
                            sums_B.add(x + elem)
                            diffs_B.add(x - elem)
                            diffs_B.add(elem - x)
                        sums_B.add(x + x)
                        in_B.add(x)
                        added = True
        
        return A, B
    
    # Greedy construction with multiple random orderings
    best_A, best_B = [], []
    best_score = 0
    
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Random permutation of [1, 256]
        order = list(range(1, n+1))
        random.shuffle(order)
        
        A, B = greedy_mis_construct(order, max_size)
        
        # Polish pass
        if time.time() < deadline - 5:
            A, B = polish(A, B, deadline - 2)
        
        # Local search: try to improve by swapping elements
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(A)):
                if time.time() > deadline - 2:
                    break
                for j in range(len(B)):
                    if time.time() > deadline - 2:
                        break
                    # Try swapping A[i] and B[j]
                    A_new = A[:i] + A[i+1:] + [B[j]]
                    B_new = B[:j] + B[j+1:] + [A[i]]
                    if is_sidon(A_new) and is_sidon(B_new):
                        dA = get_diffs(A_new)
                        dB = get_diffs(B_new)
                        if not dA & dB:
                            A, B = A_new, B_new
                            improved = True
                            break
                if improved:
                    break
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
