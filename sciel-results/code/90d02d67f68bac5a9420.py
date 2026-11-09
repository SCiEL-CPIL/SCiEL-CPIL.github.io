import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256
    max_size = 64
    
    def elapsed():
        return time.time() - start_time

    def can_add(x, S, sums_mask, diff_mask_self, diff_mask_other):
        """
        Check if x can be added to S.
        S: list of elements
        sums_mask: int bitmask where bit s is set if s is a sum in S
        diff_mask_self: int bitmask where bit d is set if d is a diff in S
        diff_mask_other: int bitmask where bit d is set if d is a diff in the other set
        Returns: (bool, new_diffs_list) or (False, None)
        """
        if (sums_mask >> (2*x)) & 1:
            return False, None
        
        for s in S:
            if (sums_mask >> (x+s)) & 1:
                return False, None
        
        new_diffs = []
        for s in S:
            d = abs(x - s)
            if d == 0: continue
            if (diff_mask_other >> d) & 1:
                return False, None
            if (diff_mask_self >> d) & 1:
                return False, None
            new_diffs.append(d)
            
        return True, new_diffs

    def greedy_build(candidates, diff_mask_other, target_size=20):
        S = []
        sums_mask = 0
        diff_mask_self = 0
        
        local_candidates = candidates[:]
        random.shuffle(local_candidates)
        
        for x in local_candidates:
            if len(S) >= target_size:
                break
            ok, new_diffs = can_add(x, S, sums_mask, diff_mask_self, diff_mask_other)
            if ok:
                S.append(x)
                sums_mask |= (1 << (2*x))
                for s in S[:-1]:
                    sums_mask |= (1 << (x+s))
                for d in new_diffs:
                    diff_mask_self |= (1 << d)
                    
        return S, sums_mask, diff_mask_self

    def polish(A, B, sums_mask_A, diff_mask_A, sums_mask_B, diff_mask_B, max_passes=3):
        """
        Multi-pass polish: try inserting elements in different orders.
        """
        for _ in range(max_passes):
            changed = False
            # Try adding to A in ascending order
            for x in range(1, 257):
                if x in A: continue
                ok, new_diffs = can_add(x, A, sums_mask_A, diff_mask_A, diff_mask_B)
                if ok:
                    A.append(x)
                    sums_mask_A |= (1 << (2*x))
                    for s in A[:-1]:
                        sums_mask_A |= (1 << (x+s))
                    for d in new_diffs:
                        diff_mask_A |= (1 << d)
                    changed = True
            
            # Try adding to B in ascending order
            for x in range(1, 257):
                if x in B: continue
                ok, new_diffs = can_add(x, B, sums_mask_B, diff_mask_B, diff_mask_A)
                if ok:
                    B.append(x)
                    sums_mask_B |= (1 << (2*x))
                    for s in B[:-1]:
                        sums_mask_B |= (1 << (x+s))
                    for d in new_diffs:
                        diff_mask_B |= (1 << d)
                    changed = True
            
            if not changed:
                break
        
        return A, B, sums_mask_A, diff_mask_A, sums_mask_B, diff_mask_B

    best_score = 0
    best_sets = ([1], [2])
    
    candidates = list(range(1, n + 1))
    
    for iteration in range(10000):
        if elapsed() > deadline:
            break
            
        # Build A first
        A, sums_A, diffs_A = greedy_build(candidates, 0, target_size=20)
        if not A:
            continue
            
        # Build B, constrained by A's diffs
        B, sums_B, diffs_B = greedy_build(candidates, diffs_A, target_size=20)
        if not B:
            continue
        
        # Polish with multiple passes
        A, B, sums_A, diffs_A, sums_B, diffs_B = polish(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_sets = (A[:], B[:])
            
            if best_score >= 32:
                break

    if best_score == 0:
        A = [1]
        B = [2]
        best_sets = (A, B)

    return list(best_sets)
