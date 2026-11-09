import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256
    max_size = 64
    
    def elapsed():
        return time.time() - start_time

    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs_A.add(abs(A[j] - A[i]))
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                diffs_B.add(abs(B[j] - B[i]))
                
        if diffs_A & diffs_B:
            return False
            
        return True

    def can_add(x, S, sums, diff_mask, other_diff_mask):
        """
        Check if x can be added to S.
        S: list of elements
        sums: set of pair sums in S
        diff_mask: bitmask of differences in S (bit d set if d in diffs)
        other_diff_mask: bitmask of differences in the other set
        Returns: (bool, new_diffs_list) or (False, None)
        """
        # Check Sidon
        if (2*x) in sums:
            return False, None
        for s in S:
            if (x + s) in sums:
                return False, None
        
        # Check differences
        new_diffs = []
        for s in S:
            d = abs(x - s)
            if d == 0:
                continue
            # Check if d is already in this set's differences (allowed, but we track it)
            # Actually, for Sidon, differences don't need to be unique, just the sums.
            # But we need disjoint difference sets between A and B.
            # So d must not be in other_diff_mask.
            if (other_diff_mask >> d) & 1:
                return False, None
            new_diffs.append(d)
            
        return True, new_diffs

    def greedy_build(candidates, other_diff_mask, target_size=16):
        """
        Greedily build a Sidon set with differences disjoint from other_diff_mask.
        """
        S = []
        sums = set()
        diff_mask = 0
        
        # Sort candidates to try smaller ones first? Or random?
        # Random is usually better for diversity, but sorted might find denser packs?
        # Let's try a mix.
        
        for x in candidates:
            if len(S) >= target_size:
                break
            ok, new_diffs = can_add(x, S, sums, diff_mask, other_diff_mask)
            if ok:
                S.append(x)
                sums.add(2*x)
                for s in S[:-1]:
                    sums.add(x + s)
                for d in new_diffs:
                    diff_mask |= (1 << d)
                    
        return S

    def local_search(A, B, time_limit=5.0):
        """
        Try to improve A and B by adding elements.
        """
        end_time = time.time() + time_limit
        A = A[:]
        B = B[:]
        
        # Precompute sums and diff masks
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        
        diff_mask_A = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = abs(A[j] - A[i])
                diff_mask_A |= (1 << d)
        diff_mask_B = 0
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                d = abs(B[j] - B[i])
                diff_mask_B |= (1 << d)
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # Try to add to A
            for x in candidates:
                if x in A or x in B: # Sets can overlap, but if x in B, diff might conflict? 
                                     # Actually, if x in A and x in B, that's allowed.
                                     # But we must ensure Sidon and diff disjointness.
                    # If x is already in A, skip.
                    if x in A:
                        continue
                ok, new_diffs = can_add(x, A, sums_A, diff_mask_A, diff_mask_B)
                if ok:
                    A.append(x)
                    sums_A.add(2*x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                    for d in new_diffs:
                        diff_mask_A |= (1 << d)
                    improved = True
                    break
            
            # Try to add to B
            for x in candidates:
                if x in B:
                    continue
                ok, new_diffs = can_add(x, B, sums_B, diff_mask_B, diff_mask_A)
                if ok:
                    B.append(x)
                    sums_B.add(2*x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                    for d in new_diffs:
                        diff_mask_B |= (1 << d)
                    improved = True
                    break
                    
        return A, B

    best_score = 0
    best_sets = ([1], [2])
    
    # Strategy:
    # 1. Random partition of differences to guide greedy build.
    # 2. Local search to polish.
    
    for iteration in range(10000):
        if elapsed() > deadline - 10:
            break
            
        # Randomly assign each difference to A, B, or None
        p_a = 0.45
        p_b = 0.45
        
        mask_A = 0
        mask_B = 0
        
        for d in range(1, 256):
            r = random.random()
            if r < p_a:
                mask_A |= (1 << d)
            elif r < p_a + p_b:
                mask_B |= (1 << d)
            
        # Build A
        candidates_A = list(range(1, 257))
        random.shuffle(candidates_A)
        A = greedy_build(candidates_A, mask_B, 16)
        
        # Build B, avoiding diffs in A
        # We need to know A's actual diff mask to constrain B properly
        # The greedy_build used mask_B as the "other" mask, which was random.
        # Now we have A. Let's compute A's real diff mask.
        real_mask_A = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = abs(A[j] - A[i])
                real_mask_A |= (1 << d)
                
        # Now build B, constrained by real_mask_A
        candidates_B = list(range(1, 257))
        random.shuffle(candidates_B)
        B = greedy_build(candidates_B, real_mask_A, 16)
        
        if A and B:
            # Local search
            A2, B2 = local_search(A, B, time_limit=2.0)
            
            if is_valid_solution(A2, B2):
                score = len(A2) + len(B2)
                if score > best_score:
                    best_score = score
                    best_sets = (A2[:], B2[:])
                    
    # Fallback
    if best_score == 0:
        A = [1]
        B = [2]
        best_sets = (A, B)

    return list(best_sets)
