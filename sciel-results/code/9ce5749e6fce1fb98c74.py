"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 50.0  # Safety margin for 60s limit

    best_total = 0
    best_A = [1]
    best_B = [2]
    
    # Precompute nothing, but use efficient structures
    
    def greedy_fill(elements_order):
        """
        Greedy fill using incremental sum sets and bitmask differences.
        Returns (A, B, total_size)
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = 0  # Bitmask for differences in A
        diffs_B = 0  # Bitmask for differences in B
        
        for x in elements_order:
            # Try adding to A
            if len(A) < max_size:
                # Check Sidon condition for A with x
                valid_sidon_A = True
                # Check x+x
                if (x + x) in sums_A:
                    valid_sidon_A = False
                else:
                    # Check x+a for all a in A
                    for a in A:
                        if (x + a) in sums_A:
                            valid_sidon_A = False
                            break
                
                if valid_sidon_A:
                    # Check difference constraints
                    # Calculate new diffs for x with existing A
                    new_diffs_mask = 0
                    for a in A:
                        d = abs(a - x)
                        if d > 0:
                            new_diffs_mask |= (1 << d)
                    
                    if not (new_diffs_mask & diffs_B):
                        # Valid! Add x to A
                        A.append(x)
                        sums_A.add(x + x)
                        for a in A[:-1]: # a is the newly added x, so iterate old A
                            sums_A.add(x + a)
                        diffs_A |= new_diffs_mask
                        continue
            
            # Try adding to B
            if len(B) < max_size:
                # Check Sidon condition for B with x
                valid_sidon_B = True
                if (x + x) in sums_B:
                    valid_sidon_B = False
                else:
                    for b in B:
                        if (x + b) in sums_B:
                            valid_sidon_B = False
                            break
                
                if valid_sidon_B:
                    new_diffs_mask = 0
                    for b in B:
                        d = abs(b - x)
                        if d > 0:
                            new_diffs_mask |= (1 << d)
                    
                    if not (new_diffs_mask & diffs_A):
                        # Valid! Add x to B
                        B.append(x)
                        sums_B.add(x + x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                        diffs_B |= new_diffs_mask
        
        return A, B, len(A) + len(B)

    # Multiple randomized passes
    elements = list(range(1, n + 1))
    
    # We can run many more trials now due to speedup
    # Estimate: each trial is O(N * max_size) ~ 256 * 64 ~ 16k ops. 
    # 1000 trials ~ 16M ops. Should be fast in Python if optimized, but sets are a bit slow.
    # Let's aim for a few hundred to a thousand depending on time.
    
    trial_count = 0
    while time.time() < deadline:
        random.shuffle(elements)
        A, B, total = greedy_fill(elements)
        
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
        trial_count += 1
        # If we've run a lot and aren't improving, maybe break early? 
        # No, keep going until deadline.
        
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation (crucial to avoid invalid score 0)
    def validate(A, B):
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
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
        
        # Check disjoint nonzero differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
        
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
        
        if diffs_A & diffs_B:
            return False
        
        return True
    
    if not validate(best_A, best_B):
        # Fallback to simple valid solution
        # This set is known to be valid? Let's construct a simple one.
        # Powers of 2 are Sidon. Diffs are distinct.
        # A = {1, 2, 4, 8, 16, 32, 64, 128}
        # B = {3, 6, 12, 24, 48, 96}
        # Diffs of A: 1, 2, 3, 4, 6, 7, 8, ...
        # Diffs of B: 3, 6, 9, 12, 18, 24... 
        # Intersection: 3, 6... So this fallback is INVALID.
        
        # Let's generate a valid fallback dynamically or use a very sparse one.
        # A = [1, 3, 7, 15, 31, 63, 127] (2^k - 1)
        # Diffs: 2, 4, 8, 16, 32, 64...
        # B = [2, 6, 14, 30, 62, 126] (2*2^k - 2)
        # Diffs: 4, 8, 16... Intersection exists.
        
        # Safest fallback:
        A_fallback = [1, 2, 4, 8, 16, 32, 64, 128]
        B_fallback = [3, 5, 9, 17, 33, 65, 129] # 2^k + 1 for k>=1? No, 3,5,9...
        # Let's just pick disjoint large gaps.
        A_fb = [1, 10, 20, 30, 40, 50, 60, 70]
        B_fb = [2, 12, 22, 32, 42, 52, 62, 72]
        # Diffs A: 10, 20, 30...
        # Diffs B: 10, 20, 30... Intersection!
        
        # Try:
        A_fb = [1, 2, 4, 8, 16, 32, 64, 128]
        B_fb = [3, 6, 12, 24, 48, 96, 192] # 3*2^k
        # Diffs A: 1, 2, 3, 4, 6, 7, 8, 14, 15...
        # Diffs B: 3, 6, 9, 12, 18, 24...
        # Intersection: 3, 6.
        
        # Okay, let's use a brute force small valid set.
        A_fb = [1, 2, 4, 8, 16, 32, 64, 128]
        B_fb = [5, 10, 20, 40, 80, 160]
        # Diffs A: 1, 2, 3, 4, 6, 7, 8, 14, 15, 16, 30...
        # Diffs B: 5, 10, 15, 20, 30, 40, 60...
        # Intersection: 15, 30.
        
        # It is hard to hardcode a valid one without checking.
        # Let's run one more greedy pass with a fixed seed if validation fails.
        # Or just return a single element set if all else fails.
        # A=[1], B=[2] is always valid.
        best_A = [1]
        best_B = [2]
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
