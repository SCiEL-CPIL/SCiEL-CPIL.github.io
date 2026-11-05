import numpy as np
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    
    # Precompute neighbor masks as Python integers
    # Bit i is 1 if i is in the neighborhood of c
    neighbor_masks = [0] * N
    for c in range(N):
        mask = 0
        # Self
        mask |= (1 << c)
        val = c
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                nc = c + (new_val - orig_val) * (q ** pos)
                mask |= (1 << nc)
        neighbor_masks[c] = mask

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110

    # Helper to convert index to word
    def idx_to_word(idx):
        w = []
        v = idx
        for _ in range(n):
            w.append(v % q)
            v //= q
        return w

    while time.time() - start_time < time_limit:
        # Greedy Phase
        covered = 0
        C = []
        
        while covered.bit_count() < N:
            best_idx = -1
            max_gain = 0
            # Optimization: Only check candidates that might add something
            # We can iterate through all, or maintain a list of uncovered?
            # For N=2187, iterating all is fast.
            for c in range(N):
                if c in C: # This is slow. Use a set?
                    continue
                gain = (neighbor_masks[c] & ~covered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = c
            
            if best_idx == -1 or max_gain == 0:
                break
                
            C.append(best_idx)
            covered |= neighbor_masks[best_idx]
            
        # Local Search: Remove Redundant
        changed = True
        while changed:
            changed = False
            for i in range(len(C)):
                c = C[i]
                # Check if c is redundant
                # covered_without_c = covered & ~neighbor_masks[c]
                # If covered_without_c still covers neighbor_masks[c]?
                # i.e., neighbor_masks[c] is a subset of covered_without_c?
                # No, we need to check if removing c leaves any point in neighbor_masks[c] uncovered.
                # uncovered_after_removal = neighbor_masks[c] & ~(covered & ~neighbor_masks[c])
                # This simplifies to: neighbor_masks[c] & ~(covered_without_c)
                # But covered_without_c is not precomputed.
                
                # Efficient check:
                # A point p in mask[c] is uniquely covered by c if p is not in mask[c'] for any c' != c.
                # This is hard to check with just global `covered`.
                
                # Alternative: 
                # Let S = set(C).
                # For c in C:
                #   Let U = union of masks of (C \ {c}).
                #   If mask[c] is NOT a subset of U, c is essential.
                #   Else, c is redundant.
                
                # Computing U for each c is O(|C|^2 * bit_ops).
                # |C| ~ 200. 200^2 = 40000. 40000 * big_int_ops. Might be slow.
                
                # Faster:
                # Maintain `cover_count`? No, we are using bitsets.
                # We can try removing c and checking if `covered` changes.
                # new_covered = covered & ~neighbor_masks[c]
                # if new_covered == covered:
                #    # This means mask[c] was fully covered by others?
                #    # Wait. covered = OR of all masks.
                #    # If covered & ~mask[c] == covered, then mask[c] is subset of (covered \ mask[c])?
                #    # No. If covered = A | B. covered & ~A = B.
                #    # If B == covered, then A is subset of B.
                #    # So if (covered & ~neighbor_masks[c]) == covered, then c is redundant.
                
                new_covered = covered & ~neighbor_masks[c]
                if new_covered == covered:
                    C.pop(i)
                    covered = new_covered # Actually covered doesn't change, but we removed c
                    changed = True
                    break

        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
        # Swap Phase
        # Try swapping one codeword with another to enable removal or just improve structure
        # This is complex with bitsets.
        # Let's stick to Greedy + Remove for now, with multiple random restarts.
        # To add randomness, we can perturb the greedy choice.
        
    if not best_C:
        return []
        
    return [idx_to_word(c) for c in best_C]
