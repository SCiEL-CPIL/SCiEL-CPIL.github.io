"""Improved program: CSP with forward checking and iterative bound tightening."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute all arithmetic progressions
    # An AP is defined by (start, step, length) where indices are start, start+step, ..., start+(length-1)*step
    # We need all APs contained in [0, n-1] (0-indexed)
    # For each step d >= 1 and start s in [0, d-1], the AP is s, s+d, s+2d, ... up to < n
    
    # Build list of APs as lists of indices
    aps = []
    # For each position, track which APs contain it (for forward checking)
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for start in range(0, d):
            if start >= n:
                break
            idxs = []
            idx = start
            while idx < n:
                idxs.append(idx)
                idx += d
            if len(idxs) >= 1:  # Include singletons
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    pos_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # We'll try to find a sign assignment with max absolute sum <= B
    # Start with B=3 and increase if infeasible
    
    # For efficiency, we'll use a CSP approach with forward checking
    # State: signs array, partial sums for each AP
    
    def try_solve(B, rng):
        """Try to find a sign assignment with all AP sums in [-B, B]."""
        signs = [0] * n  # 0 means unassigned
        ap_sums = [0] * num_aps
        ap_counts = [0] * num_aps  # How many signs assigned in each AP
        
        # Order positions by most constrained first (most APs they belong to)
        order = list(range(n))
        order.sort(key=lambda i: -len(pos_to_aps[i]))
        
        # Use backtracking with forward checking
        def backtrack(pos_idx):
            if time.time() - start_time > time_limit - 2.0:
                return False
            
            if pos_idx == n:
                return True
            
            idx = order[pos_idx]
            
            for sign in (-1, 1):
                # Check if assigning this sign keeps all APs within bounds
                feasible = True
                for ap_id in pos_to_aps[idx]:
                    new_sum = ap_sums[ap_id] + sign
                    remaining = len(aps[ap_id]) - ap_counts[ap_id] - 1
                    # After this assignment, we can still add at most 'remaining' more signs
                    # The final sum must be in [-B, B]
                    # Current sum after this assignment is new_sum
                    # We need: -B <= new_sum + (some combination of remaining signs) <= B
                    # The remaining signs can contribute between -remaining and +remaining
                    # So we need: -B <= new_sum - remaining AND new_sum + remaining <= B
                    if new_sum - remaining > B or new_sum + remaining < -B:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                # Assign
                signs[idx] = sign
                for ap_id in pos_to_aps[idx]:
                    ap_sums[ap_id] += sign
                    ap_counts[ap_id] += 1
                
                if backtrack(pos_idx + 1):
                    return True
                
                # Unassign
                signs[idx] = 0
                for ap_id in pos_to_aps[idx]:
                    ap_sums[ap_id] -= sign
                    ap_counts[ap_id] -= 1
            
            return False
        
        if backtrack(0):
            return signs
        return None
    
    # Try increasing bounds
    for B in range(1, 8):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        rng = random.Random(B * 1000 + 42)
        result = try_solve(B, rng)
        
        if result is not None:
            # Verify the result
            max_abs = 0
            for ap in aps:
                s = sum(result[i] for i in ap)
                abs_s = abs(s)
                if abs_s > max_abs:
                    max_abs = abs_s
            
            # Return the result
            return result
    
    # If we couldn't find a solution within time, return a random assignment
    # as a fallback
    rng = random.Random(12345)
    return [rng.choice((-1, 1)) for _ in range(n)]

# EVOLVE-BLOCK-END
