"""Program for the suite task; uses Sidon-based construction with greedy extension."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 512
    max_card = 48
    
    # Strategy:
    # 1. Generate a strong Sidon (B2) sequence base to guarantee no 3-term AP (k=2 case).
    # 2. Greedily extend the set by checking all subset mean conditions using bitset DP.
    # 3. To maximize cardinality, we try to pack as many numbers as possible.
    
    # Step 1: Generate a Sidon sequence within [1, 512].
    # A simple construction for Sidon sequences is x_i = i*B + i^2.
    # We need x_i <= 512.
    # Let's find the largest B such that we can fit many terms.
    # For i up to ~20, i^2 is ~400. So B can be small.
    # Alternatively, we can just greedily pick a Sidon set.
    
    # Let's generate a Sidon set greedily.
    # A set S is Sidon if all pairwise sums s_i + s_j (i <= j) are distinct.
    # This implies no 3-term AP: s_i + s_k = 2*s_j => s_i + s_k = s_j + s_j.
    # If sums are distinct, then s_i + s_k != s_j + s_j unless (i,k) = (j,j) which implies i=j=k.
    # So a Sidon set has no 3-term AP.
    
    sidon = []
    used_sums = set()
    
    # We will try to build a Sidon set of size ~30-40 to start with.
    # Then we will extend it.
    # Actually, let's just build the final set greedily, but with a bias towards Sidon properties initially?
    # Or just build a large Sidon set and then try to add more?
    
    # Let's try to build a Sidon set of size 40 first.
    for x in range(1, 513):
        # Check if x can be added to sidon
        ok = True
        for y in sidon:
            s = x + y
            if s in used_sums:
                ok = False
                break
        if ok:
            # Also check x+x
            if (x + x) in used_sums:
                ok = False
            else:
                # Add all new sums
                for y in sidon:
                    used_sums.add(x + y)
                used_sums.add(x + x)
                sidon.append(x)
        
        if len(sidon) >= 35:  # Let's start with a base of 35
            break
    
    # Now we have a Sidon set `sidon`. It has no 3-term AP.
    # We need to check if it satisfies the full condition (no element is mean of any subset of size >= 2).
    # And then try to extend it to 48.
    
    S = sidon[:]
    
    # Helper function to check if a target sum is achievable by a non-empty subset of diffs
    # using bitset DP.
    # diffs: list of integers (can be negative)
    # target: integer
    # Returns True if there exists a non-empty subset of diffs summing to target.
    
    def is_non_empty_subset_sum(diffs, target):
        if not diffs:
            return False
        
        # Determine range of possible sums
        min_sum = sum(d for d in diffs if d < 0)
        max_sum = sum(d for d in diffs if d > 0)
        
        # We need to check if target is in range
        if target < min_sum or target > max_sum:
            return False
        
        # Offset to make indices non-negative
        offset = -min_sum
        size = max_sum - min_sum + 1
        
        # Bitset representation
        # bit i represents that sum (i - offset) is achievable
        all_sums = 1  # bit 0 set, representing sum 0 (empty set)
        non_empty_sums = 0
        
        for d in diffs:
            if d == 0:
                # Adding 0 doesn't change sums, but allows forming non-empty subset
                # if there was already a non-empty subset, it remains.
                # If there wasn't, now we have {0} as a non-empty subset summing to 0.
                non_empty_sums |= all_sums
                # all_sums remains the same
                continue
            
            if d > 0:
                # Shift left by d
                new_all = all_sums << d
                new_non_empty = non_empty_sums << d
                # The new sums formed by adding d to any existing sum
                # Also, the single element d forms a non-empty subset
                new_non_empty |= (1 << (d + offset))
                
                all_sums |= new_all
                non_empty_sums |= new_non_empty
            else:
                # d < 0, shift right by -d
                shift = -d
                new_all = all_sums >> shift
                new_non_empty = non_empty_sums >> shift
                # The single element d forms a non-empty subset
                new_non_empty |= (1 << (d + offset))
                
                all_sums |= new_all
                non_empty_sums |= new_non_empty
        
        # Check if target is achievable by a non-empty subset
        target_idx = target + offset
        if 0 <= target_idx < size:
            if (non_empty_sums >> target_idx) & 1:
                return True
        return False
    
    # Now, greedily extend S from the Sidon base.
    # We iterate x from 1 to 512.
    # If x is not in S, check if adding x is valid.
    
    S_set = set(S)
    
    for x in range(1, 513):
        if x in S_set:
            continue
        
        valid = True
        
        # Check 1: Is x the mean of any subset of S of size >= 2?
        # Equivalent to: 0 is a subset sum of {s - x for s in S} using a non-empty subset.
        # Note: Since S is Sidon, no 3-term AP, so no subset of size 2 sums to 2x.
        # But we must check larger subsets.
        
        if S:
            diffs = [s - x for s in S]
            if is_non_empty_subset_sum(diffs, 0):
                valid = False
        
        if valid:
            # Check 2: For each a in S, is a the mean of a subset of S U {x} excluding a?
            # New subsets must contain x.
            # T = T0 U {x}, T0 subset S\{a}.
            # Condition: sum(T0) + x = (|T0|+1)a => sum(t - a for t in T0) = a - x.
            # T0 must be non-empty.
            for a in S:
                diffs_a = [s - a for s in S if s != a]
                target = a - x
                if is_non_empty_subset_sum(diffs_a, target):
                    valid = False
                    break
        
        if valid:
            S.append(x)
            S_set.add(x)
            if len(S) >= max_card:
                break
    
    return S
# EVOLVE-BLOCK-END
