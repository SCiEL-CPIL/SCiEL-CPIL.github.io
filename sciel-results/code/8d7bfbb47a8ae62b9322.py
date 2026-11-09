import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0
    
    # Precompute all APs (arithmetic progressions)
    # AP defined by start s (0-indexed) and step d >= 1
    # Indices: s, s+d, s+2d, ... < n
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for s in range(n):
            idxs = list(range(s, n, d))
            if len(idxs) >= 1:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Precompute AP index arrays for numpy operations
    ap_index_arrays = [np.array(idxs, dtype=np.int32) for idxs in aps]
    
    # Initialize with Thue-Morse sequence
    signs = np.array([(-1) ** (bin(i).count('1') % 2) for i in range(n)], dtype=np.int8)
    
    # Compute initial AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for k in range(num_aps):
        ap_sums[k] = np.sum(signs[ap_index_arrays[k]])
    
    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(12345)
    
    # Simulated annealing parameters
    T = 10.0
    cooling_rate = 0.99995
    min_T = 0.01
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 50000
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if T < min_T:
            break
        
        # Pick random index to flip
        i = rng.randrange(n)
        old_sign = signs[i]
        delta = -2 * old_sign
        
        affected_aps = index_to_aps[i]
        
        if not affected_aps:
            continue
        
        # Get old sums for affected APs
        old_vals = ap_sums[affected_aps].copy()
        new_vals = old_vals + delta
        
        # Compute new max
        # The new max is max of:
        # 1. New abs values of affected APs
        # 2. Old max if it's not in the affected APs (or if affected APs still have it)
        
        new_affected_max = int(np.max(np.abs(new_vals)))
        
        # Check if current_max is still achievable
        # If no affected AP had abs sum == current_max, then current_max is still valid
        # Otherwise, we need to recompute
        
        had_max = False
        for idx_ap in affected_aps:
            if abs(ap_sums[idx_ap]) == current_max:
                had_max = True
                break
        
        if had_max:
            # Need to find new global max
            # It's max(new_affected_max, max of unaffected APs)
            # Unaffected APs have sums <= current_max
            # If new_affected_max >= current_max, new max is new_affected_max
            # Otherwise, new max is max(new_affected_max, current_max) but only if some unaffected has current_max
            # Actually, if had_max is True, some affected AP had current_max
            # After update, those APs have new values
            # So the new max is max(new_affected_max, max_unaffected)
            # max_unaffected <= current_max
            # If new_affected_max >= current_max: new_max = new_affected_max
            # Else: new_max = current_max (since some unaffected might still have it, or new_affected_max)
            # Wait, if had_max is True, it means at least one affected AP had abs sum == current_max
            # After flipping, that AP has a new value
            # The new global max is max(new_affected_max, max of all unaffected APs)
            # We don't track max of unaffected separately, but we know it's <= current_max
            # So new_max = max(new_affected_max, current_max) if there exists an unaffected AP with abs sum == current_max
            # Or new_max = new_affected_max if new_affected_max > current_max
            # This is getting complex. Let's just recompute if needed.
            
            # Efficient check: is there any AP (affected or not) with abs sum >= new_affected_max?
            # If new_affected_max > current_max: new_max = new_affected_max
            # If new_affected_max <= current_max: 
            #   Check if any unaffected AP has abs sum == current_max
            #   If yes: new_max = current_max
            #   If no: new_max = max(new_affected_max, second_highest)
            
            if new_affected_max > current_max:
                actual_new_max = new_affected_max
            else:
                # Check if any unaffected AP has abs sum == current_max
                # We can check by seeing if the total count of APs with abs sum == current_max 
                # minus the count in affected APs is > 0
                # But we don't track counts easily.
                # Let's just do a quick check: scan a few unaffected APs?
                # No, that's O(n). Let's just recompute the max from all APs if needed.
                # But that's O(num_aps) which is ~30000. Too slow per step.
                
                # Alternative: track the count of APs at each abs sum level
                # Let's add a frequency array
                
                # For now, let's just compute the new max by checking:
                # The new max is the max of all AP sums after update
                # We can compute this as:
                # max(max of unaffected, new_affected_max)
                # We know max of unaffected <= current_max
                # And we know that before update, max was current_max
                # After update, the affected APs have new values
                # So the new max is max(new_affected_max, max_unaffected)
                # We don't know max_unaffected exactly, but it's <= current_max
                # And it's >= the second highest value overall (before update)
                
                # Let's just recompute if new_affected_max < current_max
                # This happens when we're trying to reduce the max
                # In that case, we need to know if any AP still has abs sum == current_max
                
                # Quick check: count how many affected APs had abs sum == current_max
                count_at_max_affected = sum(1 for v in old_vals if abs(v) == current_max)
                
                # We don't know the total count at max. Let's assume it's small.
                # If count_at_max_affected is the only ones at max, then max drops.
                # Otherwise, max stays.
                
                # This is hard to track without a frequency array.
                # Let's add a frequency array for abs sums.
                
                # For now, let's just recompute the max from scratch if new_affected_max < current_max
                # This is O(num_aps) but only happens when we're reducing the max, which is rare.
                
                # Actually, let's just compute the new max properly:
                # new_max = max(new_affected_max, max of all unaffected APs)
                # We can compute max of unaffected by:
                #   If we had a frequency array, we'd know
                #   Without it, we just have to check
                
                # Let's use a simpler approach: just compute the new max by taking the max of
                # all AP sums after the update. But that's O(num_aps).
                
                # Alternative: maintain a max-heap or sorted list of AP sums.
                # But that's complex to update.
                
                # Let's just recompute when needed. It's O(num_aps) = ~30000 operations.
                # This happens only when new_affected_max < current_max, which is when we're
                # trying to reduce the max. This should be rare in the later stages.
                
                # For now, let's just set actual_new_max = max(new_affected_max, current_max)
                # This is an upper bound. If it's wrong, we'll fix it later.
                # Actually, this is incorrect. Let's just recompute.
                
                # Recompute max from all APs
                # Update the affected APs first
                ap_sums[affected_aps] = new_vals
                actual_new_max = int(np.max(np.abs(ap_sums)))
                
                # Revert for now if we reject
                # We'll handle revert separately
                
        else:
            # No affected AP had the max, so max is unchanged or increased
            if new_affected_max > current_max:
                actual_new_max = new_affected_max
            else:
                actual_new_max = current_max
        
        # Cost difference
        cost_diff = actual_new_max - current_max
        
        # Acceptance
        if cost_diff < 0:
            accept = True
        else:
            if T < 1e-5:
                accept = False
            else:
                if cost_diff / T > 50:
                    accept = False
                else:
                    prob = math.exp(-cost_diff / T)
                    accept = rng.random() < prob
        
        if accept:
            signs[i] = -signs[i]
            ap_sums[affected_aps] = new_vals
            current_max = actual_new_max
            
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # Revert: ap_sums already has old values if we didn't update, or we need to revert
            # In the case where we recomputed, we already updated ap_sums
            # Let's handle this more carefully
            
            # If we had to recompute, ap_sums[affected_aps] is already set to new_vals
            # We need to revert it
            if had_max and new_affected_max < current_max:
                ap_sums[affected_aps] = old_vals
            # Otherwise, ap_sums was not updated (we only computed actual_new_max)
            # Wait, in the had_max case where new_affected_max < current_max, we did:
            #   ap_sums[affected_aps] = new_vals
            #   actual_new_max = int(np.max(np.abs(ap_sums)))
            # So we need to revert
            # In other cases, we didn't update ap_sums, so no revert needed
        
        T *= cooling_rate
        steps += 1
        
        if no_improve_steps > max_no_improve and T < 1.0:
            # Restart with a different initial solution
            # Keep best_signs and try again with different random seed
            break
    
    return best_signs.tolist()
