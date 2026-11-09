import time
import random
import sys

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer

    # 1. Precompute APs
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        max_len = (n - 1) // d + 1
        for L in range(1, max_len + 1):
            max_start = n - (L - 1) * d
            for a in range(1, max_start + 1):
                start_idx = a - 1
                indices = [start_idx + k * d for k in range(L)]
                ap_id = len(ap_indices)
                ap_indices.append(indices)
                for idx in indices:
                    index_to_aps[idx].append(ap_id)

    num_aps = len(ap_indices)
    
    # 2. Initialize with Thue-Morse
    s = [1] * n
    def popcount(x):
        count = 0
        while x:
            count += 1
            x &= x - 1
        return count
        
    for i in range(n):
        s[i] = 1 if popcount(i) % 2 == 0 else -1

    # 3. Compute initial sums
    ap_sums = [0] * num_aps
    for ap_id, indices in enumerate(ap_indices):
        total = 0
        for idx in indices:
            total += s[idx]
        ap_sums[ap_id] = total

    # 4. Objective: Minimize max absolute sum.
    # We use a "bucket" approach to track the maximum sum efficiently.
    # Sums range from -n to n.
    # We maintain a count of APs at each absolute sum value.
    # Max possible abs sum is n.
    sum_counts = [0] * (n + 1)
    for val in ap_sums:
        abs_val = abs(val)
        sum_counts[abs_val] += 1
        
    current_max = n
    while current_max > 0 and sum_counts[current_max] == 0:
        current_max -= 1
        
    best_s = s[:]
    best_max = current_max
    
    rng = random.Random(42)
    
    # Local Search: Simulated Annealing / Hill Climbing
    # We try to reduce the maximum discrepancy.
    # If we can't reduce the max, we try to reduce the count of APs at the max, or reduce the second max.
    
    # To speed up, we prioritize flipping indices that are in APs with high sums.
    # We'll maintain a list of "hot" indices.
    
    # Optimization: Precompute a list of indices sorted by "degree" (number of APs they are in)
    # or just pick randomly from a set of "active" indices (those in APs with |sum| >= current_max - 1).
    
    # Let's stick to a robust hill-climbing with random restarts.
    # At each step:
    # 1. Pick a random index.
    # 2. Calculate delta in "cost".
    #    Cost = (current_max, count_at_max)
    #    We want to minimize this tuple.
    
    # To make it faster, we can approximate the cost change.
    # If flipping an index reduces the number of APs at the current maximum, it's good.
    # If it creates new APs at the current maximum, it's bad.
    
    # Let's implement a simple loop.
    # To handle the 60s limit, we monitor time.
    
    while time.time() - start_time < time_limit:
        # Pick a random index
        idx = rng.randint(0, n - 1)
        
        old_sign = s[idx]
        new_sign = -old_sign
        
        # Calculate change in sums for affected APs
        affected_aps = index_to_aps[idx]
        
        # We need to check if this flip improves the "state".
        # State is defined by the distribution of absolute sums.
        # Let's track:
        # - max_sum
        # - count_max (number of APs with |sum| == max_sum)
        
        # To avoid full recount, we track changes.
        # Let's just calculate the new max and count_max if we were to flip.
        # This might be expensive if we do it naively.
        
        # Alternative: Just flip and update sum_counts.
        # If it improves, keep. If not, revert.
        
        # Let's try this simple approach first. It's O(degree(idx)) per step.
        
        # 1. Remove old contributions from sum_counts
        for ap_id in affected_aps:
            old_sum = ap_sums[ap_id]
            abs_old = abs(old_sum)
            sum_counts[abs_old] -= 1
            # Update ap_sums
            new_sum = old_sum - 2 * old_sign
            ap_sums[ap_id] = new_sum
            abs_new = abs(new_sum)
            sum_counts[abs_new] += 1
            
        # 2. Find new max
        new_max = current_max
        if sum_counts[new_max] == 0:
            while new_max > 0 and sum_counts[new_max] == 0:
                new_max -= 1
        
        # 3. Compare (new_max, sum_counts[new_max]) vs (current_max, sum_counts[current_max])
        # Note: sum_counts[current_max] might have changed if new_max == current_max.
        
        # We need to be careful with the comparison if max changed.
        # Let's store the old state properly.
        
        # Actually, the above modification is destructive. We need to revert if it's bad.
        # Let's store the affected APs' old sums and restore them if needed.
        
        # Wait, I already updated ap_sums and sum_counts.
        # If I decide to revert, I need to undo it.
        
        # Let's refine the logic:
        # 1. Calculate delta in sum_counts for affected APs.
        # 2. Determine if the new state (max, count_max) is better.
        # 3. If better, accept. Else, revert.
        
        # To do this without reverting, I can simulate the change.
        
        # Let's restart the loop logic with simulation.
        
        # Simulation:
        # For each affected AP:
        #   old_abs = abs(ap_sums[ap_id])
        #   new_abs = abs(ap_sums[ap_id] - 2*old_sign)
        #   sum_counts[old_abs] -= 1
        #   sum_counts[new_abs] += 1
        
        # Then find new_max.
        # If new_max < current_max OR (new_max == current_max AND sum_counts[new_max] < sum_counts[current_max]):
        #    Accept (apply changes to ap_sums and sum_counts)
        # Else:
        #    Revert changes to sum_counts.
        
        # This is O(degree(idx)) for simulation + O(degree(idx)) for application.
        # O(degree(idx)) is roughly n/d * ... average degree is manageable.
        
        # Let's implement this.
        
        # 1. Simulate
        for ap_id in affected_aps:
            old_sum = ap_sums[ap_id]
            old_abs = abs(old_sum)
            sum_counts[old_abs] -= 1
            new_sum = old_sum - 2 * old_sign
            new_abs = abs(new_sum)
            sum_counts[new_abs] += 1
            
        new_max = current_max
        if sum_counts[new_max] == 0:
            while new_max > 0 and sum_counts[new_max] == 0:
                new_max -= 1
                
        # Compare
        old_count_max = sum_counts[current_max] # This is the count AFTER simulation if new_max == current_max?
        # No, sum_counts has been modified.
        # We need the count at current_max BEFORE simulation to compare properly?
        # Actually, we want to minimize (max, count_at_max).
        # Current state: (current_max, old_count_at_max_before_sim)
        # New state: (new_max, sum_counts[new_max])
        
        # Wait, I lost the old_count_at_max_before_sim.
        # Let's store it before simulation.
        
        # Let's rewrite the loop body cleanly.
        
        # Revert simulation first to start clean for the next iteration if we don't accept?
        # No, let's just revert if we don't accept.
        
        # But I need to know if I should accept.
        
        # Let's try a different structure:
        # 1. Store old state of affected APs.
        # 2. Apply flip.
        # 3. Check if improved.
        # 4. If not, revert.
        
        # This is simpler.
        
        # Revert the simulation I just did (since I didn't accept yet).
        for ap_id in affected_aps:
            old_sum = ap_sums[ap_id]
            old_abs = abs(old_sum)
            sum_counts[old_abs] -= 1
            new_sum = old_sum + 2 * old_sign # Undo the -2*old_sign
            ap_sums[ap_id] = new_sum
            new_abs = abs(new_sum)
            sum_counts[new_abs] += 1
            
        # Okay, that was a mess. Let's restart the loop logic properly.
        
        # The code above is getting messy. I will replace the entire while loop block with a cleaner implementation.
        
        pass # Placeholder to stop execution of this broken thought process

    # This code is getting too complex to patch in place. I will rewrite the entire construct function in the REPLACE block.
    # For now, return the initial Thue-Morse to ensure validity.
    return s
