import random
import numpy as np
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    rng = random.Random(42)
    
    # 1. Precompute APs
    # An AP is defined by (start, step, length)
    # Indices: start, start+step, ..., start+(length-1)*step
    # We store the list of 0-based indices for each AP.
    ap_indices_list = []
    ap_end_idx = [] # The last index (0-based) of each AP
    ap_pos_maps = [[] for _ in range(N)] # pos -> list of AP indices containing pos
    
    ap_id = 0
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            # Max length
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                # Store 0-based indices
                indices_0 = [x - 1 for x in indices]
                ap_indices_list.append(indices_0)
                last_idx = indices_0[-1]
                ap_end_idx.append(last_idx)
                
                for idx in indices_0:
                    ap_pos_maps[idx].append(ap_id)
                ap_id += 1
    
    num_aps = ap_id
    
    # 2. Greedy Construction
    # We want to build x[0..N-1]
    # State: current sums for each AP
    # We only need to track sums for APs that are "active" or "completed".
    # Actually, we can just maintain an array `ap_sums` of length num_aps.
    # Initially 0.
    
    def greedy_construct():
        x = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        
        # Track the max absolute sum of all APs that have been "closed" (ended) so far.
        # An AP is closed when we reach its end index.
        max_closed_sum = 0
        
        for i in range(N):
            # Try x[i] = 1 and x[i] = -1
            best_val = 0
            best_max = float('inf')
            
            for val in [1, -1]:
                # Calculate the new max if we set x[i] = val
                # The APs affected are those in ap_pos_maps[i]
                # For each affected AP, the new sum is ap_sums[ap] + val
                
                # We need to check the max absolute sum of:
                # 1. APs that end at i (become closed)
                # 2. APs that were already closed (max_closed_sum)
                
                # Note: APs that don't end at i remain "open" or "closed".
                # If an AP is closed, its sum is fixed.
                # If an AP is open, its sum is partial.
                # The greedy heuristic from the prompt: "minimizes the maximum absolute sum over all partially completed arithmetic progressions ending at that position".
                # This implies we focus on APs ending at i.
                
                new_max_candidate = max_closed_sum
                
                # We need to update the sums for the "virtual" state to check the ends
                # To avoid modifying ap_sums permanently until we choose,
                # we can just calculate the new sums for the specific APs ending at i.
                
                # Identify APs ending at i
                # We can precompute a list of APs ending at each index for faster access
                # But ap_pos_maps[i] contains ALL APs containing i.
                # We need to filter for those where i is the LAST element.
                
                # Optimization: Precompute ap_end_maps
                # Let's do that outside.
                
                pass # Placeholder for logic
                
            # This naive approach is slow if we don't precompute end maps.
            # Let's restructure.
            
        return x, ap_sums

    # Precompute ap_end_maps: index -> list of AP ids ending there
    ap_end_maps = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        last = ap_indices_list[ap_idx][-1]
        ap_end_maps[last].append(ap_idx)

    def greedy_construct_v2():
        x = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        max_closed_sum = 0
        
        for i in range(N):
            best_val = 1
            best_score = float('inf')
            
            for val in [1, -1]:
                # Calculate potential new max
                current_max = max_closed_sum
                
                # Check APs ending at i
                for ap_idx in ap_end_maps[i]:
                    # The sum of this AP will be ap_sums[ap_idx] + val
                    # (Because ap_sums currently holds the sum of x[0..i-1] for this AP)
                    new_sum = ap_sums[ap_idx] + val
                    abs_sum = abs(new_sum)
                    if abs_sum > current_max:
                        current_max = abs_sum
                
                # Heuristic: Just minimize the new max of closed APs.
                # We could also add a term for open APs to balance them, 
                # but keeping closed ones small is the primary constraint.
                if current_max < best_score:
                    best_score = current_max
                    best_val = val
            
            x[i] = best_val
            max_closed_sum = best_score
            
            # Update ap_sums for all APs containing i
            for ap_idx in ap_pos_maps[i]:
                ap_sums[ap_idx] += best_val
                
        return x, ap_sums

    # 3. Local Search Refinement
    # We start with the greedy result and improve it.
    
    def local_search(x, ap_sums, end_time):
        # Build histogram of absolute sums
        abs_sums = np.abs(ap_sums)
        hist = np.zeros(N + 1, dtype=np.int32)
        np.add.at(hist, abs_sums, 1)
        current_max = int(np.max(abs_sums))
        
        temp = 2.0
        cooling_rate = 0.999
        
        while time.time() < end_time:
            # Pick a random position to flip
            j = rng.randint(0, N - 1)
            
            old_val = x[j]
            delta = -2 * old_val
            
            affected = ap_pos_maps[j]
            
            old_abs = np.abs(ap_sums[affected])
            new_sums = ap_sums[affected] + delta
            new_abs = np.abs(new_sums)
            
            # Update histogram
            np.add.at(hist, old_abs, -1)
            np.add.at(hist, new_abs, 1)
            
            # Determine new max
            if hist[current_max] == 0:
                # Scan down
                new_max = current_max - 1
                while new_max > 0 and hist[new_max] == 0:
                    new_max -= 1
                # Check if any new sum exceeds this
                max_new_in_affected = int(np.max(new_abs))
                if max_new_in_affected > new_max:
                    new_max = max_new_in_affected
            else:
                new_max = current_max
                max_new_in_affected = int(np.max(new_abs))
                if max_new_in_affected > new_max:
                    new_max = max_new_in_affected
            
            cost_diff = new_max - current_max
            
            accept = False
            if cost_diff <= 0:
                accept = True
            else:
                if temp > 0.01:
                    p = math.exp(-cost_diff / temp)
                    if rng.random() < p:
                        accept = True
            
            if accept:
                x[j] = -old_val
                ap_sums[affected] = new_sums
                current_max = new_max
            else:
                # Revert histogram
                np.add.at(hist, new_abs, -1)
                np.add.at(hist, old_abs, 1)
            
            temp *= cooling_rate
            
        return x, current_max

    # Main Loop: Multi-start
    best_seq = None
    best_score = float('inf')
    
    # Run greedy construction once (it's deterministic if rng is fixed, but we can randomize the tie-breaking)
    # To make it multi-start, we can randomize the choice when scores are tied, or add noise.
    # For now, let's run a few variants.
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
            
        # Slight perturbation of greedy: if tied, pick randomly
        x, ap_sums = greedy_construct_v2()
        
        # Calculate initial score
        abs_sums = np.abs(ap_sums)
        current_max = int(np.max(abs_sums))
        
        if current_max < best_score:
            best_score = current_max
            best_seq = x.copy()
        
        # Refine with Local Search
        # Allocate remaining time / restarts
        time_per_restart = (deadline - time.time()) / max(1, 20 - restart)
        end_time = time.time() + time_per_restart * 0.8 # Leave some buffer
        
        x, current_max = local_search(x, ap_sums, end_time)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = x.copy()
            
    # Final polish on the best found
    if best_seq is not None and time.time() < deadline - 2.0:
        x = best_seq.copy()
        # Recalculate sums
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        for i in range(N):
            for ap_idx in ap_pos_maps[i]:
                ap_sums[ap_idx] += x[i]
        
        end_time = deadline - 0.5
        x, current_max = local_search(x, ap_sums, end_time)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = x.copy()

    return [int(v) for v in best_seq]
