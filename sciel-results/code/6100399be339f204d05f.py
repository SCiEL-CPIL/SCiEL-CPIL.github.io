import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 110
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            aps.append(idxs)
            
    aps = np.array(aps, dtype=np.int32)
    # aps shape: (M, 4)
    
    # Build adjacency lists for delta calculation
    # aps_at[i] contains indices of other elements in APs that include i
    # We also need to know the "parity" or role? No, just the set of other 3.
    # For delta of flip i:
    # For each AP containing i, let others be the other 3.
    # If others are all same color c:
    #   if i is c, delta -= 1 (breaks mono)
    #   if i is 1-c, delta += 1 (creates mono)
    
    # Efficient delta calculation:
    # Let cols be 0/1.
    # For a specific AP (a,b,c,d), mono if all same.
    # We can precompute for each position i, the list of (other1, other2, other3).
    
    neighbors = [[] for _ in range(N)]
    for ap in aps:
        a, b, c, d = ap
        neighbors[a].append((b, c, d))
        neighbors[b].append((a, c, d))
        neighbors[c].append((a, b, d))
        neighbors[d].append((a, b, c))
        
    # Convert to numpy arrays for fast access?
    # It's tricky because lists have different lengths.
    # We can store flat arrays and offsets.
    
    total_neighbors = sum(len(n) for n in neighbors)
    neighbor_flat = np.zeros(total_neighbors * 3, dtype=np.int32)
    neighbor_offsets = np.zeros(N + 1, dtype=np.int32)
    
    ptr = 0
    for i in range(N):
        neighbor_offsets[i] = ptr
        for j, (u, v, w) in enumerate(neighbors[i]):
            neighbor_flat[ptr*3 : ptr*3+3] = [u, v, w]
            ptr += 1
    neighbor_offsets[N] = ptr
    
    def count_mono(cols):
        # cols is numpy array of 0/1
        # Check if all 4 in each AP are equal
        # cols[aps] shape (M, 4)
        # mono if min == max or all equal
        # Faster: (cols[aps[:,0]] == cols[aps[:,1]]) & (cols[aps[:,1]] == cols[aps[:,2]]) & (cols[aps[:,2]] == cols[aps[:,3]])
        c0 = cols[aps[:,0]]
        c1 = cols[aps[:,1]]
        c2 = cols[aps[:,2]]
        c3 = cols[aps[:,3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return np.sum(mono)
        
    def calc_deltas(cols):
        """
        Calculate delta for flipping each bit i.
        delta[i] = new_cost - old_cost if we flip i.
        """
        # For each i, look at its neighbors.
        # For a neighbor (u,v,w) of i:
        # If cols[u]==cols[v]==cols[w]:
        #    if cols[i] == cols[u]: delta -= 1
        #    else: delta += 1
        # Else: delta += 0
        
        deltas = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            start = neighbor_offsets[i]
            end = neighbor_offsets[i+1]
            if start == end:
                continue
            
            # Get neighbors for i
            # neighbor_flat[start*3 : end*3].reshape(-1, 3)
            nbrs = neighbor_flat[start*3 : end*3].reshape(-1, 3)
            
            # Get colors of neighbors
            c_u = cols[nbrs[:,0]]
            c_v = cols[nbrs[:,1]]
            c_w = cols[nbrs[:,2]]
            
            # Check if u,v,w are all same
            all_same = (c_u == c_v) & (c_v == c_w)
            
            if np.any(all_same):
                # For those where all_same is True:
                # If cols[i] == c_u, delta -= 1, else delta += 1
                # delta_contribution = -1 if cols[i]==c_u else +1
                # This is equivalent to: 2 * (cols[i] != c_u) - 1 ?
                # If cols[i]==c_u (True), 2*0 - 1 = -1.
                # If cols[i]!=c_u (False), 2*1 - 1 = +1.
                
                # We only care about the subset where all_same is True
                mask = all_same
                c_ref = c_u[mask]
                # cols[i] is scalar
                # delta = sum( 2*(cols[i] != c_ref) - 1 )
                # = 2 * count(cols[i] != c_ref) - count(all_same)
                # count(cols[i] != c_ref) = count(c_ref != cols[i])
                
                num_same_as_i = np.sum(c_ref == cols[i])
                total_affected = np.sum(mask)
                
                # If cols[i] is 0, and c_ref is 0, they are same.
                # delta = -1 for each same, +1 for each different.
                # delta = -1 * num_same_as_i + 1 * (total_affected - num_same_as_i)
                # delta = total_affected - 2 * num_same_as_i
                
                deltas[i] = total_affected - 2 * num_same_as_i
                
        return deltas

    def greedy_polish(cols, current_cost):
        improved = True
        while improved:
            improved = False
            deltas = calc_deltas(cols)
            # Find best negative delta
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            if min_val < 0:
                cols[min_idx] = 1 - cols[min_idx]
                current_cost += min_val
                improved = True
        return cols, current_cost

    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit):
        cols = np.array(initial_colors, dtype=np.int32)
        current_cost = count_mono(cols)
        best_cost = current_cost
        best_cols = cols.copy()
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Choose move: single flip or double flip
            # 10% chance of double flip
            if random.random() < 0.1:
                # Pick two distinct indices
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                
                # Calculate delta for flipping i and j
                # delta = delta_i + delta_j + interaction
                # Interaction: 
                # APs containing both i and j.
                # If an AP contains i and j, flipping both changes mono status differently than sum of individual flips.
                # Let's just recompute cost? No, too slow.
                # Let's derive interaction.
                # Old cost contribution of AP: 1 if mono, 0 else.
                # New cost contribution: 1 if mono after flip, 0 else.
                # Sum over all APs.
                # For APs not containing i or j: no change.
                # For APs containing i but not j: change is delta_i_part.
                # For APs containing j but not i: change is delta_j_part.
                # For APs containing both i and j: change is (new_mono - old_mono).
                
                # We can compute delta_i and delta_j individually using the logic above, 
                # but we need to handle the interaction for APs containing both.
                
                # Actually, simpler: 
                # delta_total = (cost_after - cost_before)
                # We can compute cost_before for relevant APs and cost_after.
                # Relevant APs are those containing i or j.
                
                # Let's identify APs containing i or j.
                # This is hard to do fast without precomputed AP lists per node.
                # We have neighbor_flat which gives OTHERS.
                # If i and j are in same AP, they will appear in each other's neighbor lists?
                # Yes. If (i, j, u, v) is an AP, then j, u, v are in neighbors[i].
                # And i, u, v are in neighbors[j].
                
                # Let's just use the single flip deltas and add interaction.
                # delta_i = sum over APs containing i of (new_mono - old_mono)
                # delta_j = sum over APs containing j of (new_mono - old_mono)
                # Interaction term for APs containing both i and j:
                # In delta_i, we assumed j didn't change.
                # In delta_j, we assumed i didn't change.
                # We need to correct for APs where BOTH changed.
                
                # Let's find common neighbors.
                # nbrs_i = neighbors[i]
                # nbrs_j = neighbors[j]
                # An AP contains both i and j if there exists u,v such that (u,v) is in nbrs_i and (u,v) is in nbrs_j?
                # No, nbrs_i contains (j, u, v) if j is in the AP.
                # So if (j, u, v) is in nbrs_i, then AP is (i, j, u, v).
                # Check if (i, u, v) is in nbrs_j.
                
                # This is getting complicated. Let's just do single flips for now to ensure correctness and speed.
                # If we have time, we can add pair flips.
                # For now, fall back to single flip if pair logic is too slow/complex.
                pass
            
            i = random.randint(0, N - 1)
            deltas = calc_deltas(cols)
            delta = deltas[i]
            
            if delta < 0:
                cols[i] = 1 - cols[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_cols = cols.copy()
            elif delta == 0:
                cols[i] = 1 - cols[i]
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                
                if random.random() < prob:
                    cols[i] = 1 - cols[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cols = cols.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_cols, best_cost

    global_best_cols = None
    global_best_cost = float('inf')
    
    num_restarts = 10
    time_per_restart = 110.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            initial = [0] * N
        elif restart == 1:
            initial = [1] * N
        elif restart == 2:
            initial = [i % 2 for i in range(N)]
        elif restart == 3:
            initial = [(i // 8) % 2 for i in range(N)]
        elif restart == 4:
            initial = [random.randint(0, 1) for _ in range(N)]
        else:
            if global_best_cols is not None:
                initial = global_best_cols[:]
                for _ in range(N // 4):
                    idx = random.randint(0, N - 1)
                    initial[idx] = 1 - initial[idx]
            else:
                initial = [random.randint(0, 1) for _ in range(N)]
        
        T_start = 1.0 + 0.5 * (restart % 3)
        cooling = 0.9999 - 0.00001 * (restart % 5)
        min_T = 0.005
        
        sa_cols, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.8
        )
        
        polished_cols, polished_cost = greedy_polish(sa_cols, sa_cost)
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_cols = polished_cols[:]
    
    if global_best_cols is not None:
        final_cols, final_cost = greedy_polish(global_best_cols, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_cols = final_cols
    
    return global_best_cols
