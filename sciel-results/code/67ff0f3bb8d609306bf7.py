"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Deadline guard: fall back to nearest-neighbor if time is running out
    if time.time() > _DEADLINE:
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    dm = distance_matrix
    
    # Use exact DP when n is small enough (say n <= 15)
    if n <= 15:
        # Held-Karp DP to find the optimal first node
        full_mask = (1 << n) - 1
        INF = np.inf
        
        # dp[mask][last] = min cost to have visited set 'mask' of unvisited nodes,
        # currently at unv[last], having started at current_node.
        dp = np.full((1 << n, n), INF, dtype=np.float64)
        
        # Initialize
        for i in range(n):
            dp[1 << i, i] = dm[current_node, unv[i]]
        
        # Transition
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                cur_cost = dp[mask, last]
                if cur_cost >= INF:
                    continue
                remaining = full_mask ^ mask
                if remaining == 0:
                    continue
                rem = remaining
                while rem:
                    bit = rem & (-rem)
                    next_i = bit.bit_length() - 1
                    new_mask = mask | bit
                    new_cost = cur_cost + dm[unv[last], unv[next_i]]
                    if new_cost < dp[new_mask, next_i]:
                        dp[new_mask, next_i] = new_cost
                    rem ^= bit
        
        # Find the best first node
        best_total = INF
        best_first = int(unv[0])
        
        for f in range(n):
            d_first = dm[current_node, unv[f]]
            
            if n == 1:
                total = d_first + dm[unv[f], destination_node]
            else:
                mask_f = full_mask ^ (1 << f)
                
                # Compute min cost from unv[f] to visit all nodes in mask_f and return to destination
                dp_f = np.full((1 << n, n), INF, dtype=np.float64)
                for i in range(n):
                    if i == f:
                        continue
                    if mask_f & (1 << i):
                        dp_f[1 << i, i] = dm[unv[f], unv[i]]
                
                for mask in range(1 << n):
                    if mask & ~(mask_f):
                        continue
                    for last in range(n):
                        if last == f:
                            continue
                        if not (mask & (1 << last)):
                            continue
                        cur_cost = dp_f[mask, last]
                        if cur_cost >= INF:
                            continue
                        remaining = mask_f ^ mask
                        if remaining == 0:
                            continue
                        rem = remaining
                        while rem:
                            bit = rem & (-rem)
                            next_i = bit.bit_length() - 1
                            new_mask = mask | bit
                            new_cost = cur_cost + dm[unv[last], unv[next_i]]
                            if new_cost < dp_f[new_mask, next_i]:
                                dp_f[new_mask, next_i] = new_cost
                            rem ^= bit
                
                best_from_f = INF
                for last in range(n):
                    if last == f:
                        continue
                    if mask_f & (1 << last):
                        cost = dp_f[mask_f, last] + dm[unv[last], destination_node]
                        if cost < best_from_f:
                            best_from_f = cost
                
                total = d_first + best_from_f
            
            if total < best_total:
                best_total = total
                best_first = int(unv[f])
        
        return best_first
    
    # For n > 15, use the improved heuristic
    best_score = np.inf
    best_node = int(unv[0])
    
    # Precompute distances from current to all unvisited
    dists_from_cur = dm[current_node, unv]
    
    for i in range(n):
        c = unv[i]
        d1 = dists_from_cur[i]
        
        # Pruning
        if d1 >= best_score:
            continue
        
        if n > 2:
            # Find the best next node z from c, considering the cost to go from z to destination
            # Score for c is d1 + min_z (d(c, z) + 0.5 * d(z, dest))
            remaining = np.delete(unv, i)
            dists_c = dm[c, remaining]
            dists_dest = dm[destination_node, remaining]
            
            # Combined cost for the next step
            next_costs = dists_c + 0.5 * dists_dest
            idx_z = np.argmin(next_costs)
            d2 = dists_c[idx_z]
            d3 = dists_dest[idx_z]
            
            score = d1 + d2 + 0.5 * d3
            
            if score < best_score:
                best_score = score
                best_node = int(c)
        else:
            # n == 2
            other = int(unv[1 - i])
            d2 = dm[c, other]
            d3 = dm[other, destination_node]
            score = d1 + d2 + 0.5 * d3
            if score < best_score:
                best_score = score
                best_node = int(c)
                
    return best_node
# EVOLVE-BLOCK-END
