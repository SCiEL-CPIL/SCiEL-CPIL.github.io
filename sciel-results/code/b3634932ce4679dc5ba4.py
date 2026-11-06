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
    
    # When few nodes remain, use a small exact DP (Held-Karp style) over the suffix
    # to pick the node that leads to the optimal completion of the remaining tour.
    # We consider the last k nodes: pick the first node in the optimal path from
    # current_node through all unvisited nodes back to destination_node.
    
    # Use exact DP when n is small enough (say n <= 15)
    if n <= 15:
        # Build index mapping: unv[0..n-1] -> local indices 0..n-1
        # DP state: dp[mask][last] = min cost to visit exactly the set of nodes in mask
        # (among unvisited), ending at unv[last], starting from current_node.
        # We want the min over all masks == all-nodes, last of dp[full_mask][last] + dm[unv[last], destination_node]
        # But we only need to decide which node to visit FIRST.
        # So we compute for each candidate first node c: min cost of visiting all remaining
        # after c, then returning to destination.
        
        # Actually, let's just compute the full optimal tour from current through all unvisited to destination,
        # and return the first node in that optimal path.
        
        full_mask = (1 << n) - 1
        INF = np.inf
        
        # dp[mask][last] = min cost to have visited set 'mask' of unvisited nodes,
        # currently at unv[last], having started at current_node.
        # Initialize: dp[1<<i][i] = dm[current_node, unv[i]]
        dp = np.full((1 << n, n), INF, dtype=np.float64)
        for i in range(n):
            dp[1 << i, i] = dm[current_node, unv[i]]
        
        # Transition: for each mask, for each last in mask, for each next not in mask:
        # dp[mask | (1<<next)][next] = min(dp[mask][last] + dm[unv[last], unv[next]])
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                cur_cost = dp[mask, last]
                if cur_cost >= INF:
                    continue
                # Try extending to next nodes
                remaining = full_mask ^ mask
                if remaining == 0:
                    continue
                # Iterate over set bits in remaining
                rem = remaining
                while rem:
                    bit = rem & (-rem)
                    next_i = bit.bit_length() - 1
                    new_mask = mask | bit
                    new_cost = cur_cost + dm[unv[last], unv[next_i]]
                    if new_cost < dp[new_mask, next_i]:
                        dp[new_mask, next_i] = new_cost
                    rem ^= bit
        
        # Find the best completion: min over last of dp[full_mask][last] + dm[unv[last], destination_node]
        best_total = INF
        best_first = int(unv[0])
        
        # To find the first node in the optimal path, we need to trace back.
        # Alternatively, for each candidate first node f, compute the min cost of
        # visiting all other nodes after f and returning to destination.
        # Cost if first = f: dm[current, unv[f]] + min_cost_to_visit_all_except_f_then_return
        # where min_cost_to_visit_all_except_f_then_return = min over paths from unv[f] through all other unvisited back to destination.
        
        # Let's compute for each f the optimal cost assuming f is visited first.
        # After visiting f, we need to visit all nodes in (full_mask \ {f}) and return to destination.
        # This is: min over last in (full_mask \ {f}) of [min cost from unv[f] to visit all in (full_mask\{f}) ending at unv[last]] + dm[unv[last], destination]
        
        # We can reuse the DP. Let mask_f = full_mask ^ (1 << f) (all nodes except f).
        # We need: min over last in mask_f of dp_from_f[mask_f][last] + dm[unv[last], dest]
        # where dp_from_f is similar to dp but starting from unv[f] instead of current_node.
        
        # To avoid recomputing, let's just compute the optimal first node by checking each f.
        # For n <= 15, this is feasible.
        
        for f in range(n):
            # Cost to go from current to unv[f]
            d_first = dm[current_node, unv[f]]
            
            if n == 1:
                total = d_first + dm[unv[f], destination_node]
            else:
                # After visiting f, need to visit all nodes in mask_f = full_mask ^ (1<<f)
                mask_f = full_mask ^ (1 << f)
                # Compute min cost from unv[f] to visit all nodes in mask_f and return to destination
                # Use a smaller DP or reuse structure
                # For simplicity, compute min over last in mask_f of:
                #   (min path cost from unv[f] to visit all nodes in mask_f ending at unv[last]) + dm[unv[last], dest]
                
                # We can compute this with another DP, but that's O(n * 2^n * n) per f, which is O(n^2 * 2^n) total.
                # For n=15, that's 15^2 * 32768 ≈ 7.4M operations, which is fine.
                
                # Actually, let's just use the main dp array differently.
                # The main dp starts from current_node. We need a dp that starts from unv[f].
                # Let's compute it on the fly for each f.
                
                dp_f = np.full((1 << n, n), INF, dtype=np.float64)
                # Initialize: from unv[f], go to each node i in mask_f
                for i in range(n):
                    if i == f:
                        continue
                    if mask_f & (1 << i):
                        dp_f[1 << i, i] = dm[unv[f], unv[i]]
                
                for mask in range(1 << n):
                    # Only consider masks that are subsets of mask_f
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
                
                # Find best completion from f
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
    
    # For n > 15, use the heuristic approach
    best_score = np.inf
    best_node = int(unv[0])
    
    # Precompute distances from current to all unvisited
    dists_from_cur = dm[current_node, unv]
    
    for i in range(n):
        c = unv[i]
        d1 = dists_from_cur[i]
        
        # Pruning: if immediate cost is already worse than best total, skip
        if d1 >= best_score:
            continue
        
        if n > 2:
            # Find next best node x from c among remaining
            remaining = np.delete(unv, i)
            dists_c = dm[c, remaining]
            idx_x = np.argmin(dists_c)
            x = remaining[idx_x]
            d2 = dists_c[idx_x]
            
            partial_score = d1 + 0.5 * d2
            if partial_score >= best_score:
                continue
            
            score = partial_score
            
            if n > 3:
                # Find next best node y from x among remaining (excluding c and x)
                remaining2 = np.delete(remaining, idx_x)
                dists_x = dm[x, remaining2]
                idx_y = np.argmin(dists_x)
                d3 = dists_x[idx_y]
                score += 0.25 * d3
            
            if score < best_score:
                best_score = score
                best_node = int(c)
        else:
            # n == 2
            other = int(unv[1 - i])
            d2 = dm[c, other]
            score = d1 + 0.5 * d2
            if score < best_score:
                best_score = score
                best_node = int(c)
                
    return best_node
# EVOLVE-BLOCK-END
