"""Direct heuristic for TSP next-node selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill


def _fast_cost_estimate(dm, cands, rem_set, dest, c):
    """Fast O(n) cost estimate for the tour through all candidates.
    
    For each candidate c_j, compute:
      cost_j = d(c, c_j) + d(c_j, n_j) + sum_{i in rem_j} d(x_i, m_i) + d(last, dest)
    where n_j is the nearest neighbor of c_j in rem_j, and m_i is the nearest
    neighbor of x_i in the rest of rem_j (excluding its predecessor).
    """
    n = len(cands)
    d_c = dm[c, cands]
    
    # d_jk[i, j] = d(c_i, c_j) for i != j
    d_jk = np.where(np.eye(n, dtype=bool), np.inf, dm[cands, cands])
    
    # Nearest neighbor of each c_j in the full set
    nn_idx = np.argmin(d_jk, axis=1)
    nn_dist = d_jk[np.arange(n), nn_idx]
    
    # Second nearest neighbor
    sorted_idx = np.argsort(d_jk, axis=1)
    nn2_idx = sorted_idx[:, 1]
    nn2_dist = d_jk[np.arange(n), nn2_idx]
    
    costs = np.zeros(n)
    for j in range(n):
        # Cost from c to c_j
        cost = d_c[j]
        # Edge from c_j to its NN in rem_j
        nn = nn_idx[j]
        if nn == j:
            nn = nn2_idx[j]
        cost += nn_dist[j]
        
        # Sum of nearest-neighbor distances for all other nodes in rem_j
        nn_dists_all = d_jk[np.arange(n), nn_idx]
        nn2_dists_all = d_jk[np.arange(n), nn2_idx]
        
        # For each i, the effective NN distance in rem_j:
        # if nn_i == j, use nn2_i; else use nn_i
        eff_nn = np.where(nn_idx == j, nn2_dists_all, nn_dists_all)
        
        # Sum over all i != j
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        cost += np.sum(eff_nn[mask])
        
        # Final edge from last node to dest
        # Last node is nn (the one c_j went to)
        cost += dm[cands[nn], dest]
        
        costs[j] = cost
    
    return costs


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Uses an O(n) cost estimate that balances:
    - The cost of the edge to the candidate
    - The estimated remaining tour length through the other nodes
    - The cost of closing the tour back to destination
    
    For small n, uses a more exact 2-opt refined estimate.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Exact for 2 nodes
    if n_unvisited == 2:
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # Exact for 3 nodes
    if n_unvisited == 3:
        best_cost = np.inf
        best_node = unvisited_nodes[0]
        import itertools
        for first in range(3):
            rest = [i for i in range(3) if i != first]
            for second in rest:
                third = rest[0] if rest[1] == second else rest[1]
                c = (distance_matrix[current_node, unvisited_nodes[first]] +
                     distance_matrix[unvisited_nodes[first], unvisited_nodes[second]] +
                     distance_matrix[unvisited_nodes[second], unvisited_nodes[third]] +
                     distance_matrix[unvisited_nodes[third], destination_node])
                if c < best_cost:
                    best_cost = c
                    best_node = unvisited_nodes[first]
        return int(best_node)
    
    # For n up to 12, use exact DP or permutation-based search
    if n_unvisited <= 10:
        # Use a simple beam search with width 2 for the first few nodes
        # Then exact for the rest
        dm = distance_matrix
        c = current_node
        dest = destination_node
        
        # Greedy with 2-opt for the full remaining set
        # Build a tour via nearest neighbor from c
        remaining = list(unvisited_nodes)
        tour = []
        cur = c
        while remaining:
            d_to = np.array([dm[cur, r] for r in remaining])
            idx = int(np.argmin(d_to))
            nxt = remaining.pop(idx)
            tour.append(nxt)
            cur = nxt
        
        # 2-opt the tour
        best_tour = list(tour)
        for _ in range(10):  # a few passes
            improved = False
            for i in range(len(best_tour) - 1):
                for j in range(i + 1, len(best_tour)):
                    # Try reversing segment i..j
                    new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                    old_cost = (dm[c, best_tour[0]] +
                                sum(dm[best_tour[k], best_tour[k+1]] for k in range(len(best_tour)-1)) +
                                dm[best_tour[-1], dest])
                    new_cost = (dm[c, new_tour[0]] +
                                sum(dm[new_tour[k], new_tour[k+1]] for k in range(len(new_tour)-1)) +
                                dm[new_tour[-1], dest])
                    if new_cost < old_cost - 1e-9:
                        best_tour = new_tour
                        improved = True
            if not improved:
                break
        
        # Try each candidate as the first node with 2-opt on the rest
        best_first = best_tour[0]
        best_total = (dm[c, best_first] +
                      sum(dm[best_tour[k], best_tour[k+1]] for k in range(len(best_tour)-1)) +
                      dm[best_tour[-1], dest])
        
        for first_idx in range(n_unvisited):
            if first_idx == 0:
                continue
            first_node = unvisited_nodes[first_idx]
            rest = [unvisited_nodes[i] for i in range(n_unvisited) if i != first_idx]
            
            # NN tour from first_node through rest
            sub_tour = []
            cur2 = first_node
            rem2 = list(rest)
            while rem2:
                d_to = np.array([dm[cur2, r] for r in rem2])
                idx2 = int(np.argmin(d_to))
                nxt2 = rem2.pop(idx2)
                sub_tour.append(nxt2)
                cur2 = nxt2
            
            # 2-opt the sub_tour
            for _ in range(5):
                improved = False
                for i in range(len(sub_tour) - 1):
                    for j in range(i + 1, len(sub_tour)):
                        new_sub = sub_tour[:i] + sub_tour[i:j+1][::-1] + sub_tour[j+1:]
                        old_c2 = (dm[first_node, sub_tour[0]] +
                                  sum(dm[sub_tour[k], sub_tour[k+1]] for k in range(len(sub_tour)-1)) +
                                  dm[sub_tour[-1], dest])
                        new_c2 = (dm[first_node, new_sub[0]] +
                                  sum(dm[new_sub[k], new_sub[k+1]] for k in range(len(new_sub)-1)) +
                                  dm[new_sub[-1], dest])
                        if new_c2 < old_c2 - 1e-9:
                            sub_tour = new_sub
                            improved = True
                if not improved:
                    break
            
            total = (dm[c, first_node] +
                     sum(dm[sub_tour[k], sub_tour[k+1]] for k in range(len(sub_tour)-1)) +
                     dm[sub_tour[-1], dest])
            
            if total < best_total:
                best_total = total
                best_first = first_node
        
        return int(best_first)
    
    # For larger n, use the O(n) fast cost estimate
    dm = distance_matrix
    c = current_node
    dest = destination_node
    cands = unvisited_nodes
    
    # Compute the fast cost estimate
    costs = _fast_cost_estimate(dm, cands, set(range(n_unvisited)), dest, c)
    
    # For n up to 20, try 2-opt refinements on the top candidates
    if n_unvisited <= 20 and time.time() - _START_TIME < _DEADLINE - 30:
        top_k = min(3, n_unvisited)
        top_indices = np.argsort(costs)[:top_k]
        
        best_cost = np.inf
        best_node = cands[top_indices[0]]
        
        for ti in top_indices:
            first_node = cands[ti]
            rest = [cands[i] for i in range(n_unvisited) if i != ti]
            
            # NN tour from first_node
            sub_tour = []
            cur2 = first_node
            rem2 = list(rest)
            while rem2:
                d_to = dm[cur2, rem2]
                idx2 = int(np.argmin(d_to))
                nxt2 = rem2[idx2]
                sub_tour.append(nxt2)
                rem2 = rem2[:idx2] + rem2[idx2+1:]
                cur2 = nxt2
            
            # 2-opt with local delta
            for _ in range(5):
                improved = False
                for i in range(len(sub_tour) - 1):
                    for j in range(i + 2, len(sub_tour)):
                        # Reverse segment i..j
                        # Old edges: (sub_tour[i-1], sub_tour[i]) and (sub_tour[j], sub_tour[j+1])
                        # New edges: (sub_tour[i-1], sub_tour[j]) and (sub_tour[i], sub_tour[j+1])
                        # Handle boundaries
                        prev_node = first_node if i == 0 else sub_tour[i-1]
                        next_node = dest if j == len(sub_tour)-1 else sub_tour[j+1]
                        
                        old_cost = dm[prev_node, sub_tour[i]] + dm[sub_tour[j], next_node]
                        new_cost = dm[prev_node, sub_tour[j]] + dm[sub_tour[i], next_node]
                        
                        if new_cost < old_cost - 1e-9:
                            sub_tour = sub_tour[:i] + sub_tour[i:j+1][::-1] + sub_tour[j+1:]
                            improved = True
                if not improved:
                    break
            
            total = (dm[c, first_node] +
                     dm[first_node, sub_tour[0]] +
                     sum(dm[sub_tour[k], sub_tour[k+1]] for k in range(len(sub_tour)-1)) +
                     dm[sub_tour[-1], dest])
            
            if total < best_cost:
                best_cost = total
                best_node = first_node
        
        return int(best_node)
    
    # For n up to 30, use a simpler heuristic with destination awareness
    if n_unvisited <= 30:
        # Blend nearest-neighbor cost with destination proximity
        d_to_cands = dm[c, cands]
        d_to_dest = dm[cands, dest]
        
        # Normalize both to [0, 1]
        d_c_range = d_to_cands.max() - d_to_cands.min()
        d_dest_range = d_to_dest.max() - d_to_dest.min()
        
        if d_c_range > 0:
            norm_c = (d_to_cands - d_to_cands.min()) / d_c_range
        else:
            norm_c = np.zeros(n_unvisited)
        
        if d_dest_range > 0:
            norm_dest = (d_to_dest - d_to_dest.min()) / d_dest_range
        else:
            norm_dest = np.zeros(n_unvisited)
        
        # Weight: 70% nearest neighbor, 30% destination proximity
        blended = 0.7 * norm_c + 0.3 * norm_dest
        
        best_idx = np.argmin(blended)
        return int(cands[best_idx])
    
    # For very large n, just use the fast estimate
    best_idx = np.argmin(costs)
    return int(cands[best_idx])
# EVOLVE-BLOCK-END
