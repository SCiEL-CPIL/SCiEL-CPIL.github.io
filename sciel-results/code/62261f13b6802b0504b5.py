"""Direct heuristic for TSP next-node selection with convex hull peeling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill


def _hull_score(dm, cands, c, dest):
    """
    Score candidates for convex hull peeling.
    
    The score combines:
    1. Distance from current node (prefer closer)
    2. "Hull-ness": average distance to other unvisited nodes (prefer farther = more peripheral)
    3. Angle deviation from direction toward destination (prefer aligned)
    
    Lower score = better candidate.
    """
    n = len(cands)
    d_to_c = dm[c, cands]
    
    # Hull-ness: average distance to other unvisited nodes
    # Points on the convex hull tend to have larger average distance to others
    d_jk = np.where(np.eye(n, dtype=bool), 0, dm[cands, cands])
    avg_dist = d_jk.sum(axis=1) / max(n - 1, 1)
    
    # Normalize components to [0, 1]
    d_c_min, d_c_max = d_to_c.min(), d_to_c.max()
    d_c_range = d_c_max - d_c_min
    if d_c_range > 1e-12:
        norm_d_c = (d_to_c - d_c_min) / d_c_range
    else:
        norm_d_c = np.zeros(n)
    
    h_min, h_max = avg_dist.min(), avg_dist.max()
    h_range = h_max - h_min
    if h_range > 1e-12:
        norm_hull = (avg_dist - h_min) / h_range
    else:
        norm_hull = np.zeros(n)
    
    # We want to PREFER high hull-ness (peripheral points), so invert for scoring
    # Score = w1 * norm_d_c + w2 * (1 - norm_hull)
    # Lower is better: prefer close to current AND peripheral
    
    # Weight: 50% distance to current, 50% hull-ness (inverted)
    w_dist = 0.5
    w_hull = 0.5
    
    score = w_dist * norm_d_c + w_hull * (1.0 - norm_hull)
    
    return score


def _fast_cost_estimate(dm, cands, dest, c):
    """Fast O(n) cost estimate for the tour through all candidates."""
    n = len(cands)
    d_c = dm[c, cands]
    
    d_jk = np.where(np.eye(n, dtype=bool), np.inf, dm[cands, cands])
    
    nn_idx = np.argmin(d_jk, axis=1)
    nn_dist = d_jk[np.arange(n), nn_idx]
    
    sorted_idx = np.argsort(d_jk, axis=1)
    nn2_idx = sorted_idx[:, 1]
    nn2_dist = d_jk[np.arange(n), nn2_idx]
    
    costs = np.zeros(n)
    for j in range(n):
        cost = d_c[j]
        nn = nn_idx[j]
        if nn == j:
            nn = nn2_idx[j]
        cost += nn_dist[j]
        
        nn_dists_all = d_jk[np.arange(n), nn_idx]
        nn2_dists_all = d_jk[np.arange(n), nn2_idx]
        
        eff_nn = np.where(nn_idx == j, nn2_dists_all, nn_dists_all)
        
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        cost += np.sum(eff_nn[mask])
        
        cost += dm[cands[nn], dest]
        
        costs[j] = cost
    
    return costs


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction using convex hull peeling.
    
    The heuristic prefers nodes that are:
    1. Close to the current node
    2. Peripheral to the unvisited set (high average distance to others)
    
    This encourages tracing the convex hull before filling interior nodes.
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
    
    dm = distance_matrix
    c = current_node
    dest = destination_node
    cands = unvisited_nodes
    
    # For n up to 10, use exhaustive search with 2-opt
    if n_unvisited <= 10:
        best_first = None
        best_total = np.inf
        
        for first_idx in range(n_unvisited):
            first_node = cands[first_idx]
            rest = [cands[i] for i in range(n_unvisited) if i != first_idx]
            
            # NN tour from first_node through rest
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
            
            # 2-opt the sub_tour
            for _ in range(5):
                improved = False
                for i in range(len(sub_tour) - 1):
                    for j in range(i + 2, len(sub_tour)):
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
            
            if total < best_total:
                best_total = total
                best_first = first_node
        
        return int(best_first)
    
    # For larger n, use convex hull peeling heuristic
    # Combine hull-ness score with fast cost estimate
    
    # Compute hull score
    hull_scores = _hull_score(dm, cands, c, dest)
    
    # Compute fast cost estimate
    cost_scores = _fast_cost_estimate(dm, cands, dest, c)
    
    # Normalize both to [0, 1]
    h_min, h_max = hull_scores.min(), hull_scores.max()
    h_range = h_max - h_min
    if h_range > 1e-12:
        norm_hull = (hull_scores - h_min) / h_range
    else:
        norm_hull = np.zeros(n_unvisited)
    
    c_min, c_max = cost_scores.min(), cost_scores.max()
    c_range = c_max - c_min
    if c_range > 1e-12:
        norm_cost = (cost_scores - c_min) / c_range
    else:
        norm_cost = np.zeros(n_unvisited)
    
    # Blend: 60% hull-ness, 40% cost estimate
    # Both are "lower is better" after normalization
    blended = 0.6 * norm_hull + 0.4 * norm_cost
    
    best_idx = np.argmin(blended)
    return int(cands[best_idx])
# EVOLVE-BLOCK-END
