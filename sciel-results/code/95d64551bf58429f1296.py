import time
import random
import math

DEADLINE = 115  # Safety margin for 120s limit

def _compute_score_and_hot_edges(n, adj_masks, degs):
    """
    Compute the largest degree sum over triangles and return the set of edges
    that participate in triangles achieving this max sum.
    """
    max_sum = 0
    hot_edges = set()
    # Iterate over all potential edges (u, v)
    for u in range(n):
        adj_u = adj_masks[u]
        for v in range(u + 1, n):
            if (adj_u >> v) & 1:
                common = adj_u & adj_masks[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                        hot_edges = {(u, v), (u, w), (v, w)}
                    elif s == max_sum:
                        hot_edges.add((u, v))
                        hot_edges.add((u, w))
                        hot_edges.add((v, w))
                    m ^= lsb
    return max_sum, hot_edges

def _compute_score_only(n, adj_masks, degs):
    """
    Fast computation of just the max triangle degree sum (no hot edges).
    """
    max_sum = 0
    for u in range(n):
        adj_u = adj_masks[u]
        for v in range(u + 1, n):
            if (adj_u >> v) & 1:
                common = adj_u & adj_masks[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    m ^= lsb
    return max_sum

def _initial_construction():
    """Base construction: K_{8,8} plus one internal edge (0,1)."""
    n = 16
    adj_masks = [0] * n
    degs = [0] * n
    edges = []
    
    # Add K_{8,8} edges
    for u in range(8):
        for v in range(8, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
            edges.append((u, v))
            
    # Add one internal edge to create triangles
    u, v = 0, 1
    adj_masks[u] |= (1 << v)
    adj_masks[v] |= (1 << u)
    degs[u] += 1
    degs[v] += 1
    edges.append((u, v))
    
    return adj_masks, degs, edges

def _state_to_edges(adj_masks):
    n = 16
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    # Precompute all pairs for random sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_adj_masks = None
    
    # Simulated annealing parameters
    initial_temp = 5.0
    cooling_rate = 0.9995
    min_temp = 0.1
    
    # Run multiple restarts with simulated annealing
    for restart_idx in range(500):
        if time.time() - start_time > DEADLINE:
            break
            
        # Initialize state
        adj_masks, degs, edges_list = _initial_construction()
        
        # Perturb initial state for restarts > 0
        if restart_idx > 0:
            # Perform a few random swaps to diversify
            for _ in range(5):
                if len(edges_list) < 1: continue
                rem_idx = random.randint(0, len(edges_list) - 1)
                rem_u, rem_v = edges_list[rem_idx]
                
                found = False
                for _ in range(30):
                    add_u, add_v = random.choice(all_pairs)
                    if not ((adj_masks[add_u] >> add_v) & 1):
                        found = True
                        break
                if not found: continue
                
                # Remove
                adj_masks[rem_u] &= ~(1 << rem_v)
                adj_masks[rem_v] &= ~(1 << rem_u)
                degs[rem_u] -= 1
                degs[rem_v] -= 1
                edges_list.pop(rem_idx)
                
                # Add
                adj_masks[add_u] |= (1 << add_v)
                adj_masks[add_v] |= (1 << add_u)
                degs[add_u] += 1
                degs[add_v] += 1
                edges_list.append((add_u, add_v))
        
        current_score, hot_edges = _compute_score_and_hot_edges(n, adj_masks, degs)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_masks = adj_masks[:]
            
        # Simulated annealing loop
        temp = initial_temp
        max_steps = 500
        
        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
            
            if temp < min_temp:
                break
            
            # Select edge to remove
            # Bias: Pick an edge that is part of a high-scoring triangle
            if hot_edges:
                rem_u, rem_v = random.choice(list(hot_edges))
                rem_idx = edges_list.index((rem_u, rem_v)) if (rem_u, rem_v) in edges_list else edges_list.index((rem_v, rem_u))
            else:
                rem_idx = random.randint(0, len(edges_list) - 1)
                rem_u, rem_v = edges_list[rem_idx]
            
            # Select edge to add
            # Pick a random non-edge
            add_u, add_v = None, None
            for _ in range(15):
                cand_u, cand_v = random.choice(all_pairs)
                if not ((adj_masks[cand_u] >> cand_v) & 1):
                    add_u, add_v = cand_u, cand_v
                    break
            
            if add_u is None:
                continue

            # Perform Swap
            # 1. Remove (rem_u, rem_v)
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edges_list.pop(rem_idx)
            
            # 2. Add (add_u, add_v)
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edges_list.append((add_u, add_v))
            
            # Evaluate
            new_score, new_hot_edges = _compute_score_and_hot_edges(n, adj_masks, degs)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/temp) if worse
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_score = new_score
                hot_edges = new_hot_edges
                if current_score < best_score:
                    best_score = current_score
                    best_adj_masks = adj_masks[:]
            else:
                # Revert
                # Remove (add_u, add_v)
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                edges_list.pop()
                
                # Add (rem_u, rem_v)
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                edges_list.append((rem_u, rem_v))
            
            # Cool down
            temp *= cooling_rate
            
            # Early termination if we've found a very good solution
            if current_score <= 23:
                break
                    
    # Return the best result found
    if best_adj_masks is None:
        # Fallback: return the base construction
        adj_masks, degs, edges_list = _initial_construction()
        best_adj_masks = adj_masks
        
    best_edges = _state_to_edges(best_adj_masks)
    return {'n': n, 'edges': best_edges}
