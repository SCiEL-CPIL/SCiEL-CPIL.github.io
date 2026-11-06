import time
import random
import math

DEADLINE = 145.0

def _compute_score_fast(adj_masks, degs, n=16):
    """Compute the maximum triangle degree sum using bitmask operations."""
    max_sum = 0
    for u in range(n):
        au = adj_masks[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj_masks[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    m ^= lsb
    return max_sum

def _state_to_edges(adj_masks, n=16):
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def _build_state_from_edges(edges, n=16):
    adj_masks = [0] * n
    degs = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        degs[u] += 1
        degs[v] += 1
    return adj_masks, degs

def _initial_construction():
    """Start from the balanced bipartite K8,8 plus one random edge."""
    n = 16
    adj_masks = [0] * n
    degs = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
    # Add one random edge within a part to create triangles
    if random.random() < 0.5:
        u, v = random.sample(range(8), 2)
    else:
        u, v = random.sample(range(8, 16), 2)
    adj_masks[u] |= (1 << v)
    adj_masks[v] |= (1 << u)
    degs[u] += 1
    degs[v] += 1
    return adj_masks, degs

def _random_init(n=16, m=65):
    """Random 65-edge graph."""
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = random.sample(all_pairs, m)
    return _build_state_from_edges(edges, n)

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)
    
    best_score = float('inf')
    best_adj_masks = None
    best_degs = None
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Vary initialization strategy
        if restart < 4:
            adj_masks, degs = _initial_construction()
        else:
            adj_masks, degs = _random_init(n, m_target)
        
        # Build edge list and set for fast lookups
        edge_list = []
        edge_set = set()
        for i in range(n):
            mask = adj_masks[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edge_list.append((i, j))
                    edge_set.add((i, j))
        
        current_score = _compute_score_fast(adj_masks, degs, n)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_masks = adj_masks[:]
            best_degs = degs[:]
        
        # Simulated annealing with fast score computation
        T = 8.0
        T_min = 0.01
        cooling = 0.99995
        
        while time.time() - start_time < DEADLINE:
            if not edge_list:
                break
            
            # Choose a random edge to remove
            rem_idx = random.randint(0, len(edge_list) - 1)
            rem_u, rem_v = edge_list[rem_idx]
            
            # Choose a random edge to add
            add_u, add_v = random.choice(all_pairs)
            if (add_u, add_v) in edge_set:
                continue
            
            # Apply swap
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edge_list.pop(rem_idx)
            edge_set.discard((rem_u, rem_v))
            
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edge_list.append((add_u, add_v))
            edge_set.add((add_u, add_v))
            
            new_score = _compute_score_fast(adj_masks, degs, n)
            
            delta = new_score - current_score
            
            if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj_masks = adj_masks[:]
                    best_degs = degs[:]
            else:
                # Revert
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                edge_list.pop()
                edge_set.discard((add_u, add_v))
                
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                edge_list.append((rem_u, rem_v))
                edge_set.add((rem_u, rem_v))
            
            T *= cooling
            if T < T_min:
                T = 5.0
        
        # Local polish: try all single-edge swaps to reduce score
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            current_score = _compute_score_fast(adj_masks, degs, n)
            for rem_u, rem_v in list(edge_list):
                for add_u, add_v in all_pairs:
                    if (add_u, add_v) in edge_set:
                        continue
                    # Apply swap
                    adj_masks[rem_u] &= ~(1 << rem_v)
                    adj_masks[rem_v] &= ~(1 << rem_u)
                    degs[rem_u] -= 1
                    degs[rem_v] -= 1
                    
                    adj_masks[add_u] |= (1 << add_v)
                    adj_masks[add_v] |= (1 << add_u)
                    degs[add_u] += 1
                    degs[add_v] += 1
                    
                    new_score = _compute_score_fast(adj_masks, degs, n)
                    
                    if new_score < current_score:
                        # Accept
                        edge_set.discard((rem_u, rem_v))
                        edge_set.add((add_u, add_v))
                        edge_list.remove((rem_u, rem_v))
                        edge_list.append((add_u, add_v))
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_adj_masks = adj_masks[:]
                            best_degs = degs[:]
                    else:
                        # Revert
                        adj_masks[add_u] &= ~(1 << add_v)
                        adj_masks[add_v] &= ~(1 << add_u)
                        degs[add_u] -= 1
                        degs[add_v] -= 1
                        
                        adj_masks[rem_u] |= (1 << rem_v)
                        adj_masks[rem_v] |= (1 << rem_u)
                        degs[rem_u] += 1
                        degs[rem_v] += 1
    
    if best_adj_masks is None:
        adj_masks, degs = _initial_construction()
        best_adj_masks = adj_masks[:]
    
    best_edges = _state_to_edges(best_adj_masks, n)
    return {'n': n, 'edges': best_edges}
