import time
import random

DEADLINE = 115

def _compute_score(n, adj_masks, degs):
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
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_adj_masks = None
    
    while time.time() - start_time < DEADLINE:
        # Initialize
        if random.random() < 0.5:
            adj_masks = [0] * n
            degs = [0] * n
            for u in range(8):
                for v in range(8, 16):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    degs[u] += 1
                    degs[v] += 1
            if random.random() < 0.5:
                u, v = random.sample(range(8), 2)
            else:
                u, v = random.sample(range(8, 16), 2)
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
        else:
            shuffled_pairs = all_pairs[:]
            random.shuffle(shuffled_pairs)
            edges = shuffled_pairs[:65]
            adj_masks = [0] * n
            degs = [0] * n
            for u, v in edges:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        
        current_edges_list = []
        current_edges_set = set()
        for i in range(n):
            mask = adj_masks[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    current_edges_list.append((i, j))
                    current_edges_set.add((i, j))
                    
        current_score = _compute_score(n, adj_masks, degs)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_masks = adj_masks[:]
            
        T = 10.0
        T_min = 0.01
        cooling = 0.99999
        
        while time.time() - start_time < DEADLINE:
            if not current_edges_list:
                break
                
            rem_u, rem_v = random.choice(current_edges_list)
            add_u, add_v = random.choice(all_pairs)
            
            if (add_u, add_v) in current_edges_set:
                continue
                
            # Apply swap
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            current_edges_list.remove((rem_u, rem_v))
            current_edges_set.remove((rem_u, rem_v))
            
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            current_edges_list.append((add_u, add_v))
            current_edges_set.add((add_u, add_v))
            
            new_score = _compute_score(n, adj_masks, degs)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj_masks = adj_masks[:]
            else:
                # Revert
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                current_edges_list.remove((add_u, add_v))
                current_edges_set.remove((add_u, add_v))
                
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                current_edges_list.append((rem_u, rem_v))
                current_edges_set.add((rem_u, rem_v))
                
            T *= cooling
            if T < T_min:
                T = 5.0

    best_edges = _state_to_edges(best_adj_masks)
    return {'n': n, 'edges': best_edges}
