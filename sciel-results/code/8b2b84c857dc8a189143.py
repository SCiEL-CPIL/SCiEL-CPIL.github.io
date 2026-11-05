import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 16
    target_edges = 65
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Initial state
    init_edges = set()
    for u in range(8):
        for v in range(8, 16):
            init_edges.add((u, v))
    init_edges.add((0, 1))
    
    list_edges = list(init_edges)
    is_edge = [False] * (n * n) # or dict, but n*n is small
    for u, v in list_edges:
        is_edge[u * n + v] = True
        is_edge[v * n + u] = True
        
    list_non_edges = [e for e in all_edges if not is_edge[e[0] * n + e[1]]]
    
    def get_score(list_e):
        adj_mask = [0] * n
        deg = [0] * n
        for u, v in list_e:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
        max_sum = 0
        full_mask = (1 << n) - 1
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    # w must be > v
                    # mask for w > v
                    w_mask = common & (full_mask ^ ((1 << (v + 1)) - 1))
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        w_mask &= w_mask - 1
        return max_sum

    current_score = get_score(list_edges)
    best_edges = list(list_edges)
    best_score = current_score
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    steps_since_improvement = 0
    max_steps_no_improve = 10000
    
    while time.time() < deadline:
        if not list_edges or not list_non_edges:
            break
            
        idx_out = random.randint(0, len(list_edges) - 1)
        idx_in = random.randint(0, len(list_non_edges) - 1)
        
        e_out = list_edges[idx_out]
        e_in = list_non_edges[idx_in]
        
        # Swap
        list_edges[idx_out] = list_edges[-1]
        list_edges.pop()
        list_non_edges.append(e_out)
        
        list_non_edges[idx_in] = list_non_edges[-1]
        list_non_edges.pop()
        list_edges.append(e_in)
        
        # Update is_edge (optional if we only use lists for score, but needed for non_edges rebuild on restart)
        # Actually, we can just rebuild is_edge on restart.
        # For now, we don't use is_edge in the loop, only for init and restart.
        
        new_score = get_score(list_edges)
        
        if new_score < current_score:
            current_score = new_score
            steps_since_improvement = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = list(list_edges)
        else:
            steps_since_improvement += 1
            if new_score > current_score:
                delta = new_score - current_score
                if random.random() < pow(2.718, -delta / T):
                    current_score = new_score
            else:
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        if steps_since_improvement > max_steps_no_improve:
            # Restart
            list_edges = list(best_edges)
            # Rebuild is_edge
            for i in range(n * n):
                is_edge[i] = False
            for u, v in list_edges:
                is_edge[u * n + v] = True
                is_edge[v * n + u] = True
            list_non_edges = [e for e in all_edges if not is_edge[e[0] * n + e[1]]]
            current_score = best_score
            T = 1.0
            steps_since_improvement = 0
            max_steps_no_improve = int(max_steps_no_improve * 0.8)
            if max_steps_no_improve < 2000:
                max_steps_no_improve = 2000

    return {'n': n, 'edges': [[u, v] for u, v in best_edges]}
