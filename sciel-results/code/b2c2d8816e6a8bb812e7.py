import time
import random

def count_edges_in_subgraph(adj, mask, n):
    count = 0
    m = mask
    while m:
        lsw = m & -m
        idx = lsw.bit_length() - 1
        m ^= lsw
        count += (adj[idx] & m).bit_count()
    return count

def count_k4(adj, n):
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                total += count_edges_in_subgraph(adj, common, n)
    return total // 6

def has_ind6(adj, n):
    all_bits = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = all_bits ^ (1 << i) ^ adj[i]
        
    # Backtracking for K6 in comp_adj
    def backtrack(candidates, depth):
        if depth == 6:
            return True
        if candidates.bit_count() < depth:
            return False
            
        # Heuristic: pick vertex with highest degree in candidates
        # To keep it simple and fast, just pick lowest set bit
        # But for pruning, picking high degree is better.
        # Let's try picking lowest set bit first for speed of implementation.
        lsw = candidates & -candidates
        v = lsw.bit_length() - 1
        candidates_without_v = candidates ^ lsw
        
        # Branch 1: Include v
        if backtrack(candidates_without_v & comp_adj[v], depth + 1):
            return True
            
        # Branch 2: Exclude v
        if backtrack(candidates_without_v, depth):
            return True
            
        return False
        
    return backtrack(all_bits, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_graph = None
    best_n = 0
    
    # Fallback: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    paley_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_graph = {'n': 17, 'edges': paley_edges}
    best_n = 17
    
    for target_n in range(35, 17, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few restarts
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            adj = [0] * target_n
            # Initialize with p=0.5
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            current_k4 = count_k4(adj, target_n)
            
            for step in range(2000):
                if time.time() > deadline - 10:
                    break
                    
                # If K4 is 0, check Ind6
                if current_k4 == 0:
                    if not has_ind6(adj, target_n):
                        edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        best_graph = {'n': target_n, 'edges': edges}
                        best_n = target_n
                        found = True
                        break
                    # If Ind6 exists, we need to remove it.
                    # How? Find an Ind6 and flip one of its non-edges to edge?
                    # Or just continue hill climbing on a combined score?
                    # Let's use score = current_k4 + (10 if has_ind6 else 0)
                    score = 10
                else:
                    score = current_k4
                    
                # Pick random edge
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v: continue
                
                # Flip
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                new_k4 = count_k4(adj, target_n)
                
                if new_k4 == 0:
                    new_ind6 = has_ind6(adj, target_n)
                    new_score = 10 if new_ind6 else 0
                else:
                    new_score = new_k4
                    
                if new_score < score:
                    current_k4 = new_k4
                    score = new_score
                else:
                    # Revert
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            if found:
                break
                
    return best_graph
