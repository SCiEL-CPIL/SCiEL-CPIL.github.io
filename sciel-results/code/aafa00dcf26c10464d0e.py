import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_n = q
    best_adj = adj[:]
    
    # Try to expand to 42, 41, ...
    # We try 42 first. If it fails, we try 41, etc.
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() > deadline:
            break
            
        success = False
        # Try multiple random restarts for the expansion
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Reset random seed for reproducibility of attempts?
            # Or just let it run.
            random.seed(attempt)
            
            # Start with current best_adj
            curr_adj = best_adj[:]
            curr_n = best_n
            
            # Add vertices one by one
            failed = False
            for new_n in range(curr_n + 1, target_n + 1):
                if time.time() > deadline:
                    failed = True
                    break
                    
                new_vertex = new_n - 1
                curr_adj.append(0)
                
                # Random init for new vertex
                for i in range(new_vertex):
                    if random.random() < 0.5:
                        curr_adj[new_vertex] |= (1 << i)
                        curr_adj[i] |= (1 << new_vertex)
                        
                # Repair loop
                valid = False
                for step in range(5000):
                    if time.time() > deadline:
                        failed = True
                        break
                        
                    # Check K5 via new vertex
                    nbrs = curr_adj[new_vertex]
                    k4 = _find_k_clique_in_subgraph(curr_adj, nbrs, 4)
                    if k4 is not None:
                        v = k4[0]
                        curr_adj[new_vertex] &= ~(1 << v)
                        curr_adj[v] &= ~(1 << new_vertex)
                        continue
                        
                    # Check Indep5 via new vertex
                    mask_old = (1 << new_vertex) - 1
                    non_nbrs = mask_old & (~curr_adj[new_vertex])
                    
                    indep4 = _find_independent_set_in_subgraph(curr_adj, non_nbrs, 4)
                    if indep4 is not None:
                        v = indep4[0]
                        curr_adj[new_vertex] |= (1 << v)
                        curr_adj[v] |= (1 << new_vertex)
                        continue
                        
                    # No conflicts
                    valid = True
                    break
                    
                if failed:
                    break
                if not valid:
                    failed = True
                    break
                    
                curr_n = new_n
                
            if not failed and curr_n == target_n:
                best_n = target_n
                best_adj = curr_adj
                success = True
                break
                
        if success:
            if target_n == 42:
                break

    # Convert best_adj to edges
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append((i, j))
                
    return best_n, edges

def _find_k_clique_in_subgraph(adj, mask, k):
    if mask.bit_count() < k:
        return None
        
    def backtrack(candidates, current):
        if len(current) == k:
            return current[:]
        if not candidates:
            return None
            
        c = candidates
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            c &= (c - 1)
            
            new_cands = candidates & adj[v]
            if new_cands.bit_count() < (k - len(current) - 1):
                continue
                
            res = backtrack(new_cands, current + [v])
            if res is not None:
                return res
        return None
        
    return backtrack(mask, [])

def _find_independent_set_in_subgraph(adj, mask, k):
    if mask.bit_count() < k:
        return None
        
    def backtrack(candidates, current):
        if len(current) == k:
            return current[:]
        if not candidates:
            return None
            
        c = candidates
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            c &= (c - 1)
            
            new_cands = candidates & (~adj[v])
            if new_cands.bit_count() < (k - len(current) - 1):
                continue
                
            res = backtrack(new_cands, current + [v])
            if res is not None:
                return res
        return None
        
    return backtrack(mask, [])
