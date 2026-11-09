import time
import sys

def construct():
    # Seed data
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix/bitmask
    adj = [0] * n
    degrees = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
        
    # Precompute clique and independent set properties for all subsets
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_indep[mask] = is_indep[rest] and ((adj[v] & rest) == 0)
        
    # Fallback greedy solution
    es = {tuple(e) for e in edges}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    best_parts = parts
    best_num_parts = len(parts)
    
    # Search for better solution
    start_time = time.time()
    deadline = start_time + 55
    
    # Vertex ordering: sort by degree descending to fail fast
    v_order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def solve_k(k):
        masks = [0] * k
        solution = [None]
        
        def dfs(idx, masks, num_parts_used):
            if time.time() > deadline:
                return False
                
            if idx == n:
                parts_found = []
                for i in range(num_parts_used):
                    m = masks[i]
                    verts = []
                    for j in range(n):
                        if m & (1 << j):
                            verts.append(j)
                    if verts:
                        parts_found.append(verts)
                solution[0] = parts_found
                return True
                
            v = v_order[idx]
            bit_v = 1 << v
            
            # Try adding to existing parts
            for i in range(num_parts_used):
                new_mask = masks[i] | bit_v
                if is_clique[new_mask] or is_indep[new_mask]:
                    masks[i] = new_mask
                    if dfs(idx + 1, masks, num_parts_used):
                        return True
                    masks[i] ^= bit_v
            
            # Try creating a new part
            if num_parts_used < k:
                i = num_parts_used
                masks[i] = bit_v
                if dfs(idx + 1, masks, num_parts_used + 1):
                    return True
                masks[i] = 0
                
            return False
            
        dfs(0, masks, 0)
        return solution[0]
        
    # Try to improve
    for k in range(1, best_num_parts):
        if time.time() > deadline:
            break
        sol = solve_k(k)
        if sol is not None:
            best_parts = sol
            best_num_parts = len(sol)
            break
            
    return best_parts
