import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 30
    best_edges = []
    # Seed with the known valid 30-vertex graph (10 triangles)
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
    # Try to find a graph with n > best_n
    for n in range(39, best_n, -1):
        if time.time() > deadline:
            break
            
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            neigh = [0] * n
            # Generate random graph
            # p=0.2 is a good start for n=39
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.2:
                        neigh[u] |= (1 << v)
                        neigh[v] |= (1 << u)
            
            # Prune C4s
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for u in range(n):
                    nu = neigh[u]
                    if nu == 0: continue
                    for v in range(u + 1, n):
                        nv = neigh[v]
                        if nv == 0: continue
                        inter = nu & nv
                        if inter & (inter - 1):
                            # Found C4
                            # Remove random edge
                            common = []
                            temp = inter
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                common.append(idx)
                                temp -= lsb
                            w = random.choice(common)
                            # Remove (u, w)
                            neigh[u] &= ~(1 << w)
                            neigh[w] &= ~(1 << u)
                            changed = True
                            break
                    if changed:
                        break
            
            # Rebuild edges
            final_edges = []
            for u in range(n):
                nu = neigh[u]
                while nu:
                    lsb = nu & -nu
                    v = lsb.bit_length() - 1
                    if u < v:
                        final_edges.append([u, v])
                    nu -= lsb
            
            # Check alpha < 11
            mask_all = (1 << n) - 1
            non_neigh = [0] * n
            for i in range(n):
                non_neigh[i] = mask_all & ~(neigh[i] | (1 << i))
                
            nodes_visited = 0
            def dfs(candidates_mask, count):
                nonlocal nodes_visited
                nodes_visited += 1
                if nodes_visited > 5000:
                    return None # Timeout/Abort
                if count == 11:
                    return True
                if candidates_mask == 0:
                    return False
                if candidates_mask.bit_count() < 11 - count:
                    return False
                
                # Heuristic: pick vertex with highest degree in candidate subgraph?
                # For simplicity, pick lowest index
                lsb = candidates_mask & -candidates_mask
                v = lsb.bit_length() - 1
                
                # Include v
                res = dfs(candidates_mask & non_neigh[v], count + 1)
                if res is not None:
                    return res
                
                # Exclude v
                res = dfs(candidates_mask & ~(1 << v), count)
                if res is not None:
                    return res
                
                return False

            res = dfs(mask_all, 0)
            if res is False:
                # No independent set of size 11 found
                # Valid graph
                if n > best_n:
                    best_n = n
                    best_edges = final_edges
                return {'n': n, 'edges': final_edges}
            # If res is True or None, try next instance

    return {'n': best_n, 'edges': best_edges}
