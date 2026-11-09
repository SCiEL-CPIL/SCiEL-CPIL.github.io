"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    def _precompute_path4(v, adj_list):
        """
        Precompute Path4[c][i][d] for all c, i, d < v.
        Path4[c][i][d] is True if there is a simple path of length 4 
        between i and d in the graph G_c induced by vertices 0..v-1.
        """
        Path4 = [ [[False]*v for _ in range(v)] for _ in range(4) ]
        for c in range(4):
            for i in range(v):
                if not adj_list[c][i]:
                    continue
                for d in range(v):
                    if i == d:
                        continue
                    found = False
                    # Path i-a-b-e-d
                    for a in adj_list[c][i]:
                        if a == d:
                            continue
                        for b in adj_list[c][a]:
                            if b == i or b == d or b == a:
                                continue
                            for e in adj_list[c][b]:
                                if e == i or e == d or e == a or e == b:
                                    continue
                                if d in adj_list[c][e]:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    Path4[c][i][d] = found
        return Path4

    def _constrained_backtrack(n_target, deadline):
        adj = [[-1]*n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        
        def _search(v):
            if time.time() > deadline:
                return False
            
            if v == n_target:
                return True
            
            if v == 0:
                return _search(v + 1)
            
            # Precompute Path4 for the current subgraph 0..v-1
            Path4 = _precompute_path4(v, adj_list)
            
            def _assign(idx):
                if time.time() > deadline:
                    return False
                
                if idx == v:
                    return _search(v + 1)
                
                i = idx
                for c in range(4):
                    # Check if setting adj[i][v]=c creates a C6
                    valid = True
                    for d in range(i):
                        if adj[d][v] == c:
                            if Path4[c][i][d]:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)
                    
                    if _assign(idx + 1):
                        return True
                    
                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)
                
                return False
            
            return _assign(0)
        
        if _search(0):
            colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    colors.append(adj[i][j])
            return colors
        return None

    def _verify(colors, n):
        if n < 6:
            return True
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        for c in range(4):
            for subset in combinations(range(n), 6):
                if not hasattr(_verify, '_cycles'):
                    cycles = []
                    seen = set()
                    for perm in permutations(range(1, 6)):
                        cycle = (0,) + perm
                        edges = frozenset(frozenset((cycle[k], cycle[(k+1)%6])) for k in range(6))
                        if edges not in seen:
                            seen.add(edges)
                            cycles.append(cycle)
                    _verify._cycles = cycles
                
                for cycle in _verify._cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k+1)%6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return False
        return True

    def _local_search(colors, n, deadline, max_iter=5000):
        """
        Local search: try to reduce C6 count by flipping edge colors.
        Returns (colors, n) where colors is valid if no C6.
        """
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        def _has_c6_fast(c, u, v):
            """Check if there is a C6 containing edge (u,v) of color c."""
            # We need a path of length 5 from u to v in G_c not using edge (u,v)
            # Path: u-a-b-d-e-v
            # a in N(u), e in N(v)
            # We need a path of length 3 from a to e in G_c \ {u, v}
            
            # Simple DFS for path of length 3 from a to e
            def has_path_len3(start, end, forbidden):
                # start -> x -> y -> end
                for x in adj_list[c][start]:
                    if x in forbidden or x == end:
                        continue
                    for y in adj_list[c][x]:
                        if y in forbidden or y == start or y == end:
                            continue
                        if end in adj_list[c][y]:
                            return True
                return False

            forbidden = {u, v}
            for a in adj_list[c][u]:
                if a == v:
                    continue
                for e in adj_list[c][v]:
                    if e == u or e == a:
                        continue
                    if has_path_len3(a, e, forbidden | {a, e}):
                        return True
            return False

        def _count_c6():
            count = 0
            for c in range(4):
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u][v] == c:
                            if _has_c6_fast(c, u, v):
                                count += 1
            return count
        
        # Simulated annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        best_colors = colors[:]
        best_count = _count_c6()
        current_count = best_count
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = adj[i][j]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Apply change
            adj[i][j] = new_c
            adj[j][i] = new_c
            adj_list[old_c][i].remove(j)
            adj_list[old_c][j].remove(i)
            adj_list[new_c][i].add(j)
            adj_list[new_c][j].add(i)
            
            new_count = _count_c6()
            delta = new_count - current_count
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_colors = []
                    idx = 0
                    for ii in range(n):
                        for jj in range(ii+1, n):
                            best_colors.append(adj[ii][jj])
            else:
                # Revert
                adj[i][j] = old_c
                adj[j][i] = old_c
                adj_list[new_c][i].remove(j)
                adj_list[new_c][j].remove(i)
                adj_list[old_c][i].add(j)
                adj_list[old_c][j].add(i)
            
            T = max(T_min, T * cooling)
        
        return best_colors, n

    best_n = 1
    best_colors = []
    
    # Try backtracking for increasing n
    for n_target in range(2, 19):
        if time.time() > deadline - 15:
            break
        
        colors = _constrained_backtrack(n_target, deadline - 10)
        if colors is not None:
            if _verify(colors, n_target):
                best_n = n_target
                best_colors = colors
            else:
                pass
        else:
            # Backtracking failed, try local search
            if time.time() > deadline - 15:
                break
            random_colors = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
            ls_colors, ls_n = _local_search(random_colors, n_target, deadline - 10, max_iter=3000)
            if _verify(ls_colors, ls_n):
                best_n = ls_n
                best_colors = ls_colors
    
    # Final attempt: try to grow from best_n using local search
    if best_n >= 6 and time.time() < deadline - 20:
        final_colors, final_n = _local_search(best_colors, best_n, deadline - 10, max_iter=5000)
        if _verify(final_colors, final_n) and final_n > best_n:
            best_n = final_n
            best_colors = final_colors
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
