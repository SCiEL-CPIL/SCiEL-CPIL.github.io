import time
import sys
import random

sys.setrecursionlimit(20000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj[i]).count('1')
        
    ALL = (1 << n) - 1
    deadline = time.time() + 85.0

    # --- 1. Find Connected Components ---
    comp_of = [-1] * n
    comps = []
    for s in range(n):
        if comp_of[s] != -1:
            continue
        cid = len(comps)
        mask = 0
        stack = [s]
        comp_of[s] = cid
        mask |= 1 << s
        while stack:
            x = stack.pop()
            nb = adj[x]
            while nb:
                lsb = nb & (-nb)
                nb ^= lsb
                y = lsb.bit_length() - 1
                if comp_of[y] == -1:
                    comp_of[y] = cid
                    mask |= lsb
                    stack.append(y)
        comps.append(mask)
    
    comps.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    best_len = 0
    best_path = []

    # --- 2. Heuristic Search (Greedy + Local Improve) ---
    def greedy_path(start):
        path = [start]
        seen = 1 << start
        while True:
            cand = adj[path[-1]] & ~seen
            if not cand:
                break
            best_u = -1
            best_score = float('inf')
            m = cand
            while m:
                lsb = m & (-m)
                m ^= lsb
                u = lsb.bit_length() - 1
                score = bin(adj[u] & ~seen & ~(1 << u)).count('1')
                if score < best_score:
                    best_score = score
                    best_u = u
            path.append(best_u)
            seen |= 1 << best_u
        return path

    def local_improve(path):
        if not path:
            return path
        seen_mask = 0
        for v in path:
            seen_mask |= 1 << v
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Extend ends
            for end_idx in [0, -1]:
                end_v = path[end_idx]
                cand = adj[end_v] & ~seen_mask
                if cand:
                    best_u = -1
                    best_score = float('inf')
                    m = cand
                    while m:
                        lsb = m & (-m)
                        m ^= lsb
                        u = lsb.bit_length() - 1
                        score = bin(adj[u] & ~seen_mask & ~(1 << u)).count('1')
                        if score < best_score:
                            best_score = score
                            best_u = u
                    if best_u != -1:
                        if end_idx == 0:
                            path.insert(0, best_u)
                        else:
                            path.append(best_u)
                        seen_mask |= 1 << best_u
                        improved = True
                        break # Restart loop to re-evaluate other end/insertions

            if improved:
                continue

            # Try Insert
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                cand = adj[u] & adj[v] & ~seen_mask
                if cand:
                    lsb = cand & (-cand)
                    w = lsb.bit_length() - 1
                    path.insert(i+1, w)
                    seen_mask |= 1 << w
                    improved = True
                    break

            if improved:
                continue

            # Try 2-Opt (Reverse segment) to potentially allow extension?
            # This is complex to verify if it helps extension immediately.
            # Skip for now to save time.
            
        return path

    if comps:
        main_comp = comps[0]
        verts_in_comp = []
        m = main_comp
        while m:
            lsb = m & (-m)
            m ^= lsb
            verts_in_comp.append(lsb.bit_length() - 1)
        
        # Try greedy from all vertices, sorted by degree
        verts_in_comp.sort(key=lambda x: deg[x])
        
        for s in verts_in_comp:
            if time.time() > deadline:
                break
            p = greedy_path(s)
            p = local_improve(p)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p

    # --- 3. Branch and Bound ---
    if comps and best_len < bin(comps[0]).count('1'):
        main_comp = comps[0]
        
        start_verts = []
        m = main_comp
        while m:
            lsb = m & (-m)
            m ^= lsb
            v = lsb.bit_length() - 1
            start_verts.append(v)
            
        start_verts.sort(key=lambda x: deg[x])
        
        node_count = 0
        NODE_LIMIT = 10000000 # Increased limit
        
        # Helper: Check if unvisited vertices reachable from v cover enough
        def prune_check(v, visited_mask, path_len):
            unvisited_in_comp = main_comp & ~visited_mask
            if unvisited_in_comp == 0:
                return True # Can't prune, actually found full path
            
            # BFS from v in unvisited_in_comp
            seen = 0
            frontier = adj[v] & unvisited_in_comp
            seen |= frontier
            while frontier:
                lsb = frontier & (-frontier)
                frontier ^= lsb
                u = lsb.bit_length() - 1
                new_frontier = adj[u] & unvisited_in_comp & ~seen
                seen |= new_frontier
                frontier |= new_frontier
            
            reachable_count = bin(seen).count('1')
            if path_len + reachable_count <= best_len:
                return False # Prune
            return True

        def dfs(v, visited_mask, path):
            nonlocal best_len, best_path, node_count
            
            node_count += 1
            if time.time() > deadline or node_count > NODE_LIMIT:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
            if not prune_check(v, visited_mask, len(path)):
                return True

            cand = adj[v] & (main_comp & ~visited_mask)
            if not cand:
                return True
            
            lst = []
            m = cand
            while m:
                lsb = m & (-m)
                m ^= lsb
                u = lsb.bit_length() - 1
                score = bin(adj[u] & (main_comp & ~visited_mask) & ~(1 << u)).count('1')
                lst.append((score, u))
            
            lst.sort()
            
            for score, u in lst:
                path.append(u)
                if not dfs(u, visited_mask | (1 << u), path):
                    path.pop()
                    return False
                path.pop()
                
            return True

        for s in start_verts:
            if time.time() > deadline or node_count > NODE_LIMIT:
                break
            # Only start DFS from vertices that are not "too high degree" initially?
            # Or just try all.
            dfs(s, 1 << s, [s])

    if not best_path:
        best_path = [0]
        
    return best_path
