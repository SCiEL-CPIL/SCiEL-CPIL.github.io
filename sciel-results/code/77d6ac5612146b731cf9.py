import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
adj = [[] for _ in range(n)]
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    for k in range(3, 11):
        if time.time() > deadline:
            break
            
        colors = [-1] * n
        pairs = []
        pair_to_idx = {}
        idx = 0
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((i, j))
                pair_to_idx[(i, j)] = idx
                idx += 1
                
        parent = [[-1] * n for _ in range(len(pairs))]
        rank = [[0] * n for _ in range(len(pairs))]
        
        def find(idx, v):
            if parent[idx][v] != v:
                if parent[idx][v] == -1:
                    return -1
                parent[idx][v] = find(idx, parent[idx][v])
            return parent[idx][v]
        
        def union(idx, u, v):
            ru = find(idx, u)
            rv = find(idx, v)
            if ru == -1 or rv == -1:
                return False
            if ru == rv:
                return False
            if rank[idx][ru] < rank[idx][rv]:
                ru, rv = rv, ru
            parent[idx][rv] = ru
            if rank[idx][ru] == rank[idx][rv]:
                rank[idx][ru] += 1
            return True

        def backtrack():
            if time.time() > deadline:
                return False
                
            min_vertex = -1
            min_domain_size = n + 1
            for v in range(n):
                if colors[v] == -1:
                    valid_count = 0
                    for c in range(k):
                        is_valid = True
                        for u in adj[v]:
                            if colors[u] == c:
                                is_valid = False
                                break
                        if is_valid:
                            valid_count += 1
                            
                    if valid_count < min_domain_size:
                        min_domain_size = valid_count
                        min_vertex = v
                        if valid_count == 0:
                            return False
                        if valid_count == 1:
                            break
            
            if min_vertex == -1:
                return True
            
            v = min_vertex
            for c in range(k):
                is_valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        is_valid = False
                        break
                if not is_valid:
                    continue
                
                neighbor_colors = {}
                for u in adj[v]:
                    d = colors[u]
                    if d != -1 and d != c:
                        if d not in neighbor_colors:
                            neighbor_colors[d] = []
                        neighbor_colors[d].append(u)
                        
                cycle_detected = False
                for d, neighbors in neighbor_colors.items():
                    if len(neighbors) >= 2:
                        if c < d:
                            p_idx = pair_to_idx[(c, d)]
                        else:
                            p_idx = pair_to_idx[(d, c)]
                            
                        all_in_dsu = True
                        first_root = -1
                        for u in neighbors:
                            r = find(p_idx, u)
                            if r == -1:
                                all_in_dsu = False
                                break
                            if first_root == -1:
                                first_root = r
                            elif r != first_root:
                                all_in_dsu = False
                                break
                                
                        if all_in_dsu:
                            cycle_detected = True
                            break
                            
                if cycle_detected:
                    continue
                
                # Snapshot
                parent_snap = [row[:] for row in parent]
                rank_snap = [row[:] for row in rank]
                
                colors[v] = c
                
                # Update DSUs
                for d, neighbors in neighbor_colors.items():
                    if c < d:
                        p_idx = pair_to_idx[(c, d)]
                    else:
                        p_idx = pair_to_idx[(d, c)]
                        
                    if parent[p_idx][v] == -1:
                        parent[p_idx][v] = v
                        rank[p_idx][v] = 0
                        
                    for u in neighbors:
                        if parent[p_idx][u] == -1:
                            parent[p_idx][u] = u
                            rank[p_idx][u] = 0
                        union(p_idx, v, u)
                        
                if backtrack():
                    return True
                    
                # Restore
                parent[:] = [row[:] for row in parent_snap]
                rank[:] = [row[:] for row in rank_snap]
                colors[v] = -1
                
            return False
        
        if backtrack():
            return [int(c) for c in colors]
            
    return list(range(n))
