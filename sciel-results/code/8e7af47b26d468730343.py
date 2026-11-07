import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    def build_adj(added):
        adj = [[] for _ in range(n)]
        mat = [[False] * n for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
            mat[u][v] = mat[v][u] = True
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
            mat[u][v] = mat[v][u] = True
        return adj, mat
    
    def is_triangle_free(adj, mat):
        for u in range(n):
            au = adj[u]
            for i in range(len(au)):
                a = au[i]
                rowa = mat[a]
                for j in range(i + 1, len(au)):
                    b = au[j]
                    if rowa[b]:
                        return False
        return True
    
    def get_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                if dist[node] >= 4:
                    continue
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
                        if dist[nb] > max_d:
                            max_d = dist[nb]
                            if max_d > 4:
                                return max_d
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            if -1 in dist:
                return 999
        return max_d
    
    def valid_config(added):
        if len(added) != len(set(tuple(sorted(e)) for e in added)):
            return False
        es = set(tuple(sorted(e)) for e in added)
        for e in es:
            if e in base_edges:
                return False
        adj, mat = build_adj(added)
        if not is_triangle_free(adj, mat):
            return False
        if get_diameter(adj) > 4:
            return False
        return True
    
    def reduce(added, deadline):
        added_set = set(tuple(sorted(e)) for e in added)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in list(added_set):
                trial = added_set - {e}
                if valid_config(list(trial)):
                    added_set = trial
                    improved = True
                    break
        return added_set
    
    def local_search(added, deadline):
        added_set = set(tuple(sorted(e)) for e in added)
        best_set = set(added_set)
        
        while time.time() < deadline - 2:
            added_set = reduce(added_set, deadline)
            if len(added_set) < len(best_set):
                best_set = set(added_set)
            
            improved = False
            e_list = list(added_set)
            for e in e_list:
                if time.time() > deadline - 2:
                    break
                for u in range(n):
                    for v in range(u + 1, n):
                        new_e = (u, v)
                        if new_e in base_edges or new_e in added_set:
                            continue
                        trial = (added_set - {e}) | {new_e}
                        if valid_config(list(trial)):
                            added_set = trial
                            if len(added_set) < len(best_set):
                                best_set = set(added_set)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_set
    
    deadline = time.time() + 80.0
    
    # Hub-based seeds
    seeds = []
    for hub in range(n):
        if hub % 2 == 0:
            candidates_v = [v for v in range(n) if v % 2 == 1 and v != hub]
        else:
            candidates_v = [v for v in range(n) if v % 2 == 0 and v != hub]
        added = [[hub, v] for v in candidates_v]
        added = [[u, v] for u, v in added if tuple(sorted((u, v))) not in base_edges]
        seeds.append(added)
    
    # Known good
    seeds.append([[11, 0], [11, 2], [11, 4], [11, 6], [11, 8], [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]])
    
    best = None
    best_count = 100
    
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
        
        reduced = reduce(set(tuple(sorted(e)) for e in seed), deadline - 5.0)
        
        if len(reduced) < best_count:
            best = list(reduced)
            best_count = len(reduced)
    
    if best is not None and time.time() < deadline - 5.0:
        polished = local_search(best, deadline)
        if len(polished) < best_count:
            best = list(polished)
            best_count = len(polished)
    
    if best is None:
        best = [[11, 0], [11, 2], [11, 4], [11, 6], [11, 8], [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]]
    
    return [[int(u), int(v)] for u, v in best]
# EVOLVE-BLOCK-END
