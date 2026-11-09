import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_8sets = list(combinations(range(16), 8))
    
    # Canonical Q3 bipartite graph rows (sorted)
    # M_canon rows:
    # 1010, 1101, 0111, 1011
    canon_rows = sorted([(1,0,1,0), (1,1,0,1), (0,1,1,1), (1,0,1,1)])
    
    def contains_q3_in_subset(adj, subset):
        s = set(subset)
        edge_count = 0
        for u in s:
            for v in s:
                if u < v and adj[u][v]:
                    edge_count += 1
        if edge_count < 12:
            return False
            
        for a_subset in combinations(subset, 4):
            a = set(a_subset)
            b = s - a
            
            indep_a = True
            for u in a:
                for v in a:
                    if u < v and adj[u][v]:
                        indep_a = False
                        break
                if not indep_a: break
            if not indep_a: continue
            
            indep_b = True
            for u in b:
                for v in b:
                    if u < v and adj[u][v]:
                        indep_b = False
                        break
                if not indep_b: break
            if not indep_b: continue
            
            a_list = list(a)
            b_list = list(b)
            
            M = [[0]*4 for _ in range(4)]
            for i, u in enumerate(a_list):
                for j, v in enumerate(b_list):
                    if adj[u][v]:
                        M[i][j] = 1
            
            # Check isomorphism
            for p in permutations(range(4)):
                M_p = [[M[i][p[j]] for j in range(4)] for i in range(4)]
                rows_p = sorted(tuple(row) for row in M_p)
                if rows_p == canon_rows:
                    return True
        return False

    def has_q3(adj):
        for subset in all_8sets:
            if contains_q3_in_subset(adj, subset):
                return True
        return False

    best_edges = 0
    best_adj = None
    
    for _ in range(10):
        if time.time() > deadline:
            break
        
        adj = [[False]*16 for _ in range(16)]
        for u in range(16):
            for v in range(u+1, 16):
                if random.random() < 0.7:
                    adj[u][v] = True
                    adj[v][u] = True
        
        while has_q3(adj):
            if time.time() > deadline:
                break
            found = False
            for subset in all_8sets:
                if contains_q3_in_subset(adj, subset):
                    s = set(subset)
                    edges_in_subset = []
                    for u in s:
                        for v in s:
                            if u < v and adj[u][v]:
                                edges_in_subset.append((u,v))
                    if edges_in_subset:
                        u, v = random.choice(edges_in_subset)
                        adj[u][v] = False
                        adj[v][u] = False
                        found = True
                        break
            if not found:
                break
        
        for u in range(16):
            for v in range(u+1, 16):
                if not adj[u][v]:
                    adj[u][v] = True
                    adj[v][u] = True
                    if has_q3(adj):
                        adj[u][v] = False
                        adj[v][u] = False
        
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [row[:] for row in adj]
            
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if best_adj[u][v]:
                edges.append([u, v])
    return edges
