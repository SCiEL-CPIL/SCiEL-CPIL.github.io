import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Known good: K_{3,13} + 3 edges in the 3-side
    # Vertices 0,1,2 on one side, 3-15 on other
    # Edges: all (a,b) for a in {0,1,2}, b in {3..15} = 3*13=39
    # Plus (0,1),(0,2),(1,2) = 3 more = 42
    fallback = [[a, b] for a in range(3) for b in range(3, 16)]
    fallback += [[0,1],[0,2],[1,2]]
    
    # Precompute subset data
    eight_subsets = list(itertools.combinations(range(n), 8))
    subset_data = []
    for verts in eight_subsets:
        vset = set(verts)
        biparts = []
        for A in itertools.combinations(verts, 4):
            a_set = set(A)
            B = tuple(v for v in verts if v not in a_set)
            if A < B:
                biparts.append((A, B))
        subset_data.append(biparts)
    
    def is_q3(adj, A, B):
        zeros = set()
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                if not adj[a][b]:
                    zeros.add((i, j))
        for perm in itertools.permutations(range(4)):
            if all((i, perm[i]) in zeros for i in range(4)):
                return True
        return False
    
    def has_q3(adj):
        for biparts in subset_data:
            for A, B in biparts:
                if is_q3(adj, A, B):
                    return True
        return False
    
    best_edges = set(fallback)
    best_count = len(fallback)
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        p = random.uniform(0.5, 0.8)
        adj = [[0]*n for _ in range(n)]
        edge_set = set()
        for i, j in all_edges:
            if random.random() < p:
                adj[i][j] = 1
                adj[j][i] = 1
                edge_set.add((i,j))
        
        # Remove Q3s
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            found = False
            for biparts in subset_data:
                if time.time() > deadline - 15:
                    break
                for A, B in biparts:
                    if is_q3(adj, A, B):
                        q3_edges = [(a,b) for a in A for b in B if adj[a][b]]
                        if q3_edges:
                            a, b = random.choice(q3_edges)
                            adj[a][b] = 0
                            adj[b][a] = 0
                            edge_set.discard((a,b))
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        
        # Greedy add
        for i, j in random.sample(all_edges, len(all_edges)):
            if time.time() > deadline - 5:
                break
            if not adj[i][j]:
                adj[i][j] = 1
                adj[j][i] = 1
                edge_set.add((i,j))
                if has_q3(adj):
                    adj[i][j] = 0
                    adj[j][i] = 0
                    edge_set.discard((i,j))
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = set(edge_set)
    
    return [[i, j] for i, j in best_edges]
