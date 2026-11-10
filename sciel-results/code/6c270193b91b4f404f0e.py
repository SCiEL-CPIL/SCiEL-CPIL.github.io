import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    PERMS = list(itertools.permutations(range(4)))
    
    # Initialize with seed graph
    adj = np.zeros((16, 16), dtype=np.int8)
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = 1
            adj[b][a] = 1
            
    best_adj = adj.copy()
    best_edges = np.sum(best_adj) // 2
    
    def has_q3_in_pair(U, V, adj):
        # U, V are tuples of 4
        # Check if missing edges contain a perfect matching
        # Missing edge (i,j) if adj[U[i]][V[j]] == 0
        # We need a permutation p such that adj[U[i]][V[p[i]]] == 0 for all i
        for p in PERMS:
            ok = True
            for i in range(4):
                if adj[U[i]][V[p[i]]] != 0:
                    ok = False
                    break
            if ok:
                return True
        return False

    def is_safe_to_add(u, v, adj):
        other_vertices = [x for x in range(16) if x != u and x != v]
        for S in itertools.combinations(other_vertices, 6):
            S = list(S) + [u, v]
            # S has 8 vertices
            # We need to check splits where u, v are in different parts
            # Fix u in U. Then v must be in V.
            # U must contain u. V must contain v.
            # U has 3 other vertices from S \ {u, v}.
            # V has the remaining 3 vertices from S \ {u, v} plus v.
            # So we just need to choose 3 vertices from the 6 others to join u in U.
            others = [x for x in S if x != u and x != v]
            for U_rest in itertools.combinations(others, 3):
                U = tuple([u] + list(U_rest))
                V = tuple([v] + [x for x in others if x not in U_rest])
                if has_q3_in_pair(U, V, adj):
                    return False
        return True

    # Local search
    for iteration in range(100):
        if time.time() > deadline:
            break
            
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            missing = [(u, v) for u in range(16) for v in range(u+1, 16) if adj[u][v] == 0]
            random.shuffle(missing)
            for u, v in missing:
                if is_safe_to_add(u, v, adj):
                    adj[u][v] = 1
                    adj[v][u] = 1
                    changed = True
        
        current_edges = np.sum(adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()
            
        if not changed:
             if current_edges > 0:
                 edges = [(u, v) for u in range(16) for v in range(u+1, 16) if adj[u][v] == 1]
                 if edges:
                     u, v = random.choice(edges)
                     adj[u][v] = 0
                     adj[v][u] = 0
                     continue
             else:
                 break

    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if best_adj[u][v] == 1:
                edges.append([u, v])
                
    return edges
