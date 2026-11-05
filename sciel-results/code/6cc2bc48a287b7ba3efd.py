import time
import random

def construct():
    n = 20
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Use bit_count if available, else bin count
    if hasattr(int, "bit_count"):
        def popcount(x):
            return x.bit_count()
    else:
        def popcount(x):
            return bin(x).count('1')

    def is_valid_add(adj, u, v):
        # Check if adding (u, v) creates a K3,3
        # 1. Triples containing u: A = {u, x, y}
        #    v must be connected to x and y.
        #    So x, y in N(v).
        nv = adj[v]
        # Iterate pairs in N(v)
        neighbors_v = [i for i in range(n) if (nv >> i) & 1]
        for i in range(len(neighbors_v)):
            x = neighbors_v[i]
            if x == u:
                continue
            for j in range(i + 1, len(neighbors_v)):
                y = neighbors_v[j]
                if y == u:
                    continue
                # A = {u, x, y}
                # Check if |N_old(A)| >= 2
                # N_old(A) = N(u) & N(x) & N(y)
                na = adj[u] & adj[x] & adj[y]
                if popcount(na) >= 2:
                    return False
        
        # 2. Triples containing v: A = {v, x, y}
        #    u must be connected to x and y.
        #    So x, y in N(u).
        nu = adj[u]
        neighbors_u = [i for i in range(n) if (nu >> i) & 1]
        for i in range(len(neighbors_u)):
            x = neighbors_u[i]
            if x == v:
                continue
            for j in range(i + 1, len(neighbors_u)):
                y = neighbors_u[j]
                if y == v:
                    continue
                # A = {v, x, y}
                # Check if |N_old(A)| >= 2
                # N_old(A) = N(v) & N(x) & N(y)
                na = adj[v] & adj[x] & adj[y]
                if popcount(na) >= 2:
                    return False
            
        return True

    def edge_count(adj):
        count = 0
        for i in range(n):
            count += popcount(adj[i])
        return count // 2

    def greedy_add(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_valid_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    def local_search(adj):
        best_adj = [x for x in adj]
        best_count = edge_count(best_adj)
        
        while time.time() < deadline - 5.0:
            added = False
            # Try to add any missing edge
            # Shuffle order to avoid bias?
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
            
            if added:
                current_count = edge_count(adj)
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [x for x in adj]
            else:
                # Stuck, kick
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append((u, v))
                
                if not edges:
                    break
                
                u, v = random.choice(edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Greedy add again
                greedy_add(adj)
                
                current_count = edge_count(adj)
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [x for x in adj]
                
        return best_adj, best_count

    best_global_adj = [0] * n
    best_global_count = 0
    
    # Multiple restarts
    while time.time() < deadline - 5.0:
        adj = [0] * n
        greedy_add(adj)
        adj, count = local_search(adj)
        
        if count > best_global_count:
            best_global_count = count
            best_global_adj = adj
            
        if best_global_count == 88:
            break

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_global_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
