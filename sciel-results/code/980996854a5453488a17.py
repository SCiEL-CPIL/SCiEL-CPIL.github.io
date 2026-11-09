"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def check_c4(adj, u, v):
        """Check if adding edge (u,v) creates a C4 in graph adj."""
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[v]:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    return True
        return False
    
    def is_c4_free(adj):
        """Verify graph is C4-free."""
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for a in common:
                        for b in common:
                            if a < b and b in adj[a]:
                                return False
        return True
    
    best_count = 0
    best_edges = []
    
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        adj = [set() for _ in range(n)]
        edge_set = set()
        edges = []
        
        # Greedy phase with random ordering
        order = list(all_edges)
        random.shuffle(order)
        
        for (u, v) in order:
            if time.time() > deadline - 5:
                break
            if not check_c4(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                edges.append((u, v))
        
        # Local search
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            non_edges = [e for e in all_edges if e not in edge_set]
            if not non_edges:
                break
            
            (a, b) = random.choice(non_edges)
            if not check_c4(adj, a, b):
                adj[a].add(b)
                adj[b].add(a)
                edge_set.add((a, b))
                edges.append((a, b))
            else:
                if len(edges) < 2:
                    continue
                (u, v) = random.choice(edges)
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((u, v))
                edges.remove((u, v))
                
                new_edges = []
                if not check_c4(adj, a, b):
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((a, b))
                    new_edges.append((a, b))
                
                for (x, y) in non_edges:
                    if (x, y) in edge_set or (x, y) == (a, b):
                        continue
                    if not check_c4(adj, x, y):
                        adj[x].add(y)
                        adj[y].add(x)
                        edge_set.add((x, y))
                        new_edges.append((x, y))
                        break
                
                if len(new_edges) >= 2:
                    edges.extend(new_edges)
                else:
                    for (ne_u, ne_v) in new_edges:
                        adj[ne_u].discard(ne_v)
                        adj[ne_v].discard(ne_u)
                        edge_set.discard((ne_u, ne_v))
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
                    edges.append((u, v))
        
        # Verify C4-free
        if is_c4_free(adj) and len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
        
        if best_count >= 48:
            break
    
    if not best_edges:
        return [[0, v, 0] for v in range(1, n)]
    
    return [[u, v, 0] for (u, v) in best_edges]

# EVOLVE-BLOCK-END
