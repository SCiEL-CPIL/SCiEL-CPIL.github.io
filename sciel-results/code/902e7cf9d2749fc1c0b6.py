import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    
    def can_add_edge(adj, u, v):
        nu = adj[u]
        nv = adj[v]
        cand_x = [x for x in nu if x != v and x not in nv]
        cand_z = [z for z in nv if z != u and z not in nu]
        if len(cand_x) < 2 or len(cand_z) < 2:
            return True
        for i in range(len(cand_x)):
            for j in range(i+1, len(cand_x)):
                x = cand_x[i]
                y = cand_x[j]
                if x in adj[y]:
                    continue
                for k in range(len(cand_z)):
                    for l in range(k+1, len(cand_z)):
                        z = cand_z[k]
                        w = cand_z[l]
                        if z in adj[w]:
                            continue
                        if (z in adj[x]) and (w in adj[x]) and (z in adj[y]) and (w in adj[y]):
                            return False
        return True

    def add_edge(adj, edges, u, v):
        adj[u].add(v)
        adj[v].add(u)
        edges.add((u, v))

    def remove_edge(adj, edges, u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        edges.remove((u, v))

    # Greedy construction
    adj = [set() for _ in range(n)]
    edges = set()
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            add_edge(adj, edges, u, v)
            
    # Local search with 1-2 exchange
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add a single edge
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edges:
                    continue
                if can_add_edge(adj, u, v):
                    add_edge(adj, edges, u, v)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try 1-2 exchange
        # Remove an edge, try to add two
        for a, b in list(edges):
            if time.time() > deadline:
                break
            remove_edge(adj, edges, a, b)
            
            # Try to add two edges
            added = []
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edges:
                        continue
                    if can_add_edge(adj, u, v):
                        add_edge(adj, edges, u, v)
                        added.append((u, v))
                        if len(added) == 2:
                            break
                if len(added) == 2:
                    break
                    
            if len(added) == 2:
                improved = True
            else:
                # Revert
                for u, v in added:
                    remove_edge(adj, edges, u, v)
                add_edge(adj, edges, a, b)
                
            if improved:
                break
                
    result = [[u, v] for u, v in edges]
    return result
