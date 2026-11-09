import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 15
    # Complete graph edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible = len(all_edges)  # 105
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def is_valid(adj):
        """Check if any 3-edge path s-u-v-t has deg(s)==deg(t)"""
        degs = get_degrees(adj)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Path s-u-v-t: s in adj[u], s!=v; t in adj[v], t!=u
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if s == t:
                            continue
                        if degs[s] == degs[t]:
                            return False
        return True
    
    # Start with complete graph, remove edges to make valid
    # Greedy: start complete, find violations, remove edges
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    # Find a valid subgraph by removing edges
    # Use iterative removal: find a violating path, remove an edge from it
    removed = 0
    while not is_valid(adj):
        if time.time() > deadline:
            break
        # Find a violation
        degs = get_degrees(adj)
        found = False
        for u in range(n):
            if found:
                break
            for v in list(adj[u]):
                if v <= u:
                    continue
                for s in list(adj[u]):
                    if s == v or found:
                        continue
                    for t in list(adj[v]):
                        if t == u or s == t or found:
                            continue
                        if degs[s] == degs[t]:
                            # Remove edge (u,v) or (s,u) or (v,t)
                            # Remove (u,v) as it's the middle edge
                            adj[u].discard(v)
                            adj[v].discard(u)
                            removed += 1
                            found = True
                            break
        if not found:
            break
    
    # Try to add back removed edges
    # Actually, let's do a more systematic approach
    
    # Better approach: start empty, add edges greedily
    adj2 = [set() for _ in range(n)]
    
    # Order edges by some heuristic - try to add edges that create diverse degrees
    random.seed(42)
    edge_order = all_edges[:]
    random.shuffle(edge_order)
    
    # Multiple random restarts
    best_edges = 0
    best_adj = [set() for _ in range(n)]
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        adj_t = [set() for _ in range(n)]
        random.shuffle(edge_order)
        
        for (u, v) in edge_order:
            if time.time() > deadline - 5:
                break
            # Try adding edge
            adj_t[u].add(v)
            adj_t[v].add(u)
            if is_valid(adj_t):
                pass  # keep it
            else:
                adj_t[u].discard(v)
                adj_t[v].discard(u)
        
        cnt = sum(len(adj_t[i]) for i in range(n)) // 2
        if cnt > best_edges:
            best_edges = cnt
            best_adj = [set(adj_t[i]) for i in range(n)]
    
    # Try to improve best_adj by adding more edges
    for trial in range(20):
        if time.time() > deadline - 2:
            break
        improved = False
        for (u, v) in all_edges:
            if time.time() > deadline - 2:
                break
            if v not in best_adj[u]:
                best_adj[u].add(v)
                best_adj[v].add(u)
                if is_valid(best_adj):
                    improved = True
                else:
                    best_adj[u].discard(v)
                    best_adj[v].discard(u)
    
    # Build result
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Verify
    if not is_valid(best_adj):
        # Fallback: star graph (always valid since center has unique degree)
        edges = [[0, v] for v in range(1, n)]
    
    # Ensure types
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
