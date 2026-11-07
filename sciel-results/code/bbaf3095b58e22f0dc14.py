import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = 15
    
    def get_degrees(adj):
        return [len(nei) for nei in adj]

    def is_valid(adj):
        deg = get_degrees(adj)
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # undirected, check once? No, adj is symmetric.
                # Actually, iterating u, v in adj[u] covers each edge twice if not careful, 
                # but for path s-u-v-t, u and v are ordered in the path.
                # We need to check all ordered pairs (u,v) where (u,v) in E.
                pass
        # Better implementation:
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def find_violation(adj):
        deg = get_degrees(adj)
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            return (s, u, v, t)
        return None

    def edge_count(adj):
        return sum(len(nei) for nei in adj) // 2

    def greedy_add(adj):
        # Try to add all possible missing edges
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        # Try adding
                        adj[u].add(v)
                        adj[v].add(u)
                        if not is_valid(adj):
                            adj[u].remove(v)
                            adj[v].remove(u)
                        else:
                            changed = True
        return adj

    best_edges = None
    best_count = 0
    
    # Try multiple random starts
    for p in [0.3, 0.5, 0.7, 0.8, 0.9]:
        for _ in range(20): # 20 restarts per p
            if time.time() > deadline:
                break
            
            # Init random graph
            adj = [set() for _ in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Repair
            # If invalid, remove edges from violations
            # To be safe, we might need to remove many edges.
            # Loop until valid
            max_repair_steps = 1000
            steps = 0
            while not is_valid(adj) and steps < max_repair_steps:
                viol = find_violation(adj)
                if viol is None:
                    break
                s, u, v, t = viol
                # Remove one of the edges (s,u), (u,v), (v,t)
                # Randomly pick one
                edges_to_consider = [(s, u), (u, v), (v, t)]
                # Ensure we pick a valid edge in the set representation
                # (s,u) is in adj[s] and adj[u]
                e = random.choice(edges_to_consider)
                a, b = e
                adj[a].remove(b)
                adj[b].remove(a)
                steps += 1
            
            if is_valid(adj):
                # Greedy add
                greedy_add(adj)
                count = edge_count(adj)
                if count > best_count:
                    best_count = count
                    # Convert to list of lists
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    best_edges = edges
    
    if best_edges is None:
        # Fallback to star
        best_edges = [[0, i] for i in range(1, n)]
    
    return best_edges
