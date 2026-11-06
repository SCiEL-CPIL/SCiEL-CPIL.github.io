import random
import time

def construct():
    n = 40
    deadline = time.time() + 145.0
    
    def creates_c6(u, v, adj):
        # Check if adding (u, v) creates a C6
        # This is true if there is a path of length 5 between u and v
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor, depth + 1, visited):
                        return True
                    visited.remove(neighbor)
            return False
            
        visited = {u}
        return dfs(u, 0, visited)

    best_edges = 0
    best_graph = []

    for _ in range(1000):
        if time.time() > deadline:
            break
            
        adj = [set() for _ in range(n)]
        edges = []
        
        # Random initialization
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.2:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append((i, j))
        
        # Greedy add
        for i in range(n):
            if time.time() > deadline: break
            for j in range(i+1, n):
                if j not in adj[i]:
                    if not creates_c6(i, j, adj):
                        adj[i].add(j)
                        adj[j].add(i)
                        edges.append((i, j))
                        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if not edges:
                break
                
            u, v = random.choice(edges)
            adj[u].remove(v)
            adj[v].remove(u)
            edges.remove((u, v))
            
            added = 0
            for i in range(n):
                if time.time() > deadline: break
                for j in range(i+1, n):
                    if j not in adj[i]:
                        if not creates_c6(i, j, adj):
                            adj[i].add(j)
                            adj[j].add(i)
                            edges.append((i, j))
                            added += 1
                            
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_graph = [list(e) for e in edges]
                
            if added == 0:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                break

    return best_graph
