import time
import random
import math

N = 15
DEADLINE = 85.0

def _is_valid(adj, n):
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if s == t:
                            continue
                        if deg[s] == deg[t]:
                            return False
    return True

def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def construct():
    start = time.time()
    
    # Start with K_{7,8}
    adj = [set() for _ in range(N)]
    for i in range(7):
        for j in range(7, N):
            adj[i].add(j)
            adj[j].add(i)
            
    best_adj = [set(s) for s in adj]
    best_count = _edge_count(best_adj, N)
    
    rng = random.Random(42)
    
    # Simulated Annealing / Local Search
    # We want to maximize edges.
    # Moves: Add edge, Remove edge.
    # Accept if valid.
    # If invalid, revert.
    
    current_adj = [set(s) for s in adj]
    current_count = best_count
    
    for step in range(100000):
        if time.time() > DEADLINE:
            break
            
        i = rng.randint(0, N - 1)
        j = rng.randint(0, N - 1)
        if i == j:
            continue
            
        if j in current_adj[i]:
            # Remove
            current_adj[i].discard(j)
            current_adj[j].discard(i)
        else:
            # Add
            current_adj[i].add(j)
            current_adj[j].add(i)
            
        if _is_valid(current_adj, N):
            new_count = _edge_count(current_adj, N)
            if new_count > best_count:
                best_count = new_count
                best_adj = [set(s) for s in current_adj]
        else:
            # Revert
            if j in current_adj[i]:
                current_adj[i].discard(j)
                current_adj[j].discard(i)
            else:
                current_adj[i].add(j)
                current_adj[j].add(i)
                
    # Final check
    if not _is_valid(best_adj, N):
        # Fallback
        best_adj = [set() for _ in range(N)]
        for i in range(7):
            for j in range(7, N):
                best_adj[i].add(j)
                best_adj[j].add(i)
        best_count = 56

    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges
