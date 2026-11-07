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

def _try_add_edge(adj, i, j):
    """Try adding edge (i,j) and check validity. Returns True if added."""
    adj[i].add(j)
    adj[j].add(i)
    if _is_valid(adj, N):
        return True
    adj[i].discard(j)
    adj[j].discard(i)
    return False

def _greedy_build(start_adj, deadline):
    """Greedy: try to add all possible missing edges in order."""
    adj = [set(s) for s in start_adj]
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i + 1, N):
                if j not in adj[i]:
                    if _try_add_edge(adj, i, j):
                        improved = True
    return adj

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
    
    # Try greedy completion from the bipartite start
    greedy_adj = _greedy_build(adj, DEADLINE - 5.0)
    greedy_count = _edge_count(greedy_adj, N)
    if greedy_count > best_count:
        best_count = greedy_count
        best_adj = [set(s) for s in greedy_adj]
    
    rng = random.Random(42)
    
    # Local search from best_adj
    current_adj = [set(s) for s in best_adj]
    current_count = best_count
    
    for step in range(200000):
        if time.time() > DEADLINE - 2.0:
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
                
    # Try greedy completion from best_adj
    greedy_adj2 = _greedy_build(best_adj, DEADLINE - 1.0)
    greedy_count2 = _edge_count(greedy_adj2, N)
    if greedy_count2 > best_count:
        best_count = greedy_count2
        best_adj = [set(s) for s in greedy_adj2]

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
