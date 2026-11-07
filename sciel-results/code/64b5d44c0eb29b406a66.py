import time
import random
import math

N = 15
DEADLINE = 85.0

def _is_valid(adj, n):
    """
    Check if no simple path s-u-v-t (s != t) has deg(s) == deg(t).
    Equivalent: For every edge (u, v), and every s in N(u)\{v}, t in N(v)\{u} with s != t, deg(s) != deg(t).
    """
    deg = [len(adj[i]) for i in range(n)]
    # Iterate over all edges
    for u in range(n):
        for v in adj[u]:
            if u < v: # Avoid double checking
                # Check all s in N(u) \ {v}
                for s in adj[u]:
                    if s == v:
                        continue
                    # Check all t in N(v) \ {u}
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

def _make_bipartite(n, a, b):
    """Create K_{a,b} on n vertices."""
    adj = [set() for _ in range(n)]
    # Part A: 0..a-1, Part B: a..n-1
    for i in range(a):
        for j in range(a, n):
            adj[i].add(j)
            adj[j].add(i)
    return adj

def _simulated_annealing(adj, n, seed, max_iter, t_start, t_end):
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_edges = _edge_count(adj, n)
    current_valid = _is_valid(adj, n)
    
    # If start is invalid, try to repair or restart
    if not current_valid:
        # Just return empty or random valid
        pass 

    best_adj = [set(s) for s in adj]
    best_edges = current_edges if current_valid else 0
    
    # Score: maximize edges. Validity is hard constraint.
    # We only accept moves that are valid.
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
            
        if j in adj[i]:
            # Remove
            adj[i].discard(j)
            adj[j].discard(i)
        else:
            # Add
            adj[i].add(j)
            adj[j].add(i)
            
        if _is_valid(adj, n):
            new_edges = _edge_count(adj, n)
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [set(s) for s in adj]
        else:
            # Revert if invalid
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
                
    return best_adj

def construct():
    start = time.time()
    
    # Start with K_{7,8}
    best_adj = _make_bipartite(N, 7, 8)
    best_count = _edge_count(best_adj, N)
    
    # Verify
    if not _is_valid(best_adj, N):
        # Fallback to star
        best_adj = [set() for _ in range(N)]
        for v in range(1, N):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = 14

    # Try to improve with SA
    # Multiple restarts with different seeds
    for seed in range(20):
        if time.time() - start > DEADLINE - 20:
            break
            
        # Perturb best_adj
        perturbed = [set(s) for s in best_adj]
        rng = random.Random(seed)
        # Remove 1-2 random edges to allow different structure
        for _ in range(rng.randint(1, 3)):
            all_edges = []
            for i in range(N):
                for j in perturbed[i]:
                    if i < j:
                        all_edges.append((i, j))
            if all_edges:
                a, b = rng.choice(all_edges)
                perturbed[a].discard(b)
                perturbed[b].discard(a)
                
        result = _simulated_annealing(perturbed, N, 1000 + seed, 5000, 1.0, 0.1)
        
        cnt = _edge_count(result, N)
        if cnt > best_count and _is_valid(result, N):
            best_count = cnt
            best_adj = result
            
    # Final check
    if not _is_valid(best_adj, N):
        best_adj = [set() for _ in range(N)]
        for v in range(1, N):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = 14

    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges
