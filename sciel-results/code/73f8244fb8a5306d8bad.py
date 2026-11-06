"""Model-and-optimize approach for the 16-vertex, 65-edge triangle degree-sum problem."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_DEADLINE_GUARD = 130.0  # safety margin

def _compute_score(adj, n=16, edges=None):
    """Compute max triangle degree-sum. adj is a list of sets."""
    deg = [len(adj[i]) for i in range(n)]
    if edges is None:
        edges = set()
    best = 0
    for i in range(n):
        for j in adj[i]:
            if j > i:
                common = adj[i] & adj[j]
                for k in common:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _local_search(adj, n, edge_set, max_iters=200000):
    """Local search: flip edges to maintain exactly 65 edges, minimize max triangle degree-sum."""
    deg = [len(adj[i]) for i in range(n)]
    best_score = _compute_score(adj, n)
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edge_set)
    
    random.seed(42)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for it in range(max_iters):
        if it % 500 == 0 and time.time() > _DEADLINE_GUARD:
            break
        
        # Try to find a beneficial flip
        # Pick a random existing edge to remove
        existing = list(edge_set)
        if not existing:
            break
        e_remove = random.choice(existing)
        u, v = e_remove
        if u > v:
            u, v = v, u
        
        # Pick a random non-existing edge to add
        # Use rejection sampling for speed
        for _try in range(50):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) == (u, v):
                continue
            if (a, b) in edge_set:
                continue
            e_add = (a, b)
            break
        else:
            continue
        
        # Apply the flip
        adj[u].remove(v)
        adj[v].remove(u)
        edge_set.remove((u, v))
        
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))
        
        # Recompute score (approximate: only triangles affected)
        # Affected vertices: u, v, a, b and their neighbors
        affected = set([u, v, a, b])
        affected.update(adj[u])
        affected.update(adj[v])
        affected.update(adj[a])
        affected.update(adj[b])
        
        new_score = 0
        for i in affected:
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        s = len(adj[i]) + len(adj[j]) + len(adj[k])
                        if s > new_score:
                            new_score = s
        
        # Also need to check if we missed something - do full check periodically
        if new_score < best_score:
            best_score = new_score
            best_adj = [set(s) for s in adj]
            best_edge_set = set(edge_set)
        else:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
            adj[a].remove(b)
            adj[b].remove(a)
            edge_set.remove((a, b))
    
    return best_score, best_adj, best_edge_set

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Start with K_{6,10} + 5 matching edges in B
    # Part A: 0-5 (size 6), Part B: 6-15 (size 10)
    edge_set = set()
    for u in range(6):
        for v in range(6, 16):
            edge_set.add((u, v))
    for a, b in [(6,7),(8,9),(10,11),(12,13),(14,15)]:
        edge_set.add((a, b))
    
    adj = _build_adj(n, edge_set)
    best_score = _compute_score(adj, n)
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edge_set)
    
    # Try multiple random starts and local searches
    for trial in range(10):
        if time.time() > _DEADLINE_GUARD:
            break
        
        # Create a perturbed starting point
        if trial == 0:
            adj_cur = [set(s) for s in adj]
            es_cur = set(edge_set)
        else:
            adj_cur = [set(s) for s in best_adj]
            es_cur = set(best_edge_set)
            # Perturb: flip k random edges
            k = random.randint(5, 15)
            for _ in range(k):
                existing = list(es_cur)
                if not existing:
                    break
                e_rm = random.choice(existing)
                u, v = e_rm
                adj_cur[u].discard(v)
                adj_cur[v].discard(u)
                es_cur.discard(e_rm)
                
                for _try in range(100):
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a == b or a > b:
                        if a > b:
                            a, b = b, a
                        else:
                            continue
                    if (a, b) == (u, v) or (a, b) in es_cur:
                        continue
                    adj_cur[a].add(b)
                    adj_cur[b].add(a)
                    es_cur.add((a, b))
                    break
        
        score, adj_new, es_new = _local_search(adj_cur, n, es_cur, max_iters=50000)
        
        if score < best_score:
            best_score = score
            best_adj = adj_new
            best_edge_set = es_new
        
        if best_score <= 23:
            break
    
    # Final verification
    adj_final = _build_adj(n, best_edge_set)
    final_score = _compute_score(adj_final, n)
    if final_score != best_score:
        best_score = final_score
        best_adj = adj_final
        best_edge_set = set(best_edge_set)
    
    edges = [[u, v] for u, v in sorted(best_edge_set)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
