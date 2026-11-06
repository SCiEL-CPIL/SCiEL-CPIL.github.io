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

def _local_search_sa(adj, n, edge_set, max_iters=200000):
    """Local search with Simulated Annealing: flip edges to maintain exactly 65 edges, minimize max triangle degree-sum."""
    best_score = _compute_score(adj, n)
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edge_set)
    
    # Use a local random instance to avoid global state issues
    rng = random.Random(42)
    
    start_time = time.time()
    T = 10.0 # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_score = best_score
    current_adj = [set(s) for s in adj]
    current_edge_set = set(edge_set)
    
    for it in range(max_iters):
        if it % 1000 == 0 and time.time() > _DEADLINE_GUARD:
            break
        
        # Identify worst triangle to bias removal
        existing = list(current_edge_set)
        if not existing:
            break
        
        # With 50% chance, try to pick an edge involved in a high-sum triangle
        if rng.random() < 0.5:
            # Sample a few edges to find a bad one
            candidates = rng.sample(existing, min(10, len(existing)))
            max_s = -1
            best_cand = None
            for u, v in candidates:
                # Check triangles containing (u,v)
                common = current_adj[u] & current_adj[v]
                for w in common:
                    s = len(current_adj[u]) + len(current_adj[v]) + len(current_adj[w])
                    if s > max_s:
                        max_s = s
                        best_cand = (u, v)
            if best_cand:
                e_remove = best_cand
            else:
                e_remove = rng.choice(existing)
        else:
            e_remove = rng.choice(existing)
            
        u, v = e_remove
        if u > v:
            u, v = v, u
        
        # Pick a random non-existing edge to add
        e_add = None
        for _try in range(50):
            a = rng.randint(0, n-1)
            b = rng.randint(0, n-1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) == (u, v):
                continue
            if (a, b) in current_edge_set:
                continue
            e_add = (a, b)
            break
        if not e_add:
            continue
            
        a, b = e_add
        
        # Apply the flip
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        current_edge_set.remove((u, v))
        
        current_adj[a].add(b)
        current_adj[b].add(a)
        current_edge_set.add((a, b))
        
        # Recompute score
        new_score = _compute_score(current_adj, n)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [set(s) for s in current_adj]
                best_edge_set = set(current_edge_set)
        else:
            # Revert
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edge_set.add((u, v))
            
            current_adj[a].remove(b)
            current_adj[b].remove(a)
            current_edge_set.remove((a, b))
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
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
        
        score, adj_new, es_new = _local_search_sa(adj_cur, n, es_cur, max_iters=50000)
        
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
