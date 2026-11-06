"""Model-and-optimize approach for the 16-vertex, 65-edge triangle degree-sum problem."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_DEADLINE_GUARD = 130.0  # safety margin

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_score(adj, n=16):
    """Compute max triangle degree-sum."""
    deg = [len(adj[i]) for i in range(n)]
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

def _fast_local_search(edge_set, n=16, max_iters=10000, deadline=_DEADLINE_GUARD):
    """
    Fast local search using incremental score updates.
    Tries all (remove, add) swaps.
    """
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    deg = [len(adj[i]) for i in range(n)]
    current_score = _compute_score(adj, n)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute triangles for initial state to speed up? No, dynamic update is better.
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Systematic search: iterate through all possible removals
        current_edges_list = list(edge_set)
        random.shuffle(current_edges_list)
        
        for e_out in current_edges_list:
            if time.time() > deadline:
                return best_score, best_edge_set
            
            u, v = e_out
            # Degrees of u and v decrease by 1
            deg[u] -= 1
            deg[v] -= 1
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove(e_out)
            
            # Now try all possible additions
            # We need to find e_in such that e_in is not in edge_set
            # To be fast, we can iterate over all non-edges? 
            # Or just random sample? 
            # The "inspiration" suggests systematic full-swap. 
            # However, 65 * 480 = 31200 swaps per iteration.
            # If each swap evaluation is O(V) ~ 16, that's ~500k ops.
            # 10000 iters * 500k = 5e9 ops. Too slow in Python.
            
            # We need a smarter heuristic or fewer swaps per iteration.
            # Let's try: for each e_out, try a subset of e_in candidates.
            # Or, better: just do random swaps but with FAST evaluation.
            
            # Let's revert and do random swaps with fast eval.
            edge_set.add(e_out)
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
            # Pick random e_in
            # Generate random non-edge
            while True:
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b: continue
                if a > b: a, b = b, a
                if (a, b) not in edge_set:
                    break
            
            # Apply swap
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove(e_out)
            deg[u] -= 1
            deg[v] -= 1
            
            adj[a].add(b)
            adj[b].add(a)
            edge_set.add((a, b))
            deg[a] += 1
            deg[b] += 1
            
            # Fast score update
            # Only triangles involving u, v, a, b are affected.
            # Actually, degrees changed for u, v, a, b.
            # Triangles involving any of these 4 vertices need re-evaluation.
            # Also, triangles that *lost* an edge or *gained* an edge.
            
            # Let's just recompute score for triangles involving {u,v,a,b}.
            # But we need MAX score.
            # If the previous max triangle was not affected, and new max < prev max, we are good.
            # But if prev max was affected, it might have decreased.
            # And new triangles might have increased.
            
            # Simplest correct fast update:
            # 1. Identify all triangles that contain at least one of {u,v,a,b}.
            # 2. Compute their sums.
            # 3. The new global max is max( max(affected sums), max(unaffected sums) ).
            # We don't store unaffected sums.
            
            # Alternative: Just recompute global max but only for triangles involving {u,v,a,b}
            # AND keep a set of "active" high-scoring triangles?
            
            # Given the small size (N=16), total triangles is small (max ~ 16*15*14/6 = 560).
            # A full recompute is ~ 560 * 3 additions = 1680 ops.
            # Random swap overhead is low.
            # 10000 iters * 1680 ops = 16.8M ops. This is very fast (<0.1s).
            
            new_score = _compute_score(adj, n)
            
            if new_score < best_score:
                best_score = new_score
                best_edge_set = set(edge_set)
                improved = True
            else:
                # Revert
                adj[a].remove(b)
                adj[b].remove(a)
                edge_set.remove((a, b))
                deg[a] -= 1
                deg[b] -= 1
                
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add(e_out)
                deg[u] += 1
                deg[v] += 1
                
        if not improved and it % 100 == 0:
            # Perturb if stuck
            # Flip 5 random edges
            for _ in range(5):
                e_out = random.choice(list(edge_set))
                u, v = e_out
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.remove(e_out)
                
                while True:
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a == b: continue
                    if a > b: a, b = b, a
                    if (a, b) not in edge_set:
                        break
                adj[a].add(b)
                adj[b].add(a)
                edge_set.add((a, b))
                
            # Update score after perturbation
            current_score = _compute_score(adj, n)
            if current_score < best_score:
                best_score = current_score
                best_edge_set = set(edge_set)

    return best_score, best_edge_set

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
    best_edge_set = set(edge_set)
    
    # Run fast local search
    # We can run it for a long time since it's fast
    score, es_new = _fast_local_search(edge_set, n, max_iters=50000)
    
    if score < best_score:
        best_score = score
        best_edge_set = es_new
        
    # Final verification
    adj_final = _build_adj(n, best_edge_set)
    final_score = _compute_score(adj_final, n)
    
    edges = [[u, v] for u, v in sorted(best_edge_set)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
