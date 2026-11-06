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

def _compute_score_mask(adj_masks, n=16):
    """Compute max triangle degree-sum using numpy bitmask."""
    deg = np.array([len(adj_masks[i]) for i in range(n)], dtype=np.int32)
    best = 0
    for i in range(n):
        for j in range(i+1, n):
            common = adj_masks[i] & adj_masks[j]
            if common:
                # Find all k in common
                ks = np.nonzero(common)[0]
                for k in ks:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best

def _local_search(adj, n, edge_set, max_iters=100000):
    """Local search: flip edges to maintain exactly 65 edges, minimize max triangle degree-sum."""
    deg = [len(adj[i]) for i in range(n)]
    best_score = _compute_score(adj, n)
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edge_set)
    
    random.seed(42)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = set(all_edges) - set(edge_set)
    
    for it in range(max_iters):
        if it % 200 == 0 and time.time() > _DEADLINE_GUARD:
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
        non_edge_list = list(non_edges)
        if not non_edge_list:
            break
        e_add = random.choice(non_edge_list)
        a, b = e_add
        
        # Apply the flip
        adj[u].remove(v)
        adj[v].remove(u)
        edge_set.remove((u, v))
        non_edges.add((u, v))
        
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))
        non_edges.remove((a, b))
        
        # Recompute score
        new_score = _compute_score(adj, n)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = [set(s) for s in adj]
            best_edge_set = set(edge_set)
        else:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            non_edges.discard((u, v))
            
            adj[a].remove(b)
            adj[b].remove(a)
            edge_set.remove((a, b))
            non_edges.add((a, b))
    
    return best_score, best_adj, best_edge_set

def _targeted_repair(adj, n, edge_set, max_rounds=20):
    """Targeted repair: find worst triangles and try to reduce their degree sum."""
    deg = [len(adj[i]) for i in range(n)]
    best_score = _compute_score(adj, n)
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edge_set)
    
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE_GUARD:
            break
        
        # Find all triangles and their degree sums
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        s = deg[i] + deg[j] + deg[k]
                        triangles.append((s, i, j, k))
        
        if not triangles:
            break
        
        # Sort by degree sum descending
        triangles.sort(reverse=True)
        
        # Try to fix the worst triangles
        improved = False
        for s, i, j, k in triangles[:5]:  # Try top 5 worst triangles
            if time.time() > _DEADLINE_GUARD:
                break
            
            # For each edge in the triangle, try to remove it and add a better edge
            for (u, v) in [(i, j), (j, k), (i, k)]:
                if u > v:
                    u, v = v, u
                
                # Try removing this edge and adding a non-edge that doesn't create worse triangles
                removed_edge = (u, v)
                if removed_edge not in edge_set:
                    continue
                
                # Find candidate non-edges to add
                non_edges = set((a, b) for a in range(n) for b in range(a+1, n) if (a, b) not in edge_set)
                
                # Try each non-edge
                for (a, b) in list(non_edges)[:50]:  # Limit for speed
                    # Apply flip
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edge_set.discard(removed_edge)
                    
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((a, b))
                    
                    new_score = _compute_score(adj, n)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [set(s) for s in adj]
                        best_edge_set = set(edge_set)
                        improved = True
                        break
                    else:
                        # Revert
                        adj[u].add(v)
                        adj[v].add(u)
                        edge_set.add(removed_edge)
                        
                        adj[a].discard(b)
                        adj[b].discard(a)
                        edge_set.discard((a, b))
            
            if improved:
                break
        
        if not improved:
            break
    
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
    for trial in range(15):
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
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    if (a, b) in es_cur:
                        continue
                    adj_cur[a].add(b)
                    adj_cur[b].add(a)
                    es_cur.add((a, b))
                    break
        
        # Local search
        score, adj_new, es_new = _local_search(adj_cur, n, es_cur, max_iters=30000)
        
        if score < best_score:
            best_score = score
            best_adj = adj_new
            best_edge_set = es_new
        
        # Targeted repair
        score2, adj_rep, es_rep = _targeted_repair(adj_new, n, es_new, max_rounds=10)
        
        if score2 < best_score:
            best_score = score2
            best_adj = adj_rep
            best_edge_set = es_rep
        
        if best_score <= 23:
            break
    
    # Final polish: more intensive local search
    if time.time() < _DEADLINE_GUARD:
        score_final, adj_final, es_final = _local_search(best_adj, n, best_edge_set, max_iters=50000)
        if score_final < best_score:
            best_score = score_final
            best_adj = adj_final
            best_edge_set = es_final
        
        # Another targeted repair
        score_rep, adj_rep, es_rep = _targeted_repair(best_adj, n, best_edge_set, max_rounds=20)
        if score_rep < best_score:
            best_score = score_rep
            best_adj = adj_rep
            best_edge_set = es_rep
    
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
