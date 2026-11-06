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

def _compute_score_from_adj_matrix(A, n=16):
    """Compute max triangle degree-sum from adjacency matrix."""
    deg = A.sum(axis=1).astype(int)
    best = 0
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j]:
                common = (A[i] & A[j]).nonzero()[0]
                for k in common:
                    if k > j:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _smooth_surrogate(A, n=16, beta=50.0):
    """
    Smooth surrogate for max triangle degree-sum using log-sum-exp.
    Returns a differentiable proxy that we can minimize via coordinate descent.
    """
    deg = A.sum(axis=1)
    # For each potential triangle (i,j,k) with i<j<k, compute degree sum
    # Use a soft-max over all triangles
    max_val = 0
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] > 0.5:
                common = np.where((A[i] > 0.5) & (A[j] > 0.5))[0]
                for k in common:
                    if k > j:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_val:
                            max_val = s
    return max_val

def _spectral_descent(A, n=16, edge_set=None, max_iters=500, beta=30.0):
    """
    Spectral coordinate descent: minimize a smooth surrogate of max triangle degree-sum.
    Uses edge-weight adjustments guided by spectral information.
    """
    if edge_set is None:
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] > 0.5:
                    edge_set.add((i,j))
    
    best_score = _compute_score_from_adj_matrix(A, n)
    best_A = A.copy()
    best_es = set(edge_set)
    
    # Precompute all edges and non-edges
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    non_edges = [(i,j) for i,j in all_edges if (i,j) not in edge_set]
    
    for it in range(max_iters):
        if time.time() > _DEADLINE_GUARD:
            break
        
        # Compute degree vector
        deg = A.sum(axis=1)
        
        # Find current worst triangle
        worst_s = 0
        worst_tri = None
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] > 0.5:
                    common = np.where((A[i] > 0.5) & (A[j] > 0.5))[0]
                    for k in common:
                        if k > j:
                            s = deg[i] + deg[j] + deg[k]
                            if s > worst_s:
                                worst_s = s
                                worst_tri = (i,j,k)
        
        if worst_tri is None:
            break
        
        i,j,k = worst_tri
        # Try to reduce this triangle's degree sum by removing one of its edges
        # and adding an edge that doesn't create worse triangles
        for (u,v) in [(i,j),(j,k),(i,k)]:
            if u > v:
                u,v = v,u
            if (u,v) not in edge_set:
                continue
            
            # Try each non-edge
            for (a,b) in non_edges:
                # Apply flip
                A[u,v] = 0
                A[v,u] = 0
                A[a,b] = 1
                A[b,a] = 1
                edge_set.discard((u,v))
                edge_set.add((a,b))
                
                new_score = _compute_score_from_adj_matrix(A, n)
                
                if new_score < best_score:
                    best_score = new_score
                    best_A = A.copy()
                    best_es = set(edge_set)
                else:
                    # Revert
                    A[u,v] = 1
                    A[v,u] = 1
                    A[a,b] = 0
                    A[b,a] = 0
                    edge_set.add((u,v))
                    edge_set.discard((a,b))
        
        if best_score <= 23:
            break
    
    return best_score, best_A, best_es

def _systematic_swap_search(A, n=16, edge_set=None, max_rounds=20):
    """
    Systematic full-swap local search: try ALL possible edge swaps.
    This escapes shallow local minima that random swaps get stuck in.
    """
    if edge_set is None:
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] > 0.5:
                    edge_set.add((i,j))
    
    best_score = _compute_score_from_adj_matrix(A, n)
    best_A = A.copy()
    best_es = set(edge_set)
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    non_edges = [(i,j) for i,j in all_edges if (i,j) not in edge_set]
    
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE_GUARD:
            break
        
        improved = False
        deg = A.sum(axis=1)
        
        # Try all possible swaps
        for (u,v) in list(edge_set):
            if time.time() > _DEADLINE_GUARD:
                break
            for (a,b) in non_edges:
                # Apply flip
                A[u,v] = 0
                A[v,u] = 0
                A[a,b] = 1
                A[b,a] = 1
                edge_set.discard((u,v))
                edge_set.add((a,b))
                
                new_score = _compute_score_from_adj_matrix(A, n)
                
                if new_score < best_score:
                    best_score = new_score
                    best_A = A.copy()
                    best_es = set(edge_set)
                    improved = True
                    break
                else:
                    # Revert
                    A[u,v] = 1
                    A[v,u] = 1
                    A[a,b] = 0
                    A[b,a] = 0
                    edge_set.add((u,v))
                    edge_set.discard((a,b))
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_score, best_A, best_es

def _spectral_guide_swap(A, n=16, edge_set=None, max_iters=200):
    """
    Use spectral information (eigenvector of adjacency matrix) to guide which
    edges to swap. Edges connecting high-eigenvalue vertices are "important"
    and should be preserved; edges connecting low-eigenvalue vertices are
    candidates for removal.
    """
    if edge_set is None:
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] > 0.5:
                    edge_set.add((i,j))
    
    best_score = _compute_score_from_adj_matrix(A, n)
    best_A = A.copy()
    best_es = set(edge_set)
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    non_edges = [(i,j) for i,j in all_edges if (i,j) not in edge_set]
    
    for it in range(max_iters):
        if time.time() > _DEADLINE_GUARD:
            break
        
        # Compute principal eigenvector via power iteration
        v = np.ones(n) / np.sqrt(n)
        for _ in range(20):
            v_new = A @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                break
            v_new = v_new / norm
            v = v_new
        
        # Score each edge by the product of endpoint eigenvalues
        # Edges with high score are "structurally important"
        edge_scores = {}
        for (i,j) in edge_set:
            edge_scores[(i,j)] = abs(v[i] * v[j])
        
        # Sort edges: try removing low-score edges first
        sorted_edges = sorted(edge_set, key=lambda e: edge_scores[e])
        
        improved = False
        for (u,v) in sorted_edges[:20]:  # Try bottom 20 edges
            if time.time() > _DEADLINE_GUARD:
                break
            
            for (a,b) in non_edges:
                # Apply flip
                A[u,v] = 0
                A[v,u] = 0
                A[a,b] = 1
                A[b,a] = 1
                edge_set.discard((u,v))
                edge_set.add((a,b))
                
                new_score = _compute_score_from_adj_matrix(A, n)
                
                if new_score < best_score:
                    best_score = new_score
                    best_A = A.copy()
                    best_es = set(edge_set)
                    improved = True
                    break
                else:
                    # Revert
                    A[u,v] = 1
                    A[v,u] = 1
                    A[a,b] = 0
                    A[b,a] = 0
                    edge_set.add((u,v))
                    edge_set.discard((a,b))
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_score, best_A, best_es

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
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v) in edge_set:
        A[u, v] = 1.0
        A[v, u] = 1.0
    
    best_score = _compute_score_from_adj_matrix(A, n)
    best_A = A.copy()
    best_es = set(edge_set)
    
    # Phase 1: Spectral-guided swap search
    for trial in range(5):
        if time.time() > _DEADLINE_GUARD:
            break
        
        if trial == 0:
            A_cur = A.copy()
            es_cur = set(edge_set)
        else:
            A_cur = best_A.copy()
            es_cur = set(best_es)
            # Perturb: flip k random edges
            k = random.randint(3, 10)
            for _ in range(k):
                existing = list(es_cur)
                if not existing:
                    break
                e_rm = random.choice(existing)
                u, v = e_rm
                A_cur[u, v] = 0
                A_cur[v, u] = 0
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
                    A_cur[a, b] = 1
                    A_cur[b, a] = 1
                    es_cur.add((a, b))
                    break
        
        # Spectral-guided search
        score, A_new, es_new = _spectral_guide_swap(A_cur, n, es_cur, max_iters=100)
        
        if score < best_score:
            best_score = score
            best_A = A_new
            best_es = es_new
        
        if best_score <= 23:
            break
    
    # Phase 2: Systematic full-swap search (the key improvement from knowledge)
    if time.time() < _DEADLINE_GUARD:
        score, A_new, es_new = _systematic_swap_search(best_A, n, best_es, max_rounds=10)
        
        if score < best_score:
            best_score = score
            best_A = A_new
            best_es = es_new
    
    # Phase 3: Another round of spectral-guided search from the improved state
    if time.time() < _DEADLINE_GUARD and best_score > 23:
        score, A_new, es_new = _spectral_guide_swap(best_A, n, best_es, max_iters=150)
        
        if score < best_score:
            best_score = score
            best_A = A_new
            best_es = es_new
    
    # Phase 4: Final systematic polish
    if time.time() < _DEADLINE_GUARD and best_score > 23:
        score, A_new, es_new = _systematic_swap_search(best_A, n, best_es, max_rounds=15)
        
        if score < best_score:
            best_score = score
            best_A = A_new
            best_es = es_new
    
    # Final verification
    adj_final = _build_adj(n, best_es)
    final_score = _compute_score(adj_final, n)
    if final_score != best_score:
        best_score = final_score
        best_es = set(best_es)
    
    edges = [[u, v] for u, v in sorted(best_es)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
