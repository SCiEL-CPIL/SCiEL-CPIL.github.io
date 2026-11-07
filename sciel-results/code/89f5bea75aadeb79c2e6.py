import time
import sys
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def instance(n, k, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        # K edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # I edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        K_vertices = list(range(k))
        I_vertices = list(range(k, n))
        
        # Generate candidate K-cliques (all subsets of size >= 2)
        # For k up to 30, we can't enumerate all. Use a smarter approach.
        # Key insight: In a split graph, the optimal clique partition has a specific structure.
        # Each clique is either:
        #   (a) a subset of K only, or
        #   (b) a subset of K plus at most one vertex from I
        # (because I is independent, no clique can contain two I vertices)
        
        # Strategy: Use MIP to decide:
        # 1. Which K-cliques to use (partition K into cliques)
        # 2. For each I vertex v, which K-clique(s) it attaches to
        #    (v can attach to multiple K-cliques, but the union of those K-cliques
        #     must be a clique, meaning they must be pairwise disjoint)
        
        # Simplified approach: 
        # - First, find a good partition of K into cliques
        # - Then use MIP to optimally assign I vertices to these K-cliques
        # - The key improvement: use a better K-partition via MIP or smarter greedy
        
        # Better K-partition: Use the fact that K is a complete graph.
        # Wait, K is a clique! All vertices in K are connected to each other.
        # So K itself is one big clique. But we might want to split it.
        
        # Actually, since K is a complete graph, any subset of K is a clique.
        # The question is: should we keep K as one big clique, or split it?
        # Splitting K allows I vertices to attach to smaller pieces.
        
        # Let's think about this differently. Use a MIP where:
        # - y_S = 1 if we use clique S (subset of K, |S| >= 2) as a base
        # - x_{v,S} = 1 if I-vertex v is added to clique S
        # 
        # Constraints:
        # - Every edge in K is covered exactly once: for each edge (u,w) in K,
        #   sum over all S containing both u and w of y_S = 1
        # - Every cross edge (u,v) is covered exactly once: for each cross edge (u,v),
        #   sum over all S containing u of (y_S + x_{v,S}) = 1
        #   But x_{v,S} can only be 1 if y_S = 1 (v attaches to an existing clique)
        #   Also, if x_{v,S1} = 1 and x_{v,S2} = 1, then S1 and S2 must be disjoint
        #     (so that S1 ∪ S2 ∪ {v} is a clique)
        # - For each S, at most one I vertex can be added: sum_v x_{v,S} <= 1
        # - x_{v,S} <= y_S
        # - Objective: minimize sum_S y_S (number of cliques)
        
        # This is a large MIP. For k=30, the number of candidate S's is huge.
        # We need to restrict candidates.
        
        # Key observation: In an optimal solution, each K-clique S used should be
        # "maximal" in some sense. Let's use a column generation approach or
        # restrict to cliques that are "natural" partitions.
        
        # Alternative: Use a two-phase approach
        # Phase 1: Find a good K-partition (small number of cliques)
        # Phase 2: MIP to assign I vertices
        
        # For Phase 1, since K is complete, we can partition it into cliques of
        # various sizes. The number of K-cliques in the partition affects how
        # I vertices can be assigned.
        
        # Let's try: partition K into cliques of size ~k/t for t = 1, 2, 3, ...
        # and see which gives the best total.
        
        # Actually, let's just use a single MIP with a restricted set of candidate K-cliques.
        # Candidate K-cliques: all subsets of K of size 2, 3, ..., k
        # But that's 2^k - k - 1, which is too many for k=30.
        
        # Restriction: Only consider "contiguous" or "structured" cliques?
        # Or use a greedy approach to generate good candidates.
        
        # Let's try a different angle: 
        # Since K is a clique, the edges within K form a complete graph K_k.
        # We need to cover all edges of K_k with cliques (subsets of K).
        # This is equivalent to edge-coloring K_k where each color class is a clique.
        # But in a complete graph, any two edges sharing a vertex can be in the same clique
        # only if all three vertices form a triangle (which they do in K_k).
        
        # The minimum number of cliques to cover all edges of K_k is k (use k-1 edges
        # from each vertex? No, that's not right).
        
        # Actually, for K_k, we can cover all edges with k cliques: 
        # Clique i = {i, i+1, ..., k-1} for i = 0, ..., k-1? No, that doesn't work.
        
        # Let's think: K_3 has 3 edges. We need 3 cliques of size 2 (each edge).
        # Or 1 clique of size 3. So min is 1.
        # K_4 has 6 edges. One clique of size 4 covers all. So min is 1.
        # In general, K_k can be covered by 1 clique (the whole K).
        
        # So why would we ever split K? Because splitting allows I vertices to attach
        # to smaller pieces, potentially reducing the total number of cliques.
        
        # Example: If an I vertex v is connected to all of K, we can add v to the big
        # K-clique, giving K ∪ {v}. But if v is only connected to half of K, we might
        # want to split K into two halves and add v to one half.
        
        # Let's use a MIP with a restricted candidate set.
        # Generate candidates: 
        # - The whole K
        # - All pairs (size 2)
        # - Some random subsets
        # - Structured partitions (e.g., split into t equal parts)
        
        # For simplicity, let's start with: whole K, all pairs, and a few structured splits.
        
        # Actually, let's use a completely different approach:
        # Greedy with local search.
        
        # Start with K as one clique.
        # For each I vertex v, if v is connected to all of K, add v to K.
        # Otherwise, create a new clique {u, v} for each neighbor u of v.
        # Then try to improve by merging or splitting.
        
        # This is getting complicated. Let me just improve the current approach:
        # 1. Better K-partition: use MIP to find a good partition of K
        # 2. Better I-assignment: use MIP
        
        # For K-partition, since K is complete, the optimal partition depends on
        # the I vertices. Let's just try a few K-partitions and pick the best.
        
        # Try K-partitions with t cliques for t = 1, 2, 3, ..., min(k, 10)
        best_result = None
        best_count = float('inf')
        
        for t in range(1, min(k, 8) + 1):
            if time.time() > deadline - 20:
                break
            
            # Partition K into t cliques as evenly as possible
            base_size = k // t
            remainder = k % t
            K_cliques = []
            idx = 0
            for i in range(t):
                size = base_size + (1 if i < remainder else 0)
                if size >= 2:
                    K_cliques.append(tuple(K_vertices[idx:idx+size]))
                    idx += size
                elif size == 1:
                    # Can't have a clique of size 1. Merge with previous.
                    if K_cliques:
                        K_cliques[-1] = K_cliques[-1] + (K_vertices[idx],)
                    else:
                        # First clique, need at least 2
                        K_cliques.append((K_vertices[idx], K_vertices[idx+1]))
                        idx += 2
                    idx += 1
            
            # Verify we used all vertices
            if idx != k:
                continue
            
            # Remove any cliques of size 1 (shouldn't happen)
            K_cliques = [c for c in K_cliques if len(c) >= 2]
            if not K_cliques:
                continue
            
            # Now solve MIP for I-vertex assignment
            m = len(K_cliques)
            n_I = len(I_vertices)
            
            # comp[v_idx][c_idx] = True if K_clique[c_idx] is subset of N_K(v)
            comp = [[False] * m for _ in range(n_I)]
            for vi, v in enumerate(I_vertices):
                for ci, c in enumerate(K_cliques):
                    ok = True
                    for u in c:
                        if not adj[v][u]:
                            ok = False
                            break
                    comp[vi][ci] = ok
            
            # disjoint[ci][cj] = True if K_cliques[ci] and K_cliques[cj] are disjoint
            disjoint = [[True] * m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    if i == j:
                        disjoint[i][j] = False
                    else:
                        if set(K_cliques[i]).intersection(set(K_cliques[j])):
                            disjoint[i][j] = False
            
            # MIP variables: x[vi, ci]
            n_vars = n_I * m
            if n_vars == 0:
                cliques = [list(c) for c in K_cliques]
                count = len(cliques)
                if count < best_count:
                    best_count = count
                    best_result = cliques
                continue
            
            c_obj = np.zeros(n_vars)
            for vi, v in enumerate(I_vertices):
                for ci in range(m):
                    if comp[vi][ci]:
                        c_obj[vi * m + ci] = -len(K_cliques[ci])
            
            A_rows = []
            b_ub_list = []
            
            # Constraint: for each ci, sum_v x[vi, ci] <= 1
            for ci in range(m):
                row = np.zeros(n_vars)
                for vi in range(n_I):
                    row[vi * m + ci] = 1
                A_rows.append(row)
                b_ub_list.append(1)
            
            # Constraint: for each vi, if ci and cj are not disjoint, x[vi,ci] + x[vi,cj] <= 1
            for vi in range(n_I):
                for ci in range(m):
                    for cj in range(ci + 1, m):
                        if not disjoint[ci][cj] and comp[vi][ci] and comp[vi][cj]:
                            row = np.zeros(n_vars)
                            row[vi * m + ci] = 1
                            row[vi * m + cj] = 1
                            A_rows.append(row)
                            b_ub_list.append(1)
            
            if A_rows:
                A_ub = np.array(A_rows)
                b_ub = np.array(b_ub_list)
                constraints = LinearConstraint(A_ub, -np.inf, b_ub)
            else:
                constraints = LinearConstraint(np.zeros((0, n_vars)), -np.inf, np.zeros(0))
            
            bounds = Bounds(0, 1)
            
            try:
                res = milp(c_obj, constraints=constraints, integrality=np.ones(n_vars), bounds=bounds)
            except Exception:
                res = None
            
            # Construct solution
            cliques = []
            for ci, c_k in enumerate(K_cliques):
                assigned_v = None
                if res and res.status == 0:
                    for vi, v in enumerate(I_vertices):
                        if res.x[vi * m + ci] > 0.5:
                            assigned_v = v
                            break
                if assigned_v is not None:
                    cliques.append(list(c_k) + [assigned_v])
                else:
                    cliques.append(list(c_k))
            
            # Add remaining I edges as 2-cliques
            for vi, v in enumerate(I_vertices):
                covered_neighbors = set()
                if res and res.status == 0:
                    for ci in range(m):
                        if res.x[vi * m + ci] > 0.5:
                            covered_neighbors.update(K_cliques[ci])
                
                for u in K_vertices:
                    if adj[v][u] and u not in covered_neighbors:
                        cliques.append([u, v])
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_result = cliques
        
        if best_result is None:
            # Fallback: use all edges as 2-cliques
            best_result = [list(e) for e in edges]
        
        return best_result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        try:
            cliques = solve_split_graph(*spec)
            results.append(cliques)
        except Exception as e:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            
    return results
