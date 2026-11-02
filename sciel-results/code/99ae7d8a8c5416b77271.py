"""Kissing number in dimension 11: MIS on conflict graph with Tomita coloring bound."""
import itertools
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Build candidate vectors: weight 1 (e_i, -e_i) and weight 2 (±e_i ± e_j)
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    n = len(candidates)
    
    # Precompute norms squared and dot products
    norms2 = np.array([sum(x*x for x in v) for v in candidates], dtype=np.int64)
    cand_arr = np.array(candidates, dtype=np.int64)
    dots = cand_arr @ cand_arr.T  # n x n dot product matrix
    
    # Build conflict graph: edge (i,j) if ||c_i - c_j||^2 < max(||c_i||^2, ||c_j||^2)
    # i.e., norms2[i] + norms2[j] - 2*dots[i,j] < max(norms2[i], norms2[j])
    conflict = np.zeros((n, n), dtype=bool)
    max_norms = np.maximum(norms2[:, None], norms2[None, :])
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    conflict = dist2 < max_norms
    np.fill_diagonal(conflict, False)
    
    # Convert to adjacency list (using Python lists for speed in branch-and-bound)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if conflict[i, j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Degree-based ordering for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map from original index to position in order
    pos = [0] * n
    for p, idx in enumerate(order):
        pos[idx] = p
    
    # Rebuild adjacency in ordered indices
    adj_ordered = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            adj_ordered[pos[i]].add(pos[j])
    
    # Tomita coloring for upper bound
    # We process vertices in reverse order (last in order first) for the MIS
    # Actually, standard Tomita: process in some order, color greedily
    # The color number gives an upper bound on the size of MIS in the remaining subgraph
    
    def color_bound(P):
        """Assign colors to vertices in P (a list of vertex indices in ordered space).
        Returns the number of colors used, which is an upper bound on |MIS| in P."""
        if not P:
            return 0
        # Greedy coloring: process vertices, assign smallest available color
        # To get tight bound, process in reverse degree order within P
        colors = {}
        color_classes = []
        for v in P:
            # Find smallest color class that doesn't contain a neighbor of v
            assigned = False
            for ci, cc in enumerate(color_classes):
                if not any(u in adj_ordered[v] for u in cc):
                    cc.add(v)
                    colors[v] = ci
                    assigned = True
                    break
            if not assigned:
                color_classes.append({v})
                colors[v] = len(color_classes) - 1
        return len(color_classes)
    
    # Branch and bound for MIS
    best_set = []
    best_size = 0
    
    def expand(P, C):
        """P: list of candidate vertices (in ordered index space)
           C: current selected set (list of ordered indices)
           Returns the best independent set found in this branch."""
        nonlocal best_set, best_size
        
        if not P:
            if len(C) > best_size:
                best_size = len(C)
                best_set = list(C)
            return
        
        # Upper bound: |C| + color_bound(P)
        ub = len(C) + color_bound(P)
        if ub <= best_size:
            return
        
        # Choose a vertex to branch on: pick the one with highest degree in P
        # Or just pick the last one in P (which has highest degree due to ordering)
        v = P[-1]  # highest degree remaining
        P_new = P[:-1]
        
        # Branch 1: include v
        C.append(v)
        # Remove v and its neighbors from P_new
        P_included = [u for u in P_new if u not in adj_ordered[v]]
        expand(P_included, C)
        C.pop()
        
        # Branch 2: exclude v
        expand(P_new, C)
    
    # Start with all vertices
    all_vertices = list(range(n))
    expand(all_vertices, [])
    
    # Convert back to original indices
    selected_original = [order[i] for i in best_set]
    
    result = np.array([candidates[i] for i in selected_original], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
