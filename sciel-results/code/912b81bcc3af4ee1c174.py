"""Rewritten program: Structural clique cover for split graphs."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

_DEADLINE = time.time() + 120.0
_MARGIN = 10.0


def _time_ok():
    return (time.time() - _DEADLINE) < -_MARGIN


def _build_adj(n, k, p, seed):
    """Reconstruct the split graph adjacency."""
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    # Clique part: 0 to k-1
    for u in range(k):
        for v in range(u + 1, k):
            adj[u].add(v)
            adj[v].add(u)
    # Cross edges: k to n-1 vs 0 to k-1
    for v in range(k, n):
        for u in range(k):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _solve_split(n, k, p, seed):
    """
    Solves the clique partition for a split graph.
    1. Group independent vertices by their neighborhood in the clique part.
    2. Form cliques from these groups.
    3. Cover remaining edges in the clique part using DSATUR coloring
       on the complement of the residual graph.
    """
    adj = _build_adj(n, k, p, seed)
    
    # 1. Analyze independent set vertices
    groups = {}
    for v in range(k, n):
        nbrs = frozenset(adj[v])
        if nbrs not in groups:
            groups[nbrs] = []
        groups[nbrs].append(v)
        
    cliques = []
    covered_edges_in_C = set()
    
    # 2. Form cliques for each group
    for nbrs, verts in groups.items():
        if not verts:
            continue
        clique = list(nbrs) + verts
        if len(clique) >= 2:
            cliques.append(clique)
            
        nbrs_list = list(nbrs)
        for i in range(len(nbrs_list)):
            u = nbrs_list[i]
            for j in range(i + 1, len(nbrs_list)):
                v = nbrs_list[j]
                covered_edges_in_C.add((u, v) if u < v else (v, u))
                
    # 3. Find remaining edges in C
    remaining_edges = []
    for u in range(k):
        for v in range(u + 1, k):
            if (u, v) not in covered_edges_in_C:
                remaining_edges.append((u, v))
                
    # 4. Cover remaining edges using DSATUR on complement of residual graph
    if remaining_edges:
        H_adj = [set() for _ in range(k)]
        for u, v in remaining_edges:
            H_adj[u].add(v)
            H_adj[v].add(u)
        
        # DSATUR coloring of complement of H
        # Complement has edge (u,v) if u != v and (u,v) not in H
        # Same color class = independent in complement = clique in H
        uncolored = set(range(k))
        colors = [-1] * k
        
        # Precompute degrees in H
        h_deg = [len(H_adj[i]) for i in range(k)]
        
        while uncolored:
            if not _time_ok():
                # Fallback
                for u, v in remaining_edges:
                    cliques.append([u, v])
                break
                
            # DSATUR: pick vertex with max saturation, tie-break by H-degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat_colors = set()
                for u in H_adj[v]:
                    if colors[u] != -1:
                        sat_colors.add(colors[u])
                sat = len(sat_colors)
                if sat > best_sat or (sat == best_sat and h_deg[v] > best_deg):
                    best_sat = sat
                    best_deg = h_deg[v]
                    best_v = v
            
            if best_v == -1:
                break
                
            used_colors = set()
            for u in H_adj[best_v]:
                if colors[u] != -1:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[best_v] = c
            uncolored.remove(best_v)
        
        # Extract cliques from color classes
        color_classes = {}
        for v in range(k):
            if colors[v] != -1:
                if colors[v] not in color_classes:
                    color_classes[colors[v]] = []
                color_classes[colors[v]].append(v)
        
        for c, verts in color_classes.items():
            if len(verts) >= 2:
                cliques.append(verts)
    
    return cliques


def _is_clique_set(cliques, adj, n):
    # every clique must be a true clique
    for c in cliques:
        if len(c) < 2:
            return False
        for i in range(len(c)):
            ci = c[i]
            for j in range(i + 1, len(c)):
                cj = c[j]
                if cj not in adj[ci]:
                    return False
    # every edge must be covered exactly once
    seen = set()
    for c in cliques:
        L = len(c)
        for i in range(L):
            ci = c[i]
            for j in range(i + 1, L):
                cj = c[j]
                key = (ci, cj) if ci < cj else (cj, ci)
                if key in seen:
                    return False
                seen.add(key)
    # check all edges in graph are covered
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if (u, v) not in seen:
                    return False
    return True


def construct():
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, p, seed = spec
        if not _time_ok():
            # Fallback to trivial cover if time is up
            # This might be slow to generate but is valid
            # We can just return empty? No, must be valid.
            # If time is up, we can't even build adj safely? 
            # _build_adj is fast.
            adj = _build_adj(n, k, p, seed)
            # Trivial cover: one clique per edge
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        cliques.append([u, v])
            results.append(cliques)
            continue
            
        cliques = _solve_split(n, k, p, seed)
        
        # Verify
        adj = _build_adj(n, k, p, seed)
        if not _is_clique_set(cliques, adj, n):
            # If invalid, fallback to trivial cover
            # This should not happen with the new algorithm
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        cliques.append([u, v])
                        
        # Ensure types
        cliques = [[int(v) for v in c] for c in cliques]
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
