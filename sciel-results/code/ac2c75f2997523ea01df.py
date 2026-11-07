"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = []
        # Edges within the clique C (first k vertices)
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        
        # Edges between C and I (independent set)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for C and cross edges
        # C vertices: 0 to k-1
        # I vertices: k to n-1
        
        # N_C[v] is set of neighbors of v in C
        N_C = {v: set() for v in range(k, n)}
        
        cross_edges = []
        for u, v in edges:
            if u < k and v >= k:
                N_C[v].add(u)
                cross_edges.append((u, v))
            elif u >= k and v < k:
                N_C[u].add(v)
                cross_edges.append((v, u))
        
        # Internal edges of C
        internal_edges = set()
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((min(u, v), max(u, v)))
        
        # Strategy:
        # 1. Start with S_v = N_C[v] for each v in I.
        # 2. Resolve conflicts for internal edges.
        # 3. Cover remaining internal edges.
        
        # Map each internal edge to the list of v's that cover it
        # An edge (u, w) is covered by v if u in N_C[v] and w in N_C[v]
        
        edge_to_vs = {e: [] for e in internal_edges}
        
        for v in range(k, n):
            neighbors = N_C[v]
            # All pairs in neighbors form internal edges covered by v
            for u, w in combinations(neighbors, 2):
                e = (min(u, w), max(u, w))
                if e in edge_to_vs:
                    edge_to_vs[e].append(v)
        
        # We need to ensure each edge is covered by at most 1 v.
        # If covered by > 1, we split the S_v for the "extra" v's.
        # Splitting means: S_v is partitioned into subsets such that no subset contains both u and w.
        # Simplest split: Remove w from S_v? No, we must cover cross edges.
        # We must partition N_C[v] into sets.
        # Let's represent S_v as a list of sets (parts).
        # Initially, parts[v] = [N_C[v]]
        
        parts = {v: [set(N_C[v])] for v in range(k, n)}
        
        # Resolve conflicts
        # We iterate until no conflicts? Or just once?
        # If we split a v, it might create new conflicts? No, splitting reduces coverage.
        # It might uncover an edge that was covered by another v, making it covered by 1 (good) or 0 (needs C-clique).
        
        # To be efficient, we process edges with conflicts.
        # A conflict is an edge with len(edge_to_vs[e]) > 1.
        
        # We need to update edge_to_vs when we split.
        # This is tricky. Let's just recalculate coverage after modifications?
        # Or, we can greedily fix.
        
        # Let's try a simpler approach:
        # If an edge is covered by multiple v's, pick one to keep, and for others, 
        # we ensure they don't cover it.
        # To ensure v doesn't cover (u,w), u and w must be in different parts of v.
        
        # We will maintain for each v, a list of parts.
        # And for each part, we know which internal edges it covers.
        
        # Actually, let's just use the "Seed" strategy if conflicts are too hard, 
        # but try the optimization first.
        
        # Optimization:
        # For each edge e in internal_edges:
        #   vs = [v for v in range(k, n) if e[0] in N_C[v] and e[1] in N_C[v]]
        #   if len(vs) > 1:
        #       keep_v = vs[0]
        #       for v in vs[1:]:
        #           # Split parts of v to separate e[0] and e[1]
        #           # Find the part in parts[v] that contains both
        #           for i, part in enumerate(parts[v]):
        #               if e[0] in part and e[1] in part:
        #                   # Split part into A (containing e[0]) and B (containing e[1])
        #                   A = set()
        #                   B = set()
        #                   for node in part:
        #                       if node == e[0]:
        #                           A.add(node)
        #                       elif node == e[1]:
        #                           B.add(node)
        #                       else:
        #                           # Assign to A or B arbitrarily? 
        #                           # To minimize future conflicts, maybe keep with majority?
        #                           # For now, put in A
        #                           A.add(node)
        #                   parts[v][i] = A
        #                   parts[v].append(B)
        #                   break
        
        # This greedy might not be optimal but should reduce conflicts.
        
        # After this, some edges might still be covered by >1?
        # If we split, we ensure that specific edge is not covered by that v.
        # So after processing all edges, no edge should be covered by >1 v.
        
        # Now, determine which internal edges are covered by at least one v.
        covered_edges = set()
        for v in range(k, n):
            for part in parts[v]:
                for u, w in combinations(part, 2):
                    e = (min(u, w), max(u, w))
                    covered_edges.add(e)
        
        # Remaining internal edges
        remaining_edges = internal_edges - covered_edges
        
        # Construct cliques
        cliques = []
        
        # 1. Cliques for I
        for v in range(k, n):
            for part in parts[v]:
                if len(part) >= 1: # Need at least 1 neighbor to form a clique of size >= 2 with v
                    # Clique is {v} U part
                    # But if part is empty, we can't form a clique with v? 
                    # The rule says "at least two vertices".
                    # If N_C[v] is empty, v is isolated from C. No cross edges.
                    # Then no clique needed for v?
                    # But if there are no cross edges, we don't need to cover anything for v.
                    if len(part) > 0:
                        cliques.append([v] + list(part))
                    else:
                        # If part is empty, it means we split off an empty set? 
                        # Or N_C[v] was empty.
                        # If N_C[v] is empty, we shouldn't have created a part.
                        pass
        
        # 2. Cliques for C covering remaining_edges
        # We need to partition remaining_edges into cliques.
        # Since remaining_edges is a subgraph of K_k, we can use a greedy clique cover.
        # Greedy: Find largest clique in remaining graph, remove its edges, repeat.
        
        rem_graph = {u: set() for u in range(k)}
        for u, w in remaining_edges:
            rem_graph[u].add(w)
            rem_graph[w].add(u)
            
        while remaining_edges:
            # Find a maximal clique in rem_graph
            # Simple greedy: pick a vertex with max degree, add to clique if connected to all in clique
            # Or just pick 2-cliques if complex?
            # Given k is small, let's try to find a large one.
            
            # Pick a random edge or just iterate
            # To ensure progress, pick any edge
            # Better: Find the largest clique using a simple heuristic
            # Start with a vertex, expand.
            
            best_clique = None
            best_size = 0
            
            # Try starting from each vertex in remaining edges
            vertices_in_rem = set()
            for u, w in remaining_edges:
                vertices_in_rem.add(u)
                vertices_in_rem.add(w)
                
            for start in vertices_in_rem:
                clique = {start}
                # Add neighbors of start that are connected to all in clique
                candidates = list(rem_graph[start])
                for c in candidates:
                    if all(c in rem_graph[x] for x in clique):
                        clique.add(c)
                # Refine? This is a maximal clique for start.
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if not best_clique or len(best_clique) < 2:
                # Should not happen if remaining_edges not empty
                break
                
            # Add this clique to result
            cliques.append(list(best_clique))
            
            # Remove edges covered by this clique from remaining_edges and rem_graph
            for u, w in combinations(best_clique, 2):
                e = (min(u, w), max(u, w))
                if e in remaining_edges:
                    remaining_edges.remove(e)
                    rem_graph[u].remove(w)
                    rem_graph[w].remove(u)
            
        # Convert to lists of ints
        final_cliques = [list(map(int, c)) for c in cliques]
        
        # Validation check (optional in prod, but good for dev)
        # Check every edge is in exactly one clique
        # Check every clique is valid
        
        return final_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
