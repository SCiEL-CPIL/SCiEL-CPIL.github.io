import random
from collections import defaultdict
import numpy as np

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        if P:
            u_candidates = list(P) + list(X)
            if not u_candidates:
                return
            max_deg = -1
            pivot = None
            for u in u_candidates:
                deg = len(adj[u].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P):
                if v in adj[pivot]:
                    continue
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)
        else:
             pass

    def find_maximal_cliques(n, adj):
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        return cliques

    def solve_hitting_set_greedy(n, cliques):
        if not cliques:
            return []
        
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return []
        
        # Map vertex to list of clique indices
        vertex_to_cliques = defaultdict(list)
        for i, c in enumerate(relevant_cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Greedy Strategy:
        # 1. Start with all cliques uncovered.
        # 2. At each step, find the uncovered clique with the smallest number of vertices 
        #    that are NOT yet "excluded" or something? 
        #    Actually, standard Set Cover Greedy: Pick element covering most sets.
        #    But for Hitting Set, "Most Constrained Clique" is often better.
        #    Let's try: Pick the uncovered clique C with the minimum size.
        #    Then pick a vertex v in C that covers the maximum number of OTHER uncovered cliques.
        
        uncovered = set(range(len(relevant_cliques)))
        solution = set()
        
        # Precompute clique vertices as lists for speed
        clique_verts = [list(c) for c in relevant_cliques]
        
        while uncovered:
            # Find the uncovered clique with the fewest vertices
            # Note: We don't remove vertices from cliques, we just track which cliques are covered.
            # The "size" of the clique is fixed.
            # However, a vertex might be "used up"? No, vertices can be used multiple times.
            # So the constraint is just: which cliques are still uncovered?
            
            # Heuristic: Pick the uncovered clique with the smallest number of vertices.
            # If tie, pick the one where the vertices cover the most other uncovered cliques?
            
            min_size = float('inf')
            target_clique_idx = -1
            target_clique = None
            
            for idx in uncovered:
                size = len(clique_verts[idx])
                if size < min_size:
                    min_size = size
                    target_clique_idx = idx
                    target_clique = clique_verts[idx]
            
            if target_clique is None:
                break
                
            # Now, pick the best vertex from target_clique
            best_v = -1
            max_cover = -1
            
            for v in target_clique:
                # Count how many uncovered cliques contain v
                cover_count = 0
                for idx in uncovered:
                    if v in relevant_cliques[idx]:
                        cover_count += 1
                
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_v = v
            
            # Add best_v to solution
            solution.add(best_v)
            
            # Update uncovered
            new_uncovered = set()
            for idx in uncovered:
                if best_v not in relevant_cliques[idx]:
                    new_uncovered.add(idx)
            uncovered = new_uncovered
            
        # Post-processing: Remove redundant vertices
        # A vertex v in solution is redundant if for every clique C in relevant_cliques,
        # if v is in C, then there exists u in solution (u != v) such that u is in C.
        
        changed = True
        while changed:
            changed = False
            for v in list(solution):
                redundant = True
                # Check all cliques that contain v
                for idx in vertex_to_cliques[v]:
                    c = relevant_cliques[idx]
                    # Is there another vertex in solution that is in c?
                    found_other = False
                    for u in solution:
                        if u != v and u in c:
                            found_other = True
                            break
                    if not found_other:
                        redundant = False
                        break
                
                if redundant:
                    solution.remove(v)
                    changed = True
                    
        return sorted(solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges, adj = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(n, adj)
        
        transversal = solve_hitting_set_greedy(n, cliques)
        
        results.append(transversal)
        
    return results
