"""Clique partitions of split graphs: structural approach with local refinement."""
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve(n, k, p, seed):
        # Reconstruct graph
        rng = random.Random(seed)
        neighbors_I = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors_I[v].add(u)
        
        # All edges in C (clique)
        all_C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_C_edges.add((u, w))
        
        def greedy_clique_cover(vertices, edge_set, deadline):
            """
            Partition vertices into minimum number of cliques in the graph (V, edge_set).
            Returns list of sets (cliques).
            """
            if not vertices:
                return []
            
            # Order vertices by degree in the induced subgraph (descending)
            deg = {v: 0 for v in vertices}
            for v in vertices:
                for w in vertices:
                    if v != w and (min(v, w), max(v, w)) in edge_set:
                        deg[v] += 1
            
            ordered = sorted(vertices, key=lambda x: (-deg[x], x))
            
            # Greedy clique cover: assign each vertex to the first clique where it fits
            cliques = []  # list of sets
            for v in ordered:
                placed = False
                for clique in cliques:
                    # Check if v can be added to this clique (must be adjacent to all)
                    can_add = True
                    for w in clique:
                        if (min(v, w), max(v, w)) not in edge_set:
                            can_add = False
                            break
                    if can_add:
                        clique.add(v)
                        placed = True
                        break
                if not placed:
                    cliques.append({v})
            
            # Only return cliques with at least 2 vertices
            return [c for c in cliques if len(c) >= 2]

        def solve_instance(ordering_strategy="degree_desc"):
            """
            Solves the clique partition.
            """
            covered_C_edges = set()
            cliques = []
            
            # Determine order of processing I-vertices
            I_vertices = list(range(k, n))
            if ordering_strategy == "degree_desc":
                I_vertices = sorted(I_vertices, key=lambda v: len(neighbors_I[v]), reverse=True)
            elif ordering_strategy == "degree_asc":
                I_vertices = sorted(I_vertices, key=lambda v: len(neighbors_I[v]))
            elif ordering_strategy == "random":
                rng_sort = random.Random(seed + 42)
                rng_sort.shuffle(I_vertices)
            
            for v in I_vertices:
                N_v = neighbors_I[v]
                if not N_v:
                    continue
                
                # The conflict graph on N_v is defined by edges in covered_C_edges.
                # We want to partition N_v into independent sets in this conflict graph.
                # An independent set in the conflict graph is a clique in the remaining C-graph.
                
                # Extract the subgraph of covered_C_edges induced by N_v
                conflict_edges = set()
                for u in N_v:
                    for w in N_v:
                        if u < w:
                            if (u, w) in covered_C_edges:
                                conflict_edges.add((u, w))
                
                # Greedy coloring of (N_v, conflict_edges)
                deg_conf = {u: 0 for u in N_v}
                for u in N_v:
                    for w in N_v:
                        if u != w and (min(u, w), max(u, w)) in conflict_edges:
                            deg_conf[u] += 1
                
                ordered_N_v = sorted(N_v, key=lambda x: (-deg_conf[x], x))
                
                color_map = {}
                for u in ordered_N_v:
                    forbidden = set()
                    for w in N_v:
                        if u != w and (min(u, w), max(u, w)) in conflict_edges:
                            if w in color_map:
                                forbidden.add(color_map[w])
                    
                    c = 0
                    while c in forbidden:
                        c += 1
                    color_map[u] = c
                
                # Group by color
                color_classes = defaultdict(list)
                for u, c in color_map.items():
                    color_classes[c].append(u)
                
                # For each color class, create a clique
                for c, S in color_classes.items():
                    if not S:
                        continue
                    clique = [v] + sorted(S)
                    cliques.append(clique)
                    
                    # Mark C-edges within S as covered
                    for u1, u2 in combinations(sorted(S), 2):
                        covered_C_edges.add((u1, u2))
            
            # Cover remaining C-edges
            remaining_C_edges = all_C_edges - covered_C_edges
            
            while remaining_C_edges:
                u, w = min(remaining_C_edges)
                
                # Build maximal clique containing u, w
                clique_set = {u, w}
                
                for x in range(k):
                    if x in clique_set:
                        continue
                    can_add = True
                    for y in clique_set:
                        if (min(x, y), max(x, y)) not in remaining_C_edges:
                            can_add = False
                            break
                    if can_add:
                        clique_set.add(x)
                
                clique_list = sorted(clique_set)
                cliques.append(clique_list)
                
                for u1, u2 in combinations(clique_list, 2):
                    remaining_C_edges.discard((u1, u2))
            
            return cliques

        # Try multiple strategies and keep the best
        best_cliques = None
        best_count = float('inf')
        
        strategies = ["degree_desc", "degree_asc", "random"]
        
        for strat in strategies:
            try:
                cliques = solve_instance(strat)
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques
            except:
                continue
        
        if best_cliques is None:
            # Fallback: trivial cover
            cliques = []
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            for v in range(k, n):
                for u in neighbors_I[v]:
                    cliques.append([u, v])
            best_cliques = cliques
        
        return best_cliques

    result = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        cliques = solve(*spec)
        result.append(cliques)

    # Ensure we have exactly 3 results
    while len(result) < 3:
        spec = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(result)]
        cliques = []
        n, k, p, seed = spec
        rng = random.Random(seed)
        for u in range(k):
            for w in range(u + 1, k):
                cliques.append([u, w])
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cliques.append([u, v])
        result.append(cliques)

    return result[:3]
