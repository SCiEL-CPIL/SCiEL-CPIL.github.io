"""Rewritten program with optimal split-graph clique partition."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return edges, cross_edges

    def solve_split(n, k, p, seed):
        internal_edges, cross_edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        # S_v is the set of neighbors in C for v in I
        S = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            S[v].add(u)
            
        cliques = []
        
        # Track covered internal edges
        # Internal edges are (u, w) with u, w in C, u < w
        covered_internal = set()
        
        # 1. Process vertices in I
        for v in range(k, n):
            neighbors = S[v]
            if len(neighbors) >= 2:
                # Clique {v} U neighbors
                clique = [v] + list(neighbors)
                cliques.append(clique)
                # Mark internal edges in neighbors as covered
                n_list = list(neighbors)
                for i in range(len(n_list)):
                    for j in range(i+1, len(n_list)):
                        u, w = n_list[i], n_list[j]
                        if u > w: u, w = w, u
                        covered_internal.add((u, w))
            elif len(neighbors) == 1:
                u = next(iter(neighbors))
                cliques.append([u, v])
            # else len(neighbors) == 0, do nothing
        
        # 2. Process remaining internal edges
        # Find uncovered internal edges
        uncovered = []
        for u in range(k):
            for w in range(u+1, k):
                if (u, w) not in covered_internal:
                    uncovered.append((u, w))
        
        if uncovered:
            # Build graph H on C with edges = uncovered
            # We need min clique cover of H
            # This is min coloring of complement H_bar
            # Vertices are 0..k-1
            # H_bar edge (u, w) exists if (u, w) is NOT in uncovered
            
            # Build adjacency list for H_bar
            adj_bar = [set() for _ in range(k)]
            for u in range(k):
                for w in range(u+1, k):
                    if (u, w) not in set(uncovered):
                        adj_bar[u].add(w)
                        adj_bar[w].add(u)
            
            # Greedy coloring of H_bar
            # Order vertices by degree in H_bar descending (Welsh-Powell)
            order = sorted(range(k), key=lambda x: len(adj_bar[x]), reverse=True)
            
            colors = {}
            color_count = 0
            for v in order:
                # Find available color
                used_colors = set()
                for u in adj_bar[v]:
                    if u in colors:
                        used_colors.add(colors[u])
                        
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
                if c > color_count:
                    color_count = c
            
            # Form cliques from color classes
            color_classes = {}
            for v, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(v)
            
            # Add cliques, ensuring size >= 2
            # If a class has size 1, we need to handle it.
            # Since H has edges, there is at least one class with size >= 2?
            # Not necessarily. But if H has an edge (u,w), u and w are NOT adjacent in H_bar.
            # So they can have the same color.
            # So any edge in H corresponds to a pair in the same color class.
            # So any color class containing an edge of H has size >= 2.
            # What if a color class has size 1? It means the vertex is not in any edge of H?
            # i.e., it is isolated in H.
            # If isolated in H, it is universal in H_bar.
            # It can be merged into any other clique.
            
            cliques_in_C = []
            singletons = []
            for c, vertices in color_classes.items():
                if len(vertices) >= 2:
                    cliques_in_C.append(vertices)
                else:
                    singletons.append(vertices[0])
            
            # Merge singletons into existing cliques
            for s in singletons:
                if cliques_in_C:
                    # Add s to the first clique (or any)
                    # s is universal in H, so it connects to all in H.
                    # So s + clique is a clique.
                    cliques_in_C[0].append(s)
                else:
                    # No other cliques. This implies H was empty?
                    # But we checked if uncovered is not empty.
                    # If uncovered is not empty, there is at least one edge.
                    # So there must be at least one clique of size >= 2.
                    # So this case shouldn't happen.
                    pass
            
            cliques.extend(cliques_in_C)
            
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    return [solve_split(*spec) for spec in specs]
# EVOLVE-BLOCK-END
