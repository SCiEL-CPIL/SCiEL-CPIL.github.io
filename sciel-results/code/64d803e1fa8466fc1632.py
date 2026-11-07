"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

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

    def solve_split(n, k, p, seed):
        """Find a minimal clique partition for a split graph.
        
        In a split graph with clique part C and independent set I:
        - Any clique contains at most one vertex from I
        - So cliques are either subsets of C, or {i} ∪ S where i ∈ I, S ⊆ N_C(i) clique in C
        
        Strategy:
        1. Enumerate all maximal cliques in the graph
        2. Use greedy set cover: repeatedly pick the clique covering the most uncovered edges
        """
        edges = instance(n, k, p, seed)
        edge_set = set()
        for e in edges:
            if e[0] < e[1]:
                edge_set.add(e)
            else:
                edge_set.add((e[1], e[0]))
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # C vertices: 0..k-1, I vertices: k..n-1
        C = list(range(k))
        I = list(range(k, n))
        
        # Enumerate all maximal cliques using Bron-Kerbosch with pivot
        all_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    all_cliques.append(frozenset(R))
                return
            # Choose pivot to minimize branching
            if P or X:
                pivot_candidates = P | X
                pivot = max(pivot_candidates, key=lambda v: len(pivot_candidates & adj[v]))
            else:
                return
            # Branch on P \ N(pivot)
            for v in list(P - adj[pivot]):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)
        
        # Start with empty R, all vertices in P, empty X
        bron_kerbosch(set(), set(range(n)), set())
        
        # For each maximal clique, compute which edges it covers
        clique_edges = []
        for clique in all_cliques:
            clist = list(clique)
            edge_list = []
            for i in range(len(clist)):
                for j in range(i + 1, len(clist)):
                    a, b = clist[i], clist[j]
                    e = (min(a, b), max(a, b))
                    if e in edge_set:
                        edge_list.append(e)
            if edge_list:
                clique_edges.append((clique, frozenset(edge_list)))
        
        # Greedy set cover: repeatedly pick the clique covering the most uncovered edges
        uncovered = set(edge_set)
        cliques_result = []
        
        # Sort cliques by size descending for initial preference
        clique_edges.sort(key=lambda x: -len(x[1]))
        
        while uncovered:
            if time.time() > deadline:
                break
            best_clique = None
            best_count = 0
            for clique, cl_edges in clique_edges:
                count = len(cl_edges & uncovered)
                if count > best_count:
                    best_count = count
                    best_clique = clique
                elif count == best_count and best_clique is not None:
                    # Tie-break: prefer larger clique
                    if len(clique) > len(best_clique):
                        best_clique = clique
            if best_clique is None or best_count == 0:
                # Should not happen, but safety
                break
            cliques_result.append(list(best_clique))
            # Remove covered edges
            cl_edges = frozenset()
            clist = list(best_clique)
            for i in range(len(clist)):
                for j in range(i + 1, len(clist)):
                    a, b = clist[i], clist[j]
                    cl_edges.add((min(a, b), max(a, b)))
            uncovered -= cl_edges
        
        return cliques_result

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            pass
        cliques = solve_split(*spec)
        result.append(cliques)

    return result
# EVOLVE-BLOCK-END
