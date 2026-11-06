"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def bron_kerbosch(R, P, X, adj, cliques):
        """Bron-Kerbosch with pivot to enumerate all maximal cliques."""
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot to minimize branching
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
        best_pivot = max(pivot_candidates, key=lambda v: len(P & adj[v]))
        
        # Only iterate over P \ N(pivot)
        for v in list(P - adj[best_pivot]):
            P.remove(v)
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques)
            X.add(v)
    
    def find_maximal_cliques(n, edges):
        adj = build_adjacency(n, edges)
        cliques = []
        bron_kerbosch(set(), set(range(n)), set(), adj, cliques)
        return cliques, adj
    
    def greedy_transversal(n, edges, cliques, adj):
        """Greedy set cover: repeatedly pick vertex that covers most uncovered cliques."""
        uncovered = set(range(len(cliques)))
        selected = set()
        
        # Precompute for each vertex, which cliques it belongs to
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                count = len(vertex_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
            
            selected.add(best_v)
            uncovered -= vertex_cliques[best_v]
        
        return selected
    
    def local_search_improve(n, selected, cliques, adj, time_limit):
        """Local search: try to remove vertices or swap to reduce size."""
        selected = set(selected)
        start = time.time()
        
        # Precompute coverage counts
        def coverage_counts(sel):
            counts = [0] * len(cliques)
            for i, clique in enumerate(cliques):
                for v in clique:
                    if v in sel:
                        counts[i] += 1
            return counts
        
        # Check if all cliques are covered
        def is_valid(sel):
            counts = coverage_counts(sel)
            return all(c >= 1 for c in counts)
        
        # Try removing vertices
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            counts = coverage_counts(selected)
            for v in list(selected):
                # Check if v is the only cover for any clique
                can_remove = True
                for i, clique in enumerate(cliques):
                    if v in clique and counts[i] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected.remove(v)
                    improved = True
                    break
        
        # Try swaps: remove v, add w if it helps
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            counts = coverage_counts(selected)
            for v in list(selected):
                # Cliques that v covers
                v_cliques = [i for i, c in enumerate(cliques) if v in c]
                # For each such clique, find alternative vertices
                for i in v_cliques:
                    if counts[i] > 1:
                        continue  # Already covered by others
                    # Need to find a vertex w that covers clique i
                    clique_i = cliques[i]
                    for w in clique_i:
                        if w not in selected:
                            # Check if adding w and removing v maintains validity
                            new_selected = selected - {v} | {w}
                            if is_valid(new_selected):
                                selected = new_selected
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return selected
    
    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        cliques, adj = find_maximal_cliques(n, edges)
        
        if not cliques:
            return set()
        
        # Start with greedy
        selected = greedy_transversal(n, edges, cliques, adj)
        
        # Local search improvement
        time_limit = time.time() + (deadline - time.time()) * 0.3
        selected = local_search_improve(n, selected, cliques, adj, time_limit)
        
        return selected
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Not enough time, return empty or minimal
            out.append(set())
            continue
        
        selected = solve_instance(n, p, seed)
        out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END
