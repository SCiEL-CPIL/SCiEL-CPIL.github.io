"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    # Deadline guard: aim to finish well under 150s
    t0 = time.time()
    DEADLINE = 145.0  # safety margin
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch_maximal_cliques(adj):
        """Enumerate all maximal cliques with size >= 2 using Bron-Kerbosch with pivoting."""
        n = len(adj)
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Choose pivot from P ∪ X to minimize branching
            union_px = P | X
            if not union_px:
                return
            
            # Pick pivot with maximum degree in P
            pivot = max(union_px, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), frozenset(range(n)), frozenset())
        return maximal_cliques
    
    def greedy_set_cover(cliques, vertices):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        # Map each clique to its index
        uncovered = set(range(len(cliques)))
        # For each vertex, which cliques it covers
        vertex_cliques = {v: [] for v in vertices}
        for i, clique in enumerate(cliques):
            for v in clique:
                if v in vertex_cliques:
                    vertex_cliques[v].append(i)
        
        selected = set()
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in vertices:
                if v in selected:
                    continue
                count = sum(1 for i in vertex_cliques[v] if i in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count <= 0:
                # Fallback: pick any vertex from an uncovered clique
                for i in uncovered:
                    for v in cliques[i]:
                        if v not in selected:
                            best_v = v
                            break
                    if best_v is not None:
                        break
                if best_v is None:
                    break
            
            selected.add(best_v)
            for i in vertex_cliques.get(best_v, []):
                uncovered.discard(i)
        
        return selected
    
    def local_search_improve(selected, cliques, vertices, iterations=50):
        """Local search: try removing and swapping vertices to reduce size while maintaining coverage."""
        selected = set(selected)
        
        def covers_all(sel):
            for clique in cliques:
                if not (clique & sel):
                    return False
            return True
        
        # Phase 1: Remove redundant vertices
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                test = selected - {v}
                if covers_all(test):
                    selected = test
                    improved = True
        
        # Phase 2: Try swaps (remove one, add another) to potentially reduce size
        # Actually swaps don't reduce size; try 2-for-1 replacements
        for _ in range(iterations):
            best_solution = None
            best_size = len(selected)
            
            # Try removing pairs and replacing with one
            for v1, v2 in combinations(list(selected), 2):
                without = selected - {v1, v2}
                # Find which cliques are now uncovered
                uncovered_cliques = [i for i, c in enumerate(cliques) if not (c & without)]
                if not uncovered_cliques:
                    best_solution = without
                    best_size = len(without)
                    break
                # Find a single vertex that covers all uncovered cliques
                for w in vertices:
                    if w in selected:
                        continue
                    if all(w in cliques[i] for i in uncovered_cliques):
                        candidate = without | {w}
                        if len(candidate) < best_size:
                            best_size = len(candidate)
                            best_solution = candidate
                        break
            
            if best_solution is not None:
                selected = best_solution
            else:
                break
        
        return selected
    
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE - 20:
            # Not enough time; use a quick heuristic for remaining instances
            adj = build_graph(n, p, seed)
            # Quick greedy: pick vertices from largest cliques
            cliques = bron_kerbosch_maximal_cliques(adj)
            if not cliques:
                results.append([])
                continue
            # Greedy set cover as fallback
            all_vertices = set()
            for c in cliques:
                all_vertices.update(c)
            sel = greedy_set_cover(cliques, all_vertices)
            results.append(sorted(sel))
            continue
        
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            results.append([])
            continue
        
        all_vertices = set()
        for c in cliques:
            all_vertices.update(c)
        
        # Greedy set cover
        selected = greedy_set_cover(cliques, all_vertices)
        
        # Local search improvement
        if time.time() < DEADLINE - 10:
            selected = local_search_improve(selected, cliques, all_vertices, iterations=100)
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
