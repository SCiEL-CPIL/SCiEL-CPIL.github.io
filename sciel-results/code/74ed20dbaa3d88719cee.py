import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        maximal_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(tuple(sorted(R)))
                return
            
            # Pivot selection: choose vertex in P union X with max neighbors in P
            # This optimizes the branching
            if P or X:
                # Simple pivot: pick first vertex in P if P else first in X
                # Better: max degree in P
                pivot_candidates = P | X
                # For small n, just pick one
                v = next(iter(pivot_candidates))
                candidates = P - adj[v]
            else:
                candidates = P
                
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        bron_kerbosch(set(), set(range(n)), set())
        return maximal_cliques

    def solve_instance(n, p, seed):
        if time.time() > deadline:
            # Fallback: return all vertices (valid but large)
            return list(range(n))
            
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        
        if not cliques:
            return []
            
        # Map clique index to set of vertices
        # We need to cover each clique with at least one vertex from the solution set S.
        
        # Greedy Set Cover
        # uncovered: set of clique indices
        uncovered = set(range(len(cliques)))
        selected = set()
        
        # Precompute for each vertex, which cliques it belongs to
        vertex_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        while uncovered:
            if time.time() > deadline:
                return sorted(selected)
                
            best_v = None
            best_count = -1
            # To make it slightly more random or varied if ties, but deterministic is fine for fixed inputs
            for v in range(n):
                if v in selected:
                    continue
                count = len(vertex_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
                    
            if best_v is None or best_count == 0:
                # Should not happen if uncovered is not empty
                break
                
            selected.add(best_v)
            # Remove covered cliques
            uncovered -= vertex_cliques[best_v]
            
        # Reduction Phase 1: Remove redundant single vertices
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                # Check if removing v leaves all cliques covered
                # A clique is covered if it contains any vertex in (selected - {v})
                # i.e., no clique is a subset of (all_vertices - (selected - {v}))?
                # Simpler: Check if any clique is disjoint from (selected - {v})
                # But we only care about cliques that were covered by v?
                # Actually, if a clique was covered by another vertex, it's fine.
                # So we just need to check if there exists a clique C such that C \ (selected - {v}) is empty?
                # No, we need C \ (selected - {v}) to NOT be empty for all C?
                # Wait. C is covered by S if S \cap C != empty.
                # If we remove v, new S' = S - {v}.
                # We need S' \cap C != empty for all C.
                # This is equivalent to: For all C, C is not a subset of (V - S').
                # Or: No C is disjoint from S'.
                
                # Optimization: Only check cliques that contain v?
                # If a clique C does not contain v, its coverage status doesn't change when v is removed.
                # So we only need to check cliques C such that v in C.
                # For such C, if C \ (selected - {v}) is empty, then C was ONLY covered by v.
                # If C \ (selected - {v}) is not empty, then C is still covered.
                
                is_redundant = True
                for i in vertex_cliques[v]:
                    c = cliques[i]
                    # Check if c has any other vertex in selected
                    has_other = False
                    for u in c:
                        if u != v and u in selected:
                            has_other = True
                            break
                    if not has_other:
                        is_redundant = False
                        break
                if is_redundant:
                    selected.remove(v)
                    changed = True
                    
        # Reduction Phase 2: Remove redundant pairs
        # Try removing pairs (v1, v2). 
        # This is O(|S|^2 * |Cliques|). Might be slow if |S| is large.
        # But |S| is likely small (< 10).
        changed = True
        while changed:
            changed = False
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    v1 = sel_list[i]
                    v2 = sel_list[j]
                    # Check if selected - {v1, v2} covers all cliques
                    # Only need to check cliques containing v1 or v2
                    affected_cliques = vertex_cliques[v1] | vertex_cliques[v2]
                    is_redundant = True
                    for idx in affected_cliques:
                        c = cliques[idx]
                        # Check if c has any vertex in selected - {v1, v2}
                        has_other = False
                        for u in c:
                            if u != v1 and u != v2 and u in selected:
                                has_other = True
                                break
                        if not has_other:
                            is_redundant = False
                            break
                    if is_redundant:
                        selected.remove(v1)
                        selected.remove(v2)
                        changed = True
                        break # Restart loop as list changed
                if changed:
                    break
                    
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            results.append(list(range(n)))
        else:
            results.append(solve_instance(n, p, seed))
            
    return results
