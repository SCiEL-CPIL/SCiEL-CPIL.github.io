"""Clique partition of split graphs using global clique extraction and assignment."""
import time
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _bron_kerbosch_max_cliques(adj, min_size=2, max_results=50):
        """Find up to max_results maximum cliques in graph defined by adj (list of sets)."""
        results = []
        n = len(adj)
        
        def bk(R, P, X):
            if _time_up():
                return
            if not P and not X:
                if len(R) >= min_size:
                    results.append(frozenset(R))
                    if len(results) >= max_results:
                        return
                return
            # Pivot selection
            candidates = list(P | X)
            if not candidates:
                return
            pivot = max(candidates, key=lambda v: len(P & adj[v]))
            # Iterate over P \ N(pivot)
            for v in list(P - adj[pivot]):
                if _time_up():
                    return
                if len(results) >= max_results:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return [list(c) for c in results]

    def _greedy_max_clique(adj, start_vertex=None):
        """Find a maximal clique starting from start_vertex using greedy approach."""
        if start_vertex is None:
            start_vertex = 0
        clique = {start_vertex}
        candidates = set(adj[start_vertex])
        
        while candidates:
            # Pick vertex with max degree among candidates
            best_v = max(candidates, key=lambda v: len(candidates & adj[v]))
            clique.add(best_v)
            candidates = candidates & adj[best_v]
        
        return clique

    def _color_graph(adj, k):
        """Color graph with k vertices using greedy DSATUR-like heuristic.
        Returns list of color classes (each is a set of vertices)."""
        colored = [None] * k
        colors_used = 0
        # Order vertices by degree descending
        order = sorted(range(k), key=lambda v: len(adj[v]), reverse=True)
        
        for v in order:
            # Find smallest available color
            used_colors = set()
            for u in adj[v]:
                if colored[u] is not None:
                    used_colors.add(colored[u])
            c = 0
            while c in used_colors:
                c += 1
            colored[v] = c
            colors_used = max(colors_used, c + 1)
        
        # Build color classes
        color_classes = []
        for c in range(colors_used):
            cls = [v for v in range(k) if colored[v] == c]
            if len(cls) >= 2:
                color_classes.append(sorted(cls))
        return color_classes

    def _solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Build C-vertex adjacency (edges among C vertices)
        C_adj = [set() for _ in range(k)]
        for (a, b) in edge_set:
            if a < k and b < k:
                C_adj[a].add(b)
                C_adj[b].add(a)

        # Build I-vertex neighbor sets (subset of C)
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        I_verts = [v for v in range(k, n) if len(N_C[v]) > 0]

        if not I_verts:
            # No I-edges, just cover C-C edges
            color_classes = _color_graph(C_adj, k)
            return color_classes

        # Strategy: Find good C-cliques, assign I-vertices to them
        # Step 1: Find several candidate C-cliques using Bron-Kerbosch
        candidate_cliques = _bron_kerbosch_max_cliques(C_adj, min_size=2, max_results=30)
        
        # Also add some greedy maximal cliques for diversity
        for start in range(min(k, 10)):
            gc = _greedy_max_clique(C_adj, start)
            if len(gc) >= 2:
                gc_list = list(gc)
                if gc_list not in candidate_cliques:
                    candidate_cliques.append(gc_list)
        
        if not candidate_cliques:
            # Fallback: use edges directly
            cliques = []
            for (a, b) in edge_set:
                if a < k and b < k:
                    cliques.append([a, b])
                else:
                    cliques.append([a, b])
            return cliques

        # Step 2: For each candidate set of C-cliques, assign I-vertices
        best_cliques = None
        best_count = float('inf')

        # Try different subsets of candidate cliques
        # Sort by size descending
        candidate_cliques.sort(key=lambda c: -len(c))

        # Try using top M cliques for different M
        for M in [len(candidate_cliques), max(1, len(candidate_cliques)//2), 5, 3, 2]:
            if _time_up():
                break
            M = min(M, len(candidate_cliques))
            chosen = candidate_cliques[:M]
            
            # Assign each I-vertex to the best clique
            # A clique S can absorb I-vertex v if N_C[v] ⊆ S
            # We want to maximize edges covered per clique
            # Greedy: for each I-vertex, pick clique that covers most of its neighbors
            # and hasn't been "used up" too much
            
            # Track which C-C edges are already covered by chosen cliques
            covered_cc = set()
            for clique in chosen:
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e in edge_set:
                            covered_cc.add(e)
            
            # For each I-vertex, find best clique
            # Score: number of its neighbors in the clique that are NOT already covered
            assignment = {}
            for v in I_verts:
                neigh = N_C[v]
                best_score = -1
                best_clique_idx = -1
                for idx, clique in enumerate(chosen):
                    clique_set = set(clique)
                    # New edges this would cover: (v, c) for c in clique where c in neigh and edge not covered
                    # Actually all (v,c) edges are new since v is an I-vertex
                    # But we also cover C-C edges within clique (already counted)
                    # Score = number of neighbors in clique
                    count = len(neigh & clique_set)
                    if count > best_score:
                        best_score = count
                        best_clique_idx = idx
                
                if best_clique_idx >= 0 and best_score >= 1:
                    assignment[v] = best_clique_idx
                else:
                    # No clique can absorb this vertex well
                    # Create a standalone edge clique
                    assignment[v] = -1

            # Build final clique list
            final_cliques = []
            
            # Add chosen C-cliques (only those with at least one I-vertex assigned or size >= 2)
            used_cliques = set()
            for v in I_verts:
                idx = assignment[v]
                if idx >= 0:
                    used_cliques.add(idx)
            
            for idx in used_cliques:
                final_cliques.append(list(chosen[idx]))
            
            # Add I-vertices to their assigned cliques
            for v in I_verts:
                idx = assignment[v]
                if idx >= 0:
                    final_cliques[idx].append(v)
            
            # Handle unassigned I-vertices: create edge cliques
            for v in I_verts:
                if assignment[v] == -1:
                    neigh = list(N_C[v])
                    if neigh:
                        # Just make a 2-clique with one neighbor
                        final_cliques.append([v, neigh[0]])
            
            # Cover remaining C-C edges not in any chosen clique
            # Find uncovered C-C edges
            remaining_cc = set()
            for (a, b) in edge_set:
                if a < k and b < k:
                    e = (a, b) if a < b else (b, a)
                    # Check if this edge is in any chosen clique
                    in_clique = False
                    for idx in used_cliques:
                        if a in chosen[idx] and b in chosen[idx]:
                            in_clique = True
                            break
                    if not in_clique:
                        remaining_cc.add(e)
            
            if remaining_cc:
                # Color the subgraph of remaining C-C edges
                rem_adj = [set() for _ in range(k)]
                for (a, b) in remaining_cc:
                    rem_adj[a].add(b)
                    rem_adj[b].add(a)
                
                color_classes = _color_graph(rem_adj, k)
                final_cliques.extend(color_classes)
            
            # Validate: every clique must be a valid clique (all pairs are edges)
            # and size >= 2
            valid_cliques = []
            for clique in final_cliques:
                if len(clique) < 2:
                    continue
                # Check all pairs are edges
                is_valid = True
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e not in edge_set:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    valid_cliques.append(clique)
            
            count = len(valid_cliques)
            if count < best_count:
                best_count = count
                best_cliques = valid_cliques

        if best_cliques is None or not best_cliques:
            # Fallback
            best_cliques = []
            for (a, b) in edge_set:
                best_cliques.append([a, b])

        return best_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            for (u, v) in edges:
                cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        cliques = _solve_one(n, k, p, seed)
        results.append(cliques)

    # Type conversion
    for i in range(len(results)):
        new_list = []
        for clique in results[i]:
            new_clique = [int(x) for x in clique]
            new_list.append(new_clique)
        results[i] = new_list

    return results
