import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        C_vertices = list(range(k))
        I_vertices = list(range(k, n))
        
        # 1. Generate Candidate Cliques
        # A clique in a split graph is either a subset of C (clique in G[C]) or {i} U S where i in I and S is a clique in G[C] subset of N(i).
        # We are interested in cliques that cover cross-edges.
        # Candidate: {i} U S, where S is a maximal clique in the subgraph induced by N(i) on C_vertices.
        
        candidates = [] # List of (clique_list, set_of_cross_edges_covered)
        
        for i in I_vertices:
            N_i = adj[i] & set(C_vertices)
            if not N_i:
                continue
            
            # Find maximal cliques in G[N_i]
            # Since N_i is a subset of C, and C is a clique, G[N_i] is a clique itself.
            # Wait, G[C] is a clique. So any subset of N_i is a clique.
            # The largest clique S in N_i is N_i itself.
            # So the only candidate for i is {i} U N_i.
            # Is that true?
            # Yes, because C is a clique. So all vertices in N_i are mutually adjacent.
            # So S = N_i is the unique maximal clique containing i.
            
            S = list(N_i)
            clique = [i] + S
            if len(clique) >= 2:
                # Calculate cross-edges covered: (i, c) for c in S
                cross_edges = frozenset((min(i, c), max(i, c)) for c in S)
                candidates.append((clique, cross_edges))
        
        # 2. Greedy Set Cover for Cross-Edges
        # Universe: All cross-edges
        cross_edges = set()
        for u in range(k):
            for v in range(k, n):
                if v in adj[u]:
                    cross_edges.add((u, v))
        
        uncovered = set(cross_edges)
        selected_cliques = []
        
        # Pre-compute coverage for each candidate
        # candidate: (clique_list, cross_edges_frozenset)
        
        # To speed up, we can sort candidates by size descending, but greedy picks max overlap.
        # We will iterate until no more uncovered edges or no candidates left.
        
        # Optimization: Group candidates by the set of cross edges? No, just iterate.
        # For n=90, k=30, I has 60 vertices. Candidates = 60.
        # Cross edges ~ 30 * 60 * 0.7 = 1260.
        # Greedy steps: at most 1260 (if each picks 1), usually much fewer.
        
        while uncovered:
            if time.time() > DEADLINE - 5:
                break
            
            best_candidate_idx = -1
            best_overlap = 0
            best_clique_list = None
            best_cross_set = None
            
            for idx, (clique_list, cross_set) in enumerate(candidates):
                # Count intersection
                # Since sets are small, len(cross_set & uncovered) is fast
                overlap = len(cross_set & uncovered)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_candidate_idx = idx
                    best_clique_list = clique_list
                    best_cross_set = cross_set
            
            if best_overlap == 0:
                break
            
            selected_cliques.append(best_clique_list)
            uncovered -= best_cross_set
            
            # Remove the selected candidate to avoid re-selection?
            # Actually, we can remove it because selecting it again adds no new edges.
            candidates.pop(best_candidate_idx)
            
            if not candidates:
                break
        
        # 3. Cover Remaining C-Edges
        # The edges covered by selected I-cliques are exactly the cross-edges in their cross_set.
        # Wait, the selected cliques are {i} U S. They cover cross-edges (i, c).
        # Do they cover C-edges (c1, c2)?
        # The task requires: "every edge must lie in exactly one listed clique".
        # If we list {i, c1, c2}, it contains edges (i, c1), (i, c2), AND (c1, c2).
        # So yes, the C-edges within S are also covered by this clique.
        
        # We need to track which C-edges are covered.
        covered_C_edges = set()
        for clique in selected_cliques:
            # Extract C vertices from clique
            C_in_clique = [v for v in clique if v < k]
            for u_idx in range(len(C_in_clique)):
                for v_idx in range(u_idx + 1, len(C_in_clique)):
                    u, v = C_in_clique[u_idx], C_in_clique[v_idx]
                    if u < v:
                        covered_C_edges.add((u, v))
                    else:
                        covered_C_edges.add((v, u))
        
        all_C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_C_edges.add((u, v))
        
        remaining_C_edges = all_C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Greedy Clique Cover of the residual C-graph
            remaining = set(remaining_C_edges)
            
            while remaining:
                if time.time() > DEADLINE - 5:
                    break
                
                v_set = set()
                for u, v in remaining:
                    v_set.add(u)
                    v_set.add(v)
                
                if not v_set:
                    break
                
                adj_sub = {v: set() for v in v_set}
                for u, v in remaining:
                    adj_sub[u].add(v)
                    adj_sub[v].add(u)
                
                # Greedy Max Clique
                # Sort by degree
                candidates = list(v_set)
                candidates.sort(key=lambda x: len(adj_sub[x]), reverse=True)
                
                def max_clique_search(cands, current):
                    if not cands:
                        return current
                    v = cands[0]
                    new_cands = [u for u in cands[1:] if u in adj_sub[v]]
                    res1 = max_clique_search(new_cands, current + [v])
                    res2 = max_clique_search(cands[1:], current)
                    return res1 if len(res1) > len(res2) else res2
                
                best_clique = max_clique_search(candidates, [])
                
                if not best_clique:
                    u, v = next(iter(remaining))
                    best_clique = [u, v]
                
                selected_cliques.append(best_clique)
                
                # Remove edges covered by this clique
                for u in best_clique:
                    for v in best_clique:
                        if u < v:
                            remaining.discard((u, v))
                        else:
                            remaining.discard((v, u))
        
        # 4. Finalize
        final_result = []
        for clique in selected_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: Trivial solution (each edge as a clique)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
