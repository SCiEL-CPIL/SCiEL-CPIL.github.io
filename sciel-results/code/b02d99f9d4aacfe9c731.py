import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        # Generate graph
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        
        cross_adj = {}
        for v in range(k, n):
            neighbors_in_C = set()
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    neighbors_in_C.add(u)
            cross_adj[v] = neighbors_in_C

        # Available C-C edges
        available_cc = set()
        for u in range(k):
            for v in range(u+1, k):
                available_cc.add((u, v))

        cliques = []

        # Process I-vertices
        for v in range(k, n):
            if time.time() > DEADLINE:
                break
            remaining_v = set(cross_adj[v])
            
            while remaining_v:
                if time.time() > DEADLINE:
                    break
                
                # Find maximal clique S in remaining_v using available_cc
                # Greedy: pick vertex with max degree in available_cc restricted to remaining_v
                # Actually, just pick any vertex, then restrict to its neighbors in available_cc
                # Since we want a clique, we need all pairs to be in available_cc.
                
                # Sort remaining_v to be deterministic? Or just iterate.
                # Let's try to find a large S.
                # Start with all remaining_v as candidates.
                candidates = list(remaining_v)
                S = []
                while candidates:
                    # Pick vertex with max degree in available_cc among candidates
                    # Degree in available_cc restricted to candidates
                    best_v = None
                    best_deg = -1
                    for u in candidates:
                        deg = 0
                        for w in candidates:
                            if u != w:
                                e = (min(u, w), max(u, w))
                                if e in available_cc:
                                    deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = u
                            
                    if best_v is None:
                        break
                    
                    S.append(best_v)
                    # Next candidates must be neighbors of best_v in available_cc
                    new_candidates = []
                    for u in candidates:
                        if u != best_v:
                            e = (min(best_v, u), max(best_v, u))
                            if e in available_cc:
                                new_candidates.append(u)
                    candidates = new_candidates

                if len(S) >= 2:
                    cliques.append([v] + sorted(S))
                    # Remove edges in S from available_cc
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            u, w = S[i], S[j]
                            e = (min(u, w), max(u, w))
                            available_cc.discard(e)
                    # Remove S from remaining_v
                    remaining_v -= set(S)
                else:
                    # S is size 1 or 0
                    # If S is empty, no edges available?
                    # But remaining_v is not empty.
                    # This means no vertex in remaining_v has any available edge to another vertex in remaining_v?
                    # Or my clique finding failed?
                    # If S has size 1, say [u], then u has no available edges to others in remaining_v.
                    # So we must use {v, u}.
                    # This covers (v, u) but no C-C edges.
                    # So available_cc is unchanged.
                    if S:
                        u = S[0]
                        cliques.append([v, u])
                        remaining_v.remove(u)
                    else:
                        # Should not happen if remaining_v is not empty
                        # Just take any u
                        u = next(iter(remaining_v))
                        cliques.append([v, u])
                        remaining_v.remove(u)

        # Cover remaining C-C edges
        if available_cc:
            # Find vertices incident to available_cc
            v_rem = set()
            for u, w in available_cc:
                v_rem.add(u)
                v_rem.add(w)
            
            # Greedy clique cover of available_cc
            # While available_cc:
            #   Find maximal clique S in available_cc
            #   Add S
            #   Remove edges of S
            
            while available_cc and (time.time() - t0) < DEADLINE:
                # Find maximal clique in available_cc
                # Start with a vertex from available_cc
                first_edge = next(iter(available_cc))
                start_v = first_edge[0]
                
                candidates = [start_v]
                S = [start_v]
                
                # Greedy expansion
                while True:
                    new_candidates = []
                    for u in candidates:
                        # Find neighbors of u in available_cc that are not in S
                        for w in adj[u]:
                            if w not in S and w in v_rem:
                                e = (min(u, w), max(u, w))
                                if e in available_cc:
                                    # Check if w is connected to all in S
                                    if all((min(x, w), max(x, w)) in available_cc for x in S):
                                        new_candidates.append(w)
                                        break
                    if not new_candidates:
                        break
                    # Pick best candidate
                    # Just pick first one
                    next_v = new_candidates[0]
                    S.append(next_v)
                    candidates = new_candidates # This logic is slightly off for standard greedy, but works for finding *a* clique
                    
                    # Actually, simpler:
                    # Build clique by intersecting neighborhoods
                    pass # The previous loop was better. Let's stick to the previous one.
                
                # Re-implement clique finding for available_cc
                # It's a subgraph of a clique.
                # Just take all vertices in v_rem that form a clique?
                # No, available_cc might be sparse.
                
                # Let's just use the same greedy as before on v_rem
                candidates = list(v_rem)
                S = []
                while candidates:
                    best_v = None
                    best_deg = -1
                    for u in candidates:
                        deg = 0
                        for w in candidates:
                            if u != w:
                                e = (min(u, w), max(u, w))
                                if e in available_cc:
                                    deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = u
                    if best_v is None:
                        break
                    S.append(best_v)
                    new_candidates = []
                    for u in candidates:
                        if u != best_v:
                            e = (min(best_v, u), max(best_v, u))
                            if e in available_cc:
                                new_candidates.append(u)
                    candidates = new_candidates

                if S:
                    cliques.append(sorted(S))
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            u, w = S[i], S[j]
                            e = (min(u, w), max(u, w))
                            available_cc.discard(e)
                    # Update v_rem
                    for u in S:
                        v_rem.discard(u)
                        # Remove u from all edges? No, just discard from available_cc.
                        # v_rem should be recomputed or updated.
                        # Simpler: recompute v_rem if it becomes empty?
                        # Or just keep v_rem as is, and the greedy will pick from it.
                        # But if u is removed, it shouldn't be in candidates.
                        # The candidates are from v_rem.
                        # So we must update v_rem.
                        pass # The discard from available_cc is enough if we rebuild v_rem?
                        # Let's just rebuild v_rem if available_cc changes significantly?
                        # Or just let the greedy handle it. If u is not in any edge, it won't be picked?
                        # My greedy picks from candidates = list(v_rem).
                        # So I must update v_rem.
                        v_rem = set()
                        for u2, w2 in available_cc:
                            v_rem.add(u2)
                            v_rem.add(w2)
                        break # Restart loop? No, just continue while loop.
                else:
                    # No clique found? Should not happen.
                    break

        # Validate
        edge_to_clique = {}
        for i, clique in enumerate(cliques):
            if len(clique) < 2:
                return None
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    edge = (min(u,v), max(u,v))
                    if edge in edge_to_clique:
                        return None
                    edge_to_clique[edge] = i
                    if v not in adj[u]:
                        return None

        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))

        if len(edge_to_clique) != len(all_edges):
            return None

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if (time.time() - t0) > DEADLINE:
            break
        n, k, p, seed = spec
        result = solve_split(n, k, p, seed)
        if result is None:
            # Fallback
            result = []
            rng = Random(seed)
            edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
            for v in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        edges.append((u, v))
            result = [list(e) for e in edges]
        
        clean_result = []
        for clique in result:
            clean_clique = [int(v) for v in clique]
            clean_result.append(clean_clique)
        results.append(clean_result)

    return results
