def construct():
    from random import Random
    import time
    from itertools import combinations
    from collections import defaultdict

    start_time = time.time()
    DEATHLINE = 138.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        # Reconstruct graph
        rng = Random(seed)
        neighbors_I = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors_I[v].add(u)
        
        C = list(range(k))
        I = list(range(k, n))
        
        # All edges in C
        all_C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_C_edges.add((u, w))
        
        cliques = []
        covered_C_edges = set()
        
        # Process I vertices in order of decreasing neighborhood size
        I_sorted = sorted(I, key=lambda v: len(neighbors_I[v]), reverse=True)
        
        for v in I_sorted:
            if time.time() > DEATHLINE - 2:
                break
            N_v = neighbors_I[v]
            if not N_v:
                continue
            
            # Build conflict graph on N_v: edge (a,b) exists if (a,b) is in covered_C_edges
            conflict_edges = set()
            N_v_list = sorted(N_v)
            for i, u in enumerate(N_v_list):
                for w in N_v_list[i+1:]:
                    if (u, w) in covered_C_edges:
                        conflict_edges.add((u, w))
            
            # Greedy coloring of conflict graph to find independent sets (= cliques in remaining C-graph)
            deg_conf = {u: 0 for u in N_v_list}
            for u in N_v_list:
                for w in N_v_list:
                    if u != w and (min(u, w), max(u, w)) in conflict_edges:
                        deg_conf[u] += 1
            
            ordered_N_v = sorted(N_v_list, key=lambda x: (-deg_conf[x], x))
            
            color_map = {}
            for u in ordered_N_v:
                forbidden = set()
                for w in N_v_list:
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
            
            # For each color class, create a clique {v} ∪ S
            for c, S in color_classes.items():
                if not S:
                    continue
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Mark C-edges within S as covered
                S_sorted = sorted(S)
                for i, u in enumerate(S_sorted):
                    for w in S_sorted[i+1:]:
                        covered_C_edges.add((u, w))
        
        # Cover remaining C-edges
        remaining_C_edges = all_C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Greedy clique cover for remaining edges
            # Build adjacency from remaining edges
            rem_adj = [set() for _ in range(k)]
            for u, w in remaining_C_edges:
                rem_adj[u].add(w)
                rem_adj[w].add(u)
            
            # Greedy: repeatedly find a maximal clique in remaining graph
            rem_edges = set(remaining_C_edges)
            while rem_edges:
                if time.time() > DEATHLINE - 1:
                    break
                # Find a maximal clique greedily
                # Pick vertex with highest degree in remaining graph
                degs = {u: len([w for w in rem_adj[u] if (min(u,w), max(u,w)) in rem_edges]) for u in range(k)}
                if not any(d > 0 for d in degs.values()):
                    break
                start = max(range(k), key=lambda u: degs[u])
                clique = [start]
                candidates = set(rem_adj[start]) & set(u for u in range(k) if any((min(u,w), max(u,w)) in rem_edges for w in [start]))
                
                # Greedily add vertices
                while candidates:
                    # Pick vertex with max connections to current clique
                    best = None
                    best_score = -1
                    for u in candidates:
                        score = len(set(clique) & rem_adj[u])
                        if score > best_score:
                            best_score = score
                            best = u
                    if best is None:
                        break
                    clique.append(best)
                    # Remove best from candidates, and remove non-neighbors
                    candidates = candidates & rem_adj[best]
                    # Remove best itself
                    candidates.discard(best)
                
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    # Remove edges within clique
                    for u, w in combinations(sorted(clique), 2):
                        rem_edges.discard((u, w))
                        rem_adj[u].discard(w)
                        rem_adj[w].discard(u)
                else:
                    # Single edge
                    e = next(iter(rem_edges))
                    cliques.append(list(e))
                    rem_edges.discard(e)
                    rem_adj[e[0]].discard(e[1])
                    rem_adj[e[1]].discard(e[0])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > DEATHLINE - 2:
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
    
    final = []
    for clique_list in results:
        final.append([list(int(v) for v in clique) for clique in clique_list])
    
    return final
