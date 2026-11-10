def construct():
    import time
    from random import Random
    import math

    deadline = time.time() + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def greedy_clique_cover(edges, available_edges):
        """
        Greedily find a clique cover for the graph defined by 'edges'.
        Returns list of cliques (lists of vertices).
        """
        if not edges:
            return []
        
        rem_edges = set(edges)
        cliques = []
        
        while rem_edges:
            # Find vertices involved
            vertices = set()
            for u, w in rem_edges:
                vertices.add(u)
                vertices.add(w)
            
            if not vertices:
                break
                
            # Greedy maximal clique in the subgraph defined by rem_edges
            S = []
            S_set = set()
            cand_set = set(vertices)
            
            while cand_set:
                # Pick vertex with max degree in cand_set (within rem_edges)
                best_v = None
                best_deg = -1
                for w in cand_set:
                    deg = 0
                    for x in cand_set:
                        if x == w: continue
                        e = (min(w, x), max(w, x))
                        if e in rem_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = w
                        
                if best_v is None:
                    break
                
                S.append(best_v)
                S_set.add(best_v)
                
                # Update cand_set: keep only neighbors of best_v in cand_set
                new_cand = set()
                for w in cand_set:
                    if w == best_v: continue
                    e = (min(best_v, w), max(best_v, w))
                    if e in rem_edges:
                        new_cand.add(w)
                cand_set = new_cand
                
            if S:
                cliques.append(S)
                # Remove edges in S from rem_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                # Fallback: add a single edge
                e = next(iter(rem_edges))
                cliques.append([e[0], e[1]])
                rem_edges.discard(e)
                
        return cliques

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is set of neighbors in C for v in I
        N = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            N[v].add(u)
            
        # Edges in G[C]
        C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                C_edges.add((u, w))
                
        # Initialize assignment: all C_edges assigned to 'Pure' (index -1)
        I_vertices = list(range(k, n))
        
        assignment = {e: -1 for e in C_edges} # -1 represents Pure
        
        # Pre-compute which v's can take which edge
        edge_to_vs = {e: [] for e in C_edges}
        for v in I_vertices:
            if not N[v]:
                continue
            for e in C_edges:
                u, w = e
                if u in N[v] and w in N[v]:
                    edge_to_vs[e].append(v)
                    
        buckets = {v: set() for v in I_vertices}
        buckets[-1] = set(C_edges) # Pure
        
        def calc_cost(target):
            edges = buckets[target]
            if not edges:
                return 0
            cliques = greedy_clique_cover(edges, C_edges)
            return len(cliques)
            
        # Simulated Annealing
        rng = Random(seed + 10000)
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        # Time budget for SA
        sa_deadline = time.time() + 110.0
        
        while time.time() < sa_deadline and T > min_T:
            e = rng.choice(list(C_edges))
            current_target = assignment[e]
            
            possible_targets = [-1] + edge_to_vs[e]
            
            if len(possible_targets) <= 1:
                continue
                
            new_target = rng.choice(possible_targets)
            if new_target == current_target:
                continue
                
            old_cost_c = calc_cost(current_target)
            old_cost_n = calc_cost(new_target)
            
            buckets[current_target].discard(e)
            buckets[new_target].add(e)
            
            new_cost_c = calc_cost(current_target)
            new_cost_n = calc_cost(new_target)
            
            delta = (new_cost_c + new_cost_n) - (old_cost_c + old_cost_n)
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                assignment[e] = new_target
            else:
                buckets[new_target].discard(e)
                buckets[current_target].add(e)
                
            T *= cooling
            
        # Construct final cliques
        result_cliques = []
        
        pure_edges = buckets[-1]
        pure_cliques = greedy_clique_cover(pure_edges, C_edges)
        for c in pure_cliques:
            if len(c) >= 2:
                result_cliques.append([int(x) for x in c])
                
        for v in I_vertices:
            v_edges = buckets[v]
            if not v_edges:
                continue
            v_cliques = greedy_clique_cover(v_edges, C_edges)
            for c in v_cliques:
                if len(c) >= 2:
                    result_cliques.append([int(v)] + [int(x) for x in c])
                    
        return result_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    
    for (n, k, p, seed) in specs:
        if time.time() > deadline - 10.0:
            cross_edges = instance(n, k, p, seed)
            cliques = []
            for u in range(k):
                for w in range(u+1, k):
                    cliques.append([u, w])
            for u, v in cross_edges:
                cliques.append([u, v])
            result.append(cliques)
            continue
            
        cliques = solve_instance(n, k, p, seed)
        result.append(cliques)
        
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
