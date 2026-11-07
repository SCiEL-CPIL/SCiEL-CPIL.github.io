import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for n, k, p, seed in specs:
        edges = instance(n, k, p, seed)
        
        C = list(range(k))
        I = list(range(k, n))
        
        S = {u: set() for u in C}
        
        for u, v in edges:
            if u < k and v < k:
                continue
            elif u >= k and v >= k:
                continue
            else:
                if u < k:
                    cu, iv = u, v
                else:
                    cu, iv = v, u
                S[cu].add(iv)
        
        N_C = {v: set() for v in I}
        for u in C:
            for v in S[u]:
                N_C[v].add(u)
                
        cliques = []
        for v in I:
            if N_C[v]:
                cliques.append([v] + sorted(N_C[v]))
        
        remaining_edges = []
        for i in range(k):
            for j in range(i + 1, k):
                u, w = i, j
                if not (S[u] & S[w]):
                    remaining_edges.append((u, w))
        
        if remaining_edges:
            adj_H = {u: set() for u in C}
            for u, w in remaining_edges:
                adj_H[u].add(w)
                adj_H[w].add(u)
            
            rem_edges_set = set(frozenset(e) for e in remaining_edges)
            
            # Pre-compute maximal cliques for each edge?
            # Or compute on the fly.
            
            cache = {}
            
            def solve(rem_set):
                if not rem_set:
                    return []
                key = frozenset(rem_set)
                if key in cache:
                    return cache[key]
                
                e = next(iter(rem_set))
                u, w = tuple(e)
                
                # Find maximal cliques in H containing u, w
                common = adj_H[u] & adj_H[w]
                
                # Generate all cliques containing u, w
                all_cliques = []
                def gen(curr, avail):
                    all_cliques.append(frozenset(curr))
                    for v in list(avail):
                        if all(x in adj_H[v] for x in curr):
                            new_avail = {x for x in avail if x != v and x in adj_H[v]}
                            gen(curr | {v}, new_avail)
                
                gen({u, w}, common)
                
                # Filter maximal
                maximal = []
                for c in all_cliques:
                    is_max = True
                    for c2 in all_cliques:
                        if c != c2 and c < c2:
                            is_max = False
                            break
                    if is_max:
                        maximal.append(c)
                
                best = None
                for clique in maximal:
                    clique_vertices = list(clique)
                    clique_edges = set()
                    for i in range(len(clique_vertices)):
                        for j in range(i + 1, len(clique_vertices)):
                            clique_edges.add(frozenset((clique_vertices[i], clique_vertices[j])))
                    
                    new_rem = rem_set - clique_edges
                    sub_sol = solve(new_rem)
                    if sub_sol is None:
                        continue
                    candidate = [list(clique)] + sub_sol
                    if best is None or len(candidate) < len(best):
                        best = candidate
                        
                cache[key] = best
                return best

            partition = solve(rem_edges_set)
            if partition:
                cliques.extend(partition)
        
        final_cliques = []
        for c in cliques:
            if len(c) >= 2:
                final_cliques.append(list(c))
        
        results.append(final_cliques)
        
        if time.time() > deadline:
            break

    return results
