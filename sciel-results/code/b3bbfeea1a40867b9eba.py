def construct():
    import time
    import random
    
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

    def get_maximal_cliques(n, k, edges):
        cliques = []
        cliques.append(frozenset(range(k)))
        
        adj_I = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                adj_I[v].add(u)
            elif v < k and u >= k:
                adj_I[u].add(v)
        
        for v in range(k, n):
            if adj_I[v]:
                clique = frozenset({v} | adj_I[v])
                cliques.append(clique)
        
        return cliques

    def solve_exact_cover(edges, cliques, time_limit=10):
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        edge_to_cliques = {e: [] for e in edge_set}
        for i, c in enumerate(cliques):
            for u in c:
                for v in c:
                    if u < v:
                        e = (u, v)
                        if e in edge_set:
                            edge_to_cliques[e].append(i)
        
        for e in edge_set:
            if not edge_to_cliques[e]:
                return None
        
        solution = []
        start_time = time.time()
        
        def backtrack(uncovered_edges, used_cliques):
            if time.time() - start_time > time_limit:
                return None
            
            if not uncovered_edges:
                return list(solution)
            
            min_edge = None
            min_count = float('inf')
            for e in uncovered_edges:
                available = [i for i in edge_to_cliques[e] if i not in used_cliques]
                if len(available) < min_count:
                    min_count = len(available)
                    min_edge = e
                    if min_count == 0:
                        return None
                    if min_count == 1:
                        break
            
            if min_edge is None:
                return None
                
            available = [i for i in edge_to_cliques[min_edge] if i not in used_cliques]
            
            for i in available:
                clique = cliques[i]
                
                valid = True
                for u in clique:
                    for v in clique:
                        if u < v:
                            e_prime = (u, v)
                            if e_prime in edge_set and e_prime not in uncovered_edges:
                                valid = False
                                break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                solution.append(i)
                used_cliques.add(i)
                
                new_uncovered = set(uncovered_edges)
                for u in clique:
                    for v in clique:
                        if u < v:
                            e_prime = (u, v)
                            if e_prime in new_uncovered:
                                new_uncovered.remove(e_prime)
                
                result = backtrack(new_uncovered, used_cliques)
                if result is not None:
                    return result
                
                solution.pop()
                used_cliques.remove(i)
            
            return None
        
        return backtrack(edge_set, set())

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        edges = instance(n, k, p, seed)
        cliques = get_maximal_cliques(n, k, edges)
        
        sol_indices = solve_exact_cover(edges, cliques, time_limit=40)
        
        if sol_indices is None:
            # Fallback: Greedy Set Cover (not a partition, but we need a valid partition)
            # If backtracking fails, we might return a suboptimal valid partition found by a different method
            # Or just return the seed result? No, we want to improve.
            # Let's try a simple local search on the greedy set cover?
            # For now, if it fails, we return an empty list or something valid?
            # The task says "Any broken rule makes the whole result invalid".
            # So we must return a valid partition.
            # The seed program returns a valid partition.
            # If we can't find a better one quickly, we should return the seed one.
            res = [list(e) for e in edges]
        else:
            res = [list(cliques[i]) for i in sol_indices]
        
        results.append(res)
        
    return results
