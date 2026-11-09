import time
import random

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(R, P, X, graph, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        U = P.union(X)
        if U:
            pivot = max(U, key=lambda u: len(graph[u].intersection(P)))
        else:
            pivot = None
        
        if pivot is not None:
            neighbors_of_pivot = graph[pivot]
            candidates = P.difference(neighbors_of_pivot)
        else:
            candidates = P

        for v in list(candidates):
            bron_kerbosch(R | {v}, P & graph[v], X & graph[v], graph, cliques)
            P.remove(v)
            X.add(v)

    def find_maximal_cliques(n, edges):
        graph = {i: set() for i in range(n)}
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        active_vertices = {v for v in range(n) if len(graph[v]) > 0}
        
        cliques = []
        bron_kerbosch(set(), active_vertices, set(), graph, cliques)
        return cliques

    def greedy_hitting_set(cliques, n):
        if not cliques:
            return set()
        
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        uncovered = set(range(len(cliques)))
        S = set()
        
        while uncovered:
            best_v = -1
            max_covered = -1
            for v in range(n):
                if v in S:
                    continue
                covered = sum(1 for i in vertex_to_cliques[v] if i in uncovered)
                if covered > max_covered:
                    max_covered = covered
                    best_v = v
            if best_v == -1:
                break
            S.add(best_v)
            for i in vertex_to_cliques[best_v]:
                if i in uncovered:
                    uncovered.remove(i)
        
        return S

    def advanced_local_search(S, cliques, n, time_limit):
        start_time = time.time()
        
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        counts = [0] * len(cliques)
        for v in S:
            for i in vertex_to_cliques[v]:
                counts[i] += 1
                
        current_S = set(S)
        
        def remove_vertex(v):
            for i in vertex_to_cliques[v]:
                counts[i] -= 1
            current_S.remove(v)

        def add_vertex(v):
            for i in vertex_to_cliques[v]:
                counts[i] += 1
            current_S.add(v)

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Phase 1: Remove redundant vertices
            for v in list(current_S):
                redundant = True
                for i in vertex_to_cliques[v]:
                    if counts[i] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_vertex(v)
                    improved = True
                    break
            
            if improved:
                continue

            # Phase 2: 1-for-1 Swap
            S_list = list(current_S)
            for v in S_list:
                critical_cliques = [i for i in vertex_to_cliques[v] if counts[i] == 1]
                if not critical_cliques:
                    continue
                
                swap_found = False
                for u in range(n):
                    if u in current_S:
                        continue
                    valid = True
                    for i in critical_cliques:
                        if u not in cliques[i]:
                            valid = False
                            break
                    if valid:
                        remove_vertex(v)
                        add_vertex(u)
                        improved = True
                        swap_found = True
                        break
                if swap_found:
                    break
            
            if improved:
                continue

            # Phase 3: 2-for-1 Swap
            S_list = list(current_S)
            two_opt_found = False
            for i in range(len(S_list)):
                if two_opt_found:
                    break
                v1 = S_list[i]
                C1 = [c_idx for c_idx in vertex_to_cliques[v1] if counts[c_idx] == 1]
                if not C1:
                    continue
                
                for j in range(i + 1, len(S_list)):
                    v2 = S_list[j]
                    C2 = [c_idx for c_idx in vertex_to_cliques[v2] if counts[c_idx] == 1]
                    
                    C_union = set(C1)
                    C_union.update(C2)
                    
                    if not C_union:
                        remove_vertex(v1)
                        remove_vertex(v2)
                        improved = True
                        two_opt_found = True
                        break
                    
                    swap_found = False
                    for u in range(n):
                        if u in current_S:
                            continue
                        valid = True
                        for c_idx in C_union:
                            if u not in cliques[c_idx]:
                                valid = False
                                break
                        if valid:
                            remove_vertex(v1)
                            remove_vertex(v2)
                            add_vertex(u)
                            improved = True
                            swap_found = True
                            two_opt_found = True
                            break
                    if swap_found:
                        break
                        
            if not improved:
                break

        return current_S

    def verify_hitting_set(S, cliques):
        for c in cliques:
            if not (S & c):
                return False
        return True

    def repair_hitting_set(S, cliques, n, deadline):
        S = set(S)
        if verify_hitting_set(S, cliques):
            return S
        
        uncovered = [c for c in cliques if not (S & c)]
        
        while uncovered:
            if time.time() > deadline:
                break
            
            cover_counts = {v: 0 for v in range(n)}
            for c in uncovered:
                for v in c:
                    cover_counts[v] += 1
            
            best_v = max(range(n), key=lambda x: cover_counts[x])
            S.add(best_v)
            
            new_uncovered = []
            for c in uncovered:
                if best_v not in c:
                    new_uncovered.append(c)
            uncovered = new_uncovered
        
        return S

    def solve_instance(n, p, seed, deadline):
        edges = random_graph(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        
        if not cliques:
            return []
        
        initial_S = greedy_hitting_set(cliques, n)
        
        elapsed = time.time()
        remaining_time = deadline - elapsed
        
        search_time = min(10.0, max(1.0, remaining_time * 0.8))
        
        final_S = advanced_local_search(initial_S, cliques, n, time_limit=search_time)
        
        if not verify_hitting_set(final_S, cliques):
            final_S = repair_hitting_set(final_S, cliques, n, deadline)
            
        return sorted(final_S)

    results = []
    deadline = time.time() + 140
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            results.append(list(range(n)))
            continue
        
        result = solve_instance(n, p, seed, deadline)
        results.append(result)
    
    return results

# Ensure the function is named construct
# The code above defines construct.
