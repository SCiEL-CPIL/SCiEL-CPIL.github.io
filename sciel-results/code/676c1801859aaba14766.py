import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def bron_kerbosch(R, P, X, graph, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivoting optimization
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        pivot = None
        max_count = -1
        for u in P.union(X):
            count = len(P.intersection(graph.get(u, set())))
            if count > max_count:
                max_count = count
                pivot = u
        
        if pivot is None:
            # No pivot found, iterate over all P
            for v in list(P):
                bron_kerbosch(R | {v}, P & graph.get(v, set()), X & graph.get(v, set()), graph, cliques)
                P.remove(v)
                X.add(v)
        else:
            # Iterate over P \ N(pivot)
            for v in list(P.difference(graph.get(pivot, set()))):
                bron_kerbosch(R | {v}, P & graph.get(v, set()), X & graph.get(v, set()), graph, cliques)
                P.remove(v)
                X.add(v)

    def find_maximal_cliques(n, edges):
        # Build adjacency sets
        graph = {i: set() for i in range(n)}
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        cliques = []
        bron_kerbosch(frozenset(), set(range(n)), set(), graph, cliques)
        return [list(c) for c in cliques]

    def is_valid_cover(selected, cliques):
        selected_set = set(selected)
        for c in cliques:
            if not selected_set.intersection(c):
                return False
        return True

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        
        if not cliques:
            return []
        
        best_solution = None
        
        # Multiple random restarts for greedy phase to escape local optima
        # The greedy choice is deterministic if no ties, but we can randomize ties.
        # Also, the order of vertices in P during Bron-Kerbosch doesn't affect the set of cliques, 
        # but the greedy selection order does.
        
        max_restarts = 20
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            uncovered = [c for c in cliques]
            selected = []
            
            # Greedy Phase
            while uncovered:
                # Count coverage for each vertex
                counts = {v: 0 for v in range(n)}
                for c in uncovered:
                    for v in c:
                        counts[v] += 1
                
                # Find max count
                max_count = max(counts.values())
                if max_count == 0:
                    break # Should not happen if uncovered is not empty and cliques are valid
                
                candidates = [v for v in range(n) if counts[v] == max_count]
                # Randomize choice among candidates
                v = random.choice(candidates)
                
                selected.append(v)
                selected_set = set(selected)
                
                # Update uncovered
                new_uncovered = []
                for c in uncovered:
                    if not selected_set.intersection(c):
                        new_uncovered.append(c)
                uncovered = new_uncovered
            
            # Local Search Phase
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                
                # 1. Pruning: Remove redundant vertices
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    u = selected[i]
                    temp_selected = selected[:i] + selected[i+1:]
                    if is_valid_cover(temp_selected, cliques):
                        selected = temp_selected
                        improved = True
                        break # Restart loop as list changed
                
                if not improved:
                    # 2. Swapping: Try to replace a vertex
                    selected_set = set(selected)
                    for i in range(len(selected)):
                        if time.time() > deadline:
                            break
                        u = selected[i]
                        # Try replacing u with w not in selected
                        for w in range(n):
                            if w in selected_set:
                                continue
                            temp_selected = selected[:i] + [w] + selected[i+1:]
                            if is_valid_cover(temp_selected, cliques):
                                selected = temp_selected
                                selected_set = set(selected)
                                improved = True
                                break # Restart pruning/swapping
                        if improved:
                            break
            
            # Update best solution
            if best_solution is None or len(selected) < len(best_solution):
                best_solution = selected[:]
                
        return best_solution

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices involved in edges if we run out of time
            # But we should have finished. Just append empty or best effort.
            # Let's just append what we have or a trivial solution.
            edges = random_graph(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue
            
        sol = solve_instance(n, p, seed)
        out.append(sorted(sol))
        
    return out
