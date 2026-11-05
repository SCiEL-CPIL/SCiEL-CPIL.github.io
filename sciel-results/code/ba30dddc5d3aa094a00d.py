import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def enumerate_maximal_cliques(adj, n):
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            
            PX = P | X
            if PX:
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_exact_set_cover(n, cliques):
        """
        Exact minimum set cover via branch and bound.
        Only use when number of cliques is small.
        """
        if not cliques:
            return []
        
        # Map each clique to an index
        m = len(cliques)
        
        # For each vertex, which cliques does it cover?
        vertex_covers = [0] * n  # bitmask of cliques covered
        for v in range(n):
            mask = 0
            v_bit = 1 << v
            for i, c in enumerate(cliques):
                if c & v_bit:
                    mask |= (1 << i)
            vertex_covers[v] = mask
        
        # Remove vertices that cover no cliques
        active_vertices = [v for v in range(n) if vertex_covers[v] != 0]
        
        if not active_vertices:
            return []
        
        # If some clique is covered by only one vertex, that vertex must be selected
        # Pre-process: forced vertices
        forced = []
        remaining_cliques_mask = (1 << m) - 1
        
        # Find cliques with unique cover
        for i in range(m):
            cliq_bit = 1 << i
            covering_vertices = [v for v in active_vertices if vertex_covers[v] & cliq_bit]
            if len(covering_vertices) == 1:
                forced.append(covering_vertices[0])
        
        # Remove forced vertices' coverage
        for v in forced:
            remaining_cliques_mask &= ~vertex_covers[v]
        
        if remaining_cliques_mask == 0:
            return sorted(set(forced))
        
        # Now solve the remaining problem
        # Only consider vertices that cover at least one remaining clique
        active_vertices = [v for v in active_vertices if v not in forced and (vertex_covers[v] & remaining_cliques_mask) != 0]
        
        if not active_vertices:
            return sorted(set(forced))
        
        best = [n + 1]  # best solution size
        best_solution = [None]
        
        # Order vertices by coverage (most covering first)
        active_vertices.sort(key=lambda v: (vertex_covers[v] & remaining_cliques_mask).bit_count(), reverse=True)
        
        def branch(selected_mask, covered_mask, selected_list, depth):
            if time.time() > deadline:
                return
            
            if covered_mask == remaining_cliques_mask:
                if len(selected_list) < best[0]:
                    best[0] = len(selected_list)
                    best_solution[0] = selected_list[:]
                return
            
            # Lower bound: number of uncovered cliques / max coverage of any remaining vertex
            uncovered = remaining_cliques_mask & ~covered_mask
            if uncovered == 0:
                return
            
            # Find max coverage among remaining vertices
            max_cov = 0
            for v in active_vertices:
                if selected_mask & (1 << v):
                    continue
                cov = (vertex_covers[v] & uncovered).bit_count()
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return
            
            lb = (uncovered.bit_count() + max_cov - 1) // max_cov
            if len(selected_list) + lb >= best[0]:
                return
            
            # Branch: try each uncovered vertex
            # Pick a clique with fewest covering vertices to branch on
            min_count = n + 1
            min_clique = -1
            temp = uncovered
            while temp:
                lsb = temp & -temp
                i = lsb.bit_length() - 1
                count = 0
                for v in active_vertices:
                    if (selected_mask & (1 << v)) == 0 and (vertex_covers[v] & (1 << i)):
                        count += 1
                if count < min_count:
                    min_count = count
                    min_clique = i
                temp ^= lsb
            
            if min_clique == -1:
                return
            
            # Branch on vertices that cover min_clique
            for v in active_vertices:
                if selected_mask & (1 << v):
                    continue
                if not (vertex_covers[v] & (1 << min_clique)):
                    continue
                
                new_selected_mask = selected_mask | (1 << v)
                new_covered_mask = covered_mask | (vertex_covers[v] & remaining_cliques_mask)
                selected_list.append(v)
                branch(new_selected_mask, new_covered_mask, selected_list, depth + 1)
                selected_list.pop()
        
        branch(0, 0, [], 0)
        
        if best_solution[0] is not None:
            return sorted(set(forced + best_solution[0]))
        else:
            # Fallback: greedy
            return greedy_set_cover(n, cliques, forced)

    def greedy_set_cover(n, cliques, forced=None):
        if forced is None:
            forced = []
        
        if not cliques:
            return sorted(set(forced))
        
        remaining = set(cliques)
        selected = list(forced)
        selected_set = set(forced)
        
        # Remove cliques already covered by forced
        for v in forced:
            v_bit = 1 << v
            to_remove = []
            for c in remaining:
                if c & v_bit:
                    to_remove.append(c)
            for c in to_remove:
                remaining.discard(c)
        
        if not remaining:
            return sorted(set(forced))
        
        while remaining and time.time() < deadline:
            best_vertex = -1
            max_hits = 0
            
            for v in range(n):
                if v in selected_set:
                    continue
                v_bit = 1 << v
                hits = 0
                for c in remaining:
                    if c & v_bit:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_vertex = v
            
            if best_vertex == -1 or max_hits == 0:
                break
            
            selected.append(best_vertex)
            selected_set.add(best_vertex)
            v_bit = 1 << best_vertex
            
            to_remove = []
            for c in remaining:
                if c & v_bit:
                    to_remove.append(c)
            for c in to_remove:
                remaining.discard(c)
        
        # Fallback for any remaining
        if remaining:
            for c in remaining:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if v not in selected_set:
                    selected.append(v)
                    selected_set.add(v)
        
        return sorted(selected)

    def local_search_improve(n, cliques, initial_solution):
        """
        Try to improve the solution by removing vertices or swapping.
        """
        if not cliques:
            return list(initial_solution)
        
        solution = list(initial_solution)
        solution_set = set(solution)
        
        # Check if solution is valid
        def is_valid(sol_set):
            for c in cliques:
                if not any((1 << v) & c for v in sol_set):
                    return False
            return True
        
        if not is_valid(solution_set):
            # Fix by adding missing vertices
            for c in cliques:
                if not any((1 << v) & c for v in solution_set):
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    solution.append(v)
                    solution_set.add(v)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing each vertex
            for i in range(len(solution)):
                v = solution[i]
                new_set = solution_set - {v}
                if is_valid(new_set):
                    solution.pop(i)
                    solution_set.discard(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove v, add w
            for i in range(len(solution)):
                v = solution[i]
                for w in range(n):
                    if w in solution_set:
                        continue
                    new_set = solution_set - {v} | {w}
                    if is_valid(new_set):
                        solution[i] = w
                        solution_set.discard(v)
                        solution_set.add(w)
                        improved = True
                        break
                if improved:
                    break
        
        return sorted(solution)

    def solve_instance(adj, n):
        cliques = enumerate_maximal_cliques(adj, n)
        cliques = [c for c in cliques if c.bit_count() >= 2]
        
        if not cliques:
            return []
        
        # Strategy based on number of cliques
        m = len(cliques)
        
        if m <= 30:
            # Use exact branch and bound
            result = solve_exact_set_cover(n, cliques)
            # Try to improve with local search
            result = local_search_improve(n, cliques, result)
            return result
        else:
            # Use greedy + local search
            result = greedy_set_cover(n, cliques)
            result = local_search_improve(n, cliques, result)
            return result

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            adj = random_graph(n, p, seed)
            results.append(list(range(n)))
            continue

        adj = random_graph(n, p, seed)
        transversal = solve_instance(adj, n)
        results.append(transversal)

    return results
