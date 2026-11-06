import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    n = 256
    max_sum = 2 * n  # 512

    # Perfect squares in [2, 512]
    squares = set()
    i = 1
    while i * i <= max_sum:
        if i * i >= 2:
            squares.add(i * i)
        i += 1

    # Diagonal constraint: 2a not in squares
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    num_cand = len(candidates)

    # Build conflict graph: conflict[a] = set of b in candidates where a+b is a square
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    # Matching-aware upper bound:
    # The conflict graph is a union of |squares| matchings (one per square sum s: edges {a,b} where a+b=s).
    # For a set S of remaining vertices, we can compute a tighter bound by:
    # 1. For each square s, the edges with sum s form a matching. In any independent set,
    #    we can pick at most ceil(|S_s|/2) from vertices in S that are endpoints of s-edges.
    # 2. But matchings overlap on vertices, so we use a fractional packing bound:
    #    alpha(G[S]) <= |S| - max over packings of matchings of (sum of matching sizes)
    #    More practically: use greedy coloring on the induced subgraph, but also
    #    compute a bound based on the maximum degree in each matching.
    #
    # A simpler tight bound: since max degree <= len(squares) ~ 22,
    # we use: alpha <= |S| * (1 - 1/(d_max+1)) as a loose bound,
    # but better: use the fact that in a graph that's a union of k matchings,
    # we can find a clique cover or use the Lovasz theta bound approximation.
    #
    # Practical approach: compute a greedy coloring bound AND a matching-packing bound.
    # For matching packing: greedily find disjoint matchings in G[S], the number of
    # vertices "covered" by these matchings gives a lower bound on vertices that must
    # be excluded, so alpha <= |S| - (number of edges in max matching packing) ... 
    # Actually, each edge in a matching forces us to exclude at least one endpoint.
    # If we find a matching M in G[S], then alpha(G[S]) <= |S| - |M| is NOT correct.
    # Correct: alpha(G[S]) <= |S| - |M| where M is a matching is wrong.
    # Actually: if M is a matching of size m in G[S], then at most |S| - m vertices
    # can be in an independent set? No, that's not right either.
    # The correct statement: alpha(G) >= |V| - |M| for any matching M is false.
    # Let me use a different approach: for each vertex v in S, deg_S(v) <= k where k = len(squares).
    # By Turan's theorem or greedy: alpha >= |S|/(k+1), but we want upper bound.
    # Upper bound via greedy coloring is the most practical.
    # Let me add: for each connected component, compute a tighter bound.

    def greedy_color_bound(nodes, adj):
        """Compute upper bound via greedy coloring (number of colors = upper bound on alpha)."""
        nodes = set(nodes)
        if not nodes:
            return 0
        uncolored = set(nodes)
        color_count = 0
        while uncolored:
            # Pick node with smallest degree in uncolored
            best_node = None
            best_deg = float('inf')
            for node in uncolored:
                deg = len(adj[node] & uncolored)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
            uncolored.discard(best_node)
            for nb in adj[best_node] & uncolored:
                uncolored.discard(nb)
            color_count += 1
        return color_count

    def tight_upper_bound(nodes, adj):
        """Compute upper bound using component-wise greedy coloring."""
        nodes = set(nodes)
        if not nodes:
            return 0
        
        # Find connected components
        visited = set()
        total_ub = 0
        for start in nodes:
            if start in visited:
                continue
            component = set()
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                component.add(v)
                for nb in adj[v]:
                    if nb in nodes and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            total_ub += greedy_color_bound(component, adj)
        
        return total_ub

    # Branch and bound for maximum independent set
    def branch_and_bound(nodes, current_set, best_size_ref):
        if time.time() > deadline - 3.0:
            return current_set, best_size_ref[0]

        nodes = set(nodes)
        if not nodes:
            if len(current_set) > best_size_ref[0]:
                best_size_ref[0] = len(current_set)
            return current_set, best_size_ref[0]

        ub = len(current_set) + tight_upper_bound(nodes, conflict)
        if ub <= best_size_ref[0]:
            return current_set, best_size_ref[0]

        # Pick a vertex to branch on: highest degree in remaining graph
        best_v = None
        best_deg = -1
        for v in nodes:
            deg = len(conflict[v] & nodes)
            if deg > best_deg:
                best_deg = deg
                best_v = v

        if best_v is None:
            return current_set, best_size_ref[0]

        # Branch 1: include best_v
        new_nodes = nodes - {best_v} - conflict[best_v]
        new_set = current_set | {best_v}
        result_set, result_size = branch_and_bound(new_nodes, new_set, best_size_ref)
        best_size_ref[0] = max(best_size_ref[0], result_size)

        # Branch 2: exclude best_v
        nodes.discard(best_v)
        result_set, result_size = branch_and_bound(nodes, current_set, best_size_ref)
        best_size_ref[0] = max(best_size_ref[0], result_size)

        return result_set, best_size_ref[0]

    # Greedy initial solution
    def greedy_solution(pool):
        pool = set(pool)
        selected = set()
        while pool:
            min_deg_node = None
            min_deg = float('inf')
            for node in sorted(pool):
                deg = len(conflict[node] & pool)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = node
                    if min_deg == 0:
                        break
            if min_deg_node is None:
                break
            selected.add(min_deg_node)
            pool.discard(min_deg_node)
            pool.difference_update(conflict[min_deg_node])
        return selected

    # Try to add more elements
    def try_add(solution):
        sol_set = set(solution)
        changed = True
        while changed:
            changed = False
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    sol_set.add(a)
                    changed = True
        return sol_set

    # Local search with add, 1-swap, 1-for-2 swap, and 2-swap moves
    def local_search(solution, max_iters=5000):
        sol_set = set(solution)
        best_set = set(sol_set)
        best_size = len(sol_set)

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            improved = False

            # Try to add
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    sol_set.add(a)
                    improved = True
                    if len(sol_set) > best_size:
                        best_size = len(sol_set)
                        best_set = set(sol_set)
                    break

            if improved:
                continue

            # Try 1-swap: remove one, add one or more
            for remove_candidate in list(sol_set):
                temp_set = sol_set - {remove_candidate}
                for a in candidates:
                    if a not in temp_set and a != remove_candidate and not (conflict[a] & temp_set):
                        temp_set.add(a)

                if len(temp_set) > len(sol_set):
                    sol_set = temp_set
                    if len(sol_set) > best_size:
                        best_size = len(sol_set)
                        best_set = set(sol_set)
                    improved = True
                    break

            if improved:
                continue

            # Try 1-for-2 swap (from inspiration): remove one u, add two v1, v2
            # that were blocked only by u
            found_1for2 = False
            for u in list(sol_set):
                candidates_for_u = []
                for v in candidates:
                    if v in sol_set:
                        continue
                    if u not in conflict[v]:
                        continue
                    common = conflict[v] & sol_set
                    if len(common) == 1 and u in common:
                        candidates_for_u.append(v)

                if len(candidates_for_u) >= 2:
                    # Find two non-adjacent candidates
                    for i in range(len(candidates_for_u)):
                        if found_1for2:
                            break
                        for j in range(i + 1, len(candidates_for_u)):
                            if candidates_for_u[i] not in conflict[candidates_for_u[j]]:
                                sol_set.remove(u)
                                sol_set.add(candidates_for_u[i])
                                sol_set.add(candidates_for_u[j])
                                improved = True
                                found_1for2 = True
                                if len(sol_set) > best_size:
                                    best_size = len(sol_set)
                                    best_set = set(sol_set)
                                break
                    if found_1for2:
                        break

            if improved:
                continue

            # Try 2-swap: remove two, add more
            sol_list = list(sol_set)
            found_2swap = False
            for i in range(len(sol_list)):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, len(sol_list)):
                    temp_set = sol_set - {sol_list[i], sol_list[j]}
                    for a in candidates:
                        if a not in temp_set and a != sol_list[i] and a != sol_list[j] and not (conflict[a] & temp_set):
                            temp_set.add(a)

                    if len(temp_set) > len(sol_set):
                        sol_set = temp_set
                        if len(sol_set) > best_size:
                            best_size = len(sol_set)
                            best_set = set(sol_set)
                        improved = True
                        found_2swap = True
                        break
                if found_2swap:
                    break

            if not improved:
                # Perturbation: remove 1-3 random vertices and refill
                if time.time() > deadline - 5.0:
                    break
                k_remove = random.randint(1, 3)
                to_remove = random.sample(list(sol_set), min(k_remove, len(sol_set)))
                for v in to_remove:
                    sol_set.discard(v)

                # Refill greedily
                refill_improved = True
                while refill_improved:
                    refill_improved = False
                    for a in candidates:
                        if a not in sol_set and not (conflict[a] & sol_set):
                            sol_set.add(a)
                            refill_improved = True
                            break

                if len(sol_set) > best_size:
                    best_size = len(sol_set)
                    best_set = set(sol_set)

        return best_set

    # Start with greedy
    best_solution = greedy_solution(candidates)
    best_solution = try_add(best_solution)
    best_size = len(best_solution)

    # Randomized restarts with local search
    for seed in range(300):
        if time.time() > deadline - 15.0:
            break
        random.seed(seed)
        pool = set(candidates)
        selected = set()
        while pool:
            degrees = {}
            for node in pool:
                degrees[node] = len(conflict[node] & pool)
            min_deg = min(degrees.values())
            min_nodes = [node for node, deg in degrees.items() if deg == min_deg]
            node = random.choice(min_nodes)
            selected.add(node)
            pool.discard(node)
            pool.difference_update(conflict[node])

        selected = try_add(selected)
        selected = local_search(selected, max_iters=3000)
        if len(selected) > best_size:
            best_size = len(selected)
            best_solution = selected

    # Local search on best solution
    best_solution = local_search(best_solution, max_iters=10000)
    best_size = len(best_solution)

    # Try B&B from a reduced problem with remaining time
    remaining_time = deadline - 8.0 - (time.time() - start_time)
    if remaining_time > 5.0:
        bbb_start = time.time()

        def bbb(nodes, current_set, best_size_ref):
            if time.time() - bbb_start > remaining_time - 2.0:
                return current_set, best_size_ref[0]

            nodes = set(nodes)
            if not nodes:
                if len(current_set) > best_size_ref[0]:
                    best_size_ref[0] = len(current_set)
                return current_set, best_size_ref[0]

            ub = len(current_set) + tight_upper_bound(nodes, conflict)
            if ub <= best_size_ref[0]:
                return current_set, best_size_ref[0]

            # Pick vertex with highest degree
            best_v = None
            best_deg = -1
            for v in nodes:
                deg = len(conflict[v] & nodes)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_v is None:
                return current_set, best_size_ref[0]

            # Include best_v
            new_nodes = nodes - {best_v} - conflict[best_v]
            new_set = current_set | {best_v}
            rs, rsize = bbb(new_nodes, new_set, best_size_ref)
            best_size_ref[0] = max(best_size_ref[0], rsize)

            # Exclude best_v
            nodes.discard(best_v)
            rs, rsize = bbb(nodes, current_set, best_size_ref)
            best_size_ref[0] = max(best_size_ref[0], rsize)

            return rs, best_size_ref[0]

        best_size_ref = [best_size]
        bbb_solution, bbb_size = bbb(set(candidates), set(), best_size_ref)
        if bbb_size > best_size:
            best_solution = bbb_solution
            best_size = bbb_size

    # Final local search
    best_solution = local_search(best_solution, max_iters=15000)
    best_size = len(best_solution)

    # Final verification
    best_set = set(best_solution)
    for a in best_set:
        if 2 * a in squares:
            return []
    for i, a in enumerate(best_set):
        for b in best_set:
            if a < b and a + b in squares:
                return []

    return list(best_set)
