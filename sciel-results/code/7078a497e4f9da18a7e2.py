import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 78.0  # safety margin before 90s kill

    n = 256
    max_sum = 2 * n  # 512

    # Perfect squares in [2, 512]
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2a not a square
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)

    # Build conflict graph: conflict[a] = set of b in candidates such that a+b in squares
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    def try_add(sol):
        """Greedy fill: add any candidate that has no conflict with sol."""
        sol_set = set(sol)
        improved = list(sol)
        changed = True
        while changed:
            changed = False
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    improved.append(a)
                    sol_set.add(a)
                    changed = True
        return improved

    def greedy_min_degree(pool):
        pool = set(pool)
        selected = []
        while pool:
            best_node = None
            best_deg = float('inf')
            for node in pool:
                deg = len(conflict[node] & pool)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
                    if deg == 0:
                        break
            if best_node is None:
                break
            selected.append(best_node)
            pool.discard(best_node)
            pool.difference_update(conflict[best_node])
        return selected

    def greedy_shuffle(pool, rng):
        """Randomized greedy: process in random order, add if no conflict."""
        pool = list(pool)
        rng.shuffle(pool)
        selected = []
        sel_set = set()
        for node in pool:
            if not (conflict[node] & sel_set):
                selected.append(node)
                sel_set.add(node)
        return selected

    def local_search_swap(solution, max_iters=2000):
        """1-swap local search: try removing one element and adding up to two.
        Accept any move that increases cardinality."""
        sol_set = set(solution)
        best = list(solution)
        best_size = len(best)
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 3.0:
                break
            improved = False
            # First: pure add (no removal)
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    best.append(a)
                    sol_set.add(a)
                    best_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap: remove x, add y (and maybe z)
            found = False
            for x in list(sol_set):
                # Candidates we could add after removing x
                # A node a is addable if a not in sol_set and not (conflict[a] & (sol_set - {x}))
                # and a != x
                for y in candidates:
                    if y in sol_set or y == x:
                        continue
                    if conflict[y] & (sol_set - {x}):
                        continue
                    # Try adding y
                    new_size = best_size  # remove x, add y => same size
                    # Try also adding z
                    for z in candidates:
                        if z in sol_set or z == x or z == y:
                            continue
                        if conflict[z] & (sol_set - {x}) or z in conflict[y]:
                            continue
                        # remove x, add y and z => size +1
                        new_sol = list(sol_set - {x}) + [y, z]
                        # verify validity quickly
                        new_set = set(new_sol)
                        ok = True
                        for a in new_set:
                            if 2 * a in squares:
                                ok = False
                                break
                        if ok:
                            for a in new_set:
                                for b in conflict[a]:
                                    if b in new_set:
                                        ok = False
                                        break
                                if not ok:
                                    break
                        if ok:
                            best = new_sol
                            sol_set = new_set
                            best_size = len(best)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                # No improving swap found; stop
                break
        return best

    # Phase 1: deterministic min-degree greedy + fill
    best_solution = greedy_min_degree(candidates)
    best_solution = try_add(best_solution)
    best_solution = local_search_swap(best_solution)
    best_size = len(best_solution)

    # Phase 2: many randomized restarts (shuffle greedy + fill + swap)
    rng = random.Random(12345)
    restart = 0
    while time.time() - start_time < deadline - 5.0:
        restart += 1
        # Mix strategies
        if restart % 3 == 0:
            # Min-degree with random tie-breaking
            pool = set(candidates)
            selected = []
            while pool:
                degrees = {}
                for node in pool:
                    degrees[node] = len(conflict[node] & pool)
                min_deg = min(degrees.values())
                min_nodes = [node for node, deg in degrees.items() if deg == min_deg]
                node = rng.choice(min_nodes)
                selected.append(node)
                pool.discard(node)
                pool.difference_update(conflict[node])
            cur = try_add(selected)
        else:
            cur = try_add(greedy_shuffle(candidates, rng))

        # Local search swap
        cur = local_search_swap(cur, max_iters=1500)
        if len(cur) > best_size:
            best_size = len(cur)
            best_solution = cur

    # Final verification
    best_set = set(best_solution)
    for a in best_solution:
        if 2 * a in squares:
            return [1]
    for i, a in enumerate(best_solution):
        for b in best_solution[i + 1:]:
            if a + b in squares:
                return [1]

    if not best_solution:
        return [1]
    return best_solution
