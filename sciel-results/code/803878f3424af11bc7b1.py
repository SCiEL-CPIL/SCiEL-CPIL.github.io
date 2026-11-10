# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    deadline = time.time() + 585  # Safety margin before 600s limit

    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Precompute for each element, which other elements are forbidden with it
    forbidden_with = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            forbidden_with[i].add(j1)
            forbidden_with[i].add(j2)

    # Also build a bitmask representation for faster operations
    # forbidden_mask[i] is a 205-bit integer where bit j is 1 if j is forbidden with i
    forbidden_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in forbidden_with[i]:
            mask |= (1 << j)
        forbidden_mask[i] = mask

    def greedy(order):
        """Greedy: process candidates in given order, add if compatible with all selected."""
        selected = []
        selected_set = set()
        for candidate in order:
            if candidate in selected_set:
                continue
            ok = True
            for s in selected:
                if candidate in forbidden_with[s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected

    def greedy_min_degree(seed_val):
        """
        Greedy independent set construction using minimum-degree heuristic.
        At each step, pick the vertex with minimum degree in the remaining candidate set,
        with random tie-breaking controlled by seed_val.
        """
        rng = random.Random(seed_val)
        candidates = set(range(m))
        selected = set()

        while candidates:
            min_deg = m + 1
            min_nodes = []
            for v in candidates:
                deg = len(forbidden_with[v] & candidates)
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)

            v = rng.choice(min_nodes)
            selected.add(v)
            candidates.discard(v)
            for u in forbidden_with[v]:
                if u in candidates:
                    candidates.discard(u)

        return selected

    def greedy_max_saturation(seed_val):
        """
        Greedy: at each step, pick the vertex that is compatible with the most
        remaining candidates (max saturation), breaking ties randomly.
        """
        rng = random.Random(seed_val)
        candidates = set(range(m))
        selected = set()

        while candidates:
            max_compat = -1
            max_nodes = []
            for v in candidates:
                # Count how many candidates are compatible with v
                compat_count = sum(1 for u in candidates if u != v and u not in forbidden_with[v])
                if compat_count > max_compat:
                    max_compat = compat_count
                    max_nodes = [v]
                elif compat_count == max_compat:
                    max_nodes.append(v)

            v = rng.choice(max_nodes)
            selected.add(v)
            candidates.discard(v)
            for u in forbidden_with[v]:
                if u in candidates:
                    candidates.discard(u)

        return selected

    def local_search_improve(initial_set, seed_val, time_limit=30):
        """
        Improve an initial independent set using local search:
        1. Try to add any vertex that doesn't conflict.
        2. Try 1-1 swaps to escape local optima, then re-add.
        3. Try 2-1 swaps (remove two, add three) for larger improvements.
        """
        rng = random.Random(seed_val)
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        start_time = time.time()

        def try_add_to(s):
            """Try to greedily add vertices to set s. Returns (new_set, improved)."""
            improved = True
            while improved:
                improved = False
                for v in range(m):
                    if v in s:
                        continue
                    can_add = True
                    for u in s:
                        if v in forbidden_with[u]:
                            can_add = False
                            break
                    if can_add:
                        s.add(v)
                        improved = True
                        break
            return s

        # Phase 1: Greedy addition
        current = try_add_to(current)
        if len(current) > best_size:
            best = set(current)
            best_size = len(current)

        # Phase 2: Random 1-1 swaps with re-addition
        for iteration in range(200):
            if time.time() - start_time > time_limit:
                break

            if not current:
                break

            # Pick a random vertex to remove
            u = rng.choice(list(current))

            # Find candidates that can replace u
            new_set = set(current)
            new_set.discard(u)

            # Find all vertices compatible with new_set
            candidates_to_add = [v for v in range(m) if v not in new_set and all(v not in forbidden_with[x] for x in new_set)]

            if not candidates_to_add:
                continue

            v = rng.choice(candidates_to_add)
            new_set.add(v)

            # Try to add more
            new_set = try_add_to(new_set)

            if len(new_set) > len(current):
                current = new_set
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            else:
                # Accept with some probability to explore
                if rng.random() < 0.2:
                    current = new_set
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)

        # Phase 3: 2-1 swaps (remove two vertices, try to add three)
        for iteration in range(50):
            if time.time() - start_time > time_limit:
                break

            if len(current) < 2:
                break

            # Pick two random vertices to remove
            u1, u2 = rng.sample(list(current), 2)
            new_set = set(current)
            new_set.discard(u1)
            new_set.discard(u2)

            # Find all vertices compatible with new_set
            candidates_to_add = [v for v in range(m) if v not in new_set and all(v not in forbidden_with[x] for x in new_set)]

            if len(candidates_to_add) < 3:
                continue

            # Try to pick 3 compatible vertices from candidates
            # They must also be mutually compatible
            found = False
            for attempt in range(20):
                sample = rng.sample(candidates_to_add, min(3, len(candidates_to_add)))
                # Check mutual compatibility
                valid = True
                for i in range(len(sample)):
                    for j in range(i + 1, len(sample)):
                        if sample[i] in forbidden_with[sample[j]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_set.update(sample)
                    found = True
                    break

            if not found:
                continue

            # Try to add more
            new_set = try_add_to(new_set)

            if len(new_set) > len(current):
                current = new_set
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)

        return best

    def branch_and_bound(vertices, selected, best_size):
        """
        Branch and bound for maximum independent set.
        vertices: list of candidate vertices
        selected: set of currently selected vertices
        best_size: current best size found
        Returns: best independent set found
        """
        if not vertices:
            return set(selected) if len(selected) > best_size[0] else None

        # Upper bound: can we possibly beat best_size?
        if len(selected) + len(vertices) <= best_size[0]:
            return None

        # Greedy coloring upper bound (simplified: just use len(vertices))
        # For better bounds, we'd use a greedy coloring of the induced subgraph

        # Choose a vertex to branch on
        # Heuristic: pick vertex with max degree in remaining graph
        max_deg = -1
        pivot = vertices[0]
        for v in vertices:
            deg = len(forbidden_with[v] & set(vertices))
            if deg > max_deg:
                max_deg = deg
                pivot = v

        # Branch 1: include pivot
        new_selected = set(selected)
        new_selected.add(pivot)
        # Remove pivot and its neighbors from vertices
        neighbors = forbidden_with[pivot]
        new_vertices = [v for v in vertices if v != pivot and v not in neighbors]

        result = branch_and_bound(new_vertices, new_selected, best_size)
        if result is not None:
            return result

        # Branch 2: exclude pivot
        new_vertices = [v for v in vertices if v != pivot]
        result = branch_and_bound(new_vertices, selected, best_size)
        if result is not None:
            return result

        return None

    def greedy_with_pruning(order, time_limit=10):
        """
        Greedy with backtracking: at each step, try candidates in order,
        but if we get stuck, backtrack and try a different choice.
        """
        start_time = time.time()
        best = set()
        best_size = 0

        def backtrack(index, current_set, remaining):
            nonlocal best, best_size

            if time.time() - start_time > time_limit:
                return

            if not remaining:
                if len(current_set) > best_size:
                    best = set(current_set)
                    best_size = len(current_set)
                return

            if len(current_set) + len(remaining) <= best_size:
                return

            v = remaining[0]
            rest = remaining[1:]

            # Option 1: include v
            new_current = set(current_set)
            new_current.add(v)
            # Remove v and its neighbors from remaining
            neighbors = forbidden_with[v]
            new_remaining = [x for x in rest if x not in neighbors and x != v]
            backtrack(index + 1, new_current, new_remaining)

            # Option 2: exclude v
            backtrack(index + 1, current_set, rest)

        backtrack(0, set(), list(order))
        return best

    # Start with multiple greedy strategies
    best = set()
    best_size = 0

    # Deterministic greedy in natural order
    result = greedy(list(range(m)))
    best = set(result)
    best_size = len(best)

    # Try various orderings
    rng = random.Random(42)

    # Min-degree greedy
    for i in range(30):
        if time.time() > deadline:
            break
        result = greedy_min_degree(i)
        if len(result) > best_size:
            best = set(result)
            best_size = len(result)

    # Max saturation greedy
    for i in range(20):
        if time.time() > deadline:
            break
        result = greedy_max_saturation(i)
        if len(result) > best_size:
            best = set(result)
            best_size = len(result)

    # Random order greedy
    for i in range(50):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        result = greedy(order)
        if len(result) > best_size:
            best = set(result)
            best_size = len(result)

    # Apply local search to best
    if time.time() < deadline - 60:
        improved_best = local_search_improve(list(best), 42, time_limit=30)
        if len(improved_best) > best_size:
            best = set(improved_best)
            best_size = len(improved_best)

    # Continue with more random restarts and local search
    attempts = 0
    while time.time() < deadline - 10:
        if attempts % 3 == 0:
            result = greedy_min_degree(attempts)
        elif attempts % 3 == 1:
            order = list(range(m))
            rng.shuffle(order)
            result = greedy(order)
        else:
            result = greedy_max_saturation(attempts)

        # Apply local search if promising
        if len(result) >= best_size - 2:
            improved_result = local_search_improve(result, attempts, time_limit=15)
            if len(improved_result) > best_size:
                best = set(improved_result)
                best_size = len(improved_result)

        attempts += 1

    # Final: try branch and bound on a reduced problem if time permits
    if time.time() < deadline - 30 and best_size < 15:
        # Try to find a better solution using B&B on a subset
        # Use the complement of best as initial candidates
        complement = [v for v in range(m) if v not in best]
        if complement:
            # Shrink to a manageable size
            rng.shuffle(complement)
            subset = complement[:50]
            best_size_tracker = [best_size]
            result = branch_and_bound(subset, set(best), best_size_tracker)
            if result is not None and len(result) > best_size:
                best = set(result)
                best_size = len(best)

    return np.array(list(best), dtype=np.int64)


# EVOLVE-BLOCK-END
