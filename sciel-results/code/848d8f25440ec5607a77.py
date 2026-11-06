import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140.0
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2

    # 1. Build edges
    edges = []
    edge_to_idx = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx

    # 2. Build squares (faces)
    squares = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(nverts):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ mask

                    e1 = edge_to_idx[(min(v1, v2), max(v1, v2))]
                    e2 = edge_to_idx[(min(v2, v4), max(v2, v4))]
                    e3 = edge_to_idx[(min(v4, v3), max(v4, v3))]
                    e4 = edge_to_idx[(min(v3, v1), max(v3, v1))]

                    squares.append([e1, e2, e3, e4])

    num_squares = len(squares)

    # 3. For each edge, find all squares containing it
    edge_to_squares = [[] for _ in range(total_edges)]
    for s_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_to_squares[e_idx].append(s_idx)

    # Precompute square edge sets for fast membership
    square_edges = [set(sq) for sq in squares]

    # 4. Fast can_add check using numpy-style counting
    def can_add_fast(e_idx, selected_set):
        for s_idx in edge_to_squares[e_idx]:
            sq = square_edges[s_idx]
            count = 0
            for e in sq:
                if e != e_idx and e in selected_set:
                    count += 1
                    if count >= 3:
                        return False
        return True

    def get_conflict_count(e_idx, selected_set):
        """Return number of squares that would be completed if e_idx is added."""
        conflicts = 0
        for s_idx in edge_to_squares[e_idx]:
            sq = square_edges[s_idx]
            count = 0
            for e in sq:
                if e != e_idx and e in selected_set:
                    count += 1
            if count >= 3:
                conflicts += 1
        return conflicts

    def greedy_init(seed=None):
        """Random greedy initialization."""
        if seed is not None:
            random.seed(seed)
        selected = set()
        order = list(range(total_edges))
        random.shuffle(order)
        for e_idx in order:
            if can_add_fast(e_idx, selected):
                selected.add(e_idx)
        return selected

    def local_search(selected, time_budget=5.0):
        """Local search: try to add edges, and do swaps when blocked."""
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            candidates = [e for e in range(total_edges) if e not in selected]
            random.shuffle(candidates)

            for e_idx in candidates:
                if can_add_fast(e_idx, selected):
                    selected.add(e_idx)
                    improved = True
                else:
                    # Try swaps: find a blocking square and try removing one of the other edges
                    for s_idx in edge_to_squares[e_idx]:
                        sq = square_edges[s_idx]
                        others = [e for e in sq if e != e_idx and e in selected]
                        if len(others) >= 3:
                            # Try removing each other edge
                            for f_idx in others:
                                selected.discard(f_idx)
                                if can_add_fast(e_idx, selected):
                                    selected.add(e_idx)
                                    improved = True
                                    break
                                else:
                                    selected.add(f_idx)
                            if improved:
                                break
                        if improved:
                            break

            # After a pass, try adding any newly addable edges
            for e_idx in range(total_edges):
                if e_idx not in selected and can_add_fast(e_idx, selected):
                    selected.add(e_idx)
                    improved = True

        return selected

    def hill_climb_2swap(selected, time_budget=5.0):
        """2-swap local search: remove 2 edges, add 1-2 edges if possible."""
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            selected_list = list(selected)
            random.shuffle(selected_list)

            for i, e1 in enumerate(selected_list):
                if time.time() > start_time + time_budget:
                    break
                for e2 in selected_list[i+1:]:
                    if time.time() > start_time + time_budget:
                        break
                    # Try removing both e1 and e2
                    new_selected = selected - {e1, e2}
                    # Try adding any unselected edge
                    for e3 in range(total_edges):
                        if e3 not in new_selected and can_add_fast(e3, new_selected):
                            new_selected.add(e3)
                            # Try adding one more
                            for e4 in range(total_edges):
                                if e4 not in new_selected and can_add_fast(e4, new_selected):
                                    new_selected.add(e4)
                            if len(new_selected) > len(selected):
                                selected = new_selected
                                improved = True
                                break
                            break
        return selected

    best_count = 0
    best_set = set()

    # Try multiple random seeds with varied strategies
    rng = random.Random(42)
    num_restarts = 80

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # Vary the initialization strategy
        strategy = restart % 4

        if strategy == 0:
            # Standard random greedy
            selected = greedy_init()
        elif strategy == 1:
            # Greedy with bias toward high-degree edges
            selected = set()
            order = list(range(total_edges))
            # Sort by degree in square graph (number of squares containing edge)
            order.sort(key=lambda e: len(edge_to_squares[e]), reverse=True)
            random.shuffle(order)
            for e_idx in order:
                if can_add_fast(e_idx, selected):
                    selected.add(e_idx)
        elif strategy == 2:
            # Start from a partial solution and complete
            selected = set()
            # Pick a random subset to start
            subset_size = total_edges // 4
            subset = random.sample(range(total_edges), subset_size)
            for e_idx in subset:
                if can_add_fast(e_idx, selected):
                    selected.add(e_idx)
            # Complete greedily
            for e_idx in range(total_edges):
                if e_idx not in selected and can_add_fast(e_idx, selected):
                    selected.add(e_idx)
        else:
            # Iterated local search: perturb a good solution
            if best_set:
                selected = set(best_set)
                # Remove k random edges
                k = random.randint(5, 15)
                to_remove = random.sample(list(selected), min(k, len(selected)))
                for e in to_remove:
                    selected.discard(e)
                # Re-add greedily
                for e_idx in range(total_edges):
                    if e_idx not in selected and can_add_fast(e_idx, selected):
                        selected.add(e_idx)
            else:
                selected = greedy_init()

        # Local search
        selected = local_search(selected, time_budget=1.5)

        count = len(selected)
        if count > best_count:
            best_count = count
            best_set = set(selected)

        # Occasionally do a more aggressive local search on the best
        if restart % 3 == 0 and count >= best_count - 3:
            selected = local_search(set(best_set), time_budget=2.0)
            if len(selected) > best_count:
                best_count = len(selected)
                best_set = set(selected)

    # Final polish on the best solution
    if time.time() < deadline - 5:
        best_set = local_search(best_set, time_budget=8.0)
        best_count = len(best_set)

    # Try 2-swap if time permits
    if time.time() < deadline - 3:
        best_set = hill_climb_2swap(best_set, time_budget=5.0)
        best_count = len(best_set)

    # Convert best_set to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = edges[e_idx]
        result.append([u, v])

    return result
