import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 90s kill

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build conflict sets (0-indexed internally)
    # conflict[i] = set of j (0-indexed) such that (i+1) + (j+1) is a square
    # Also exclude vertices where 2*(i+1) is a square (self-conflict)
    conflict = [set() for _ in range(N)]
    valid_vertices = []

    for i in range(N):
        a = i + 1
        if 2 * a in squares:
            continue  # Cannot be in set due to self-conflict
        valid_vertices.append(i)

    for i in valid_vertices:
        a = i + 1
        for j in valid_vertices:
            if j == i:
                continue
            b = j + 1
            if a + b in squares:
                conflict[i].add(j)

    num_valid = len(valid_vertices)
    valid_set = set(valid_vertices)

    def is_independent(S):
        """Check if S is an independent set."""
        for i in S:
            for j in S:
                if i != j and j in conflict[i]:
                    return False
        return True

    def greedy_construct(order):
        """Greedy: add vertices in given order if they don't conflict with current set."""
        selected = set()
        for v in order:
            if v not in valid_set:
                continue
            # Check if v conflicts with any selected vertex
            can_add = True
            for s in selected:
                if s in conflict[v]:
                    can_add = False
                    break
            if can_add:
                selected.add(v)
        return selected

    def local_improve(selected):
        """Try to add any non-conflicting valid vertex not yet in selected."""
        improved = True
        changed = True
        while changed:
            changed = False
            for v in valid_vertices:
                if v in selected:
                    continue
                can_add = True
                for s in selected:
                    if s in conflict[v]:
                        can_add = False
                        break
                if can_add:
                    selected.add(v)
                    changed = True
                    break
        return selected

    def local_search(selected, max_swaps=200):
        """Swap-based local search: try removing one and adding one or two others."""
        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            sel_list = list(selected)
            random.shuffle(sel_list)
            for remove_v in sel_list:
                if (time.time() - start_time) > DEADLINE:
                    break
                # Try removing remove_v and adding one or two others
                new_selected = set(selected)
                new_selected.discard(remove_v)

                # Try adding single vertex
                best_gain = 0
                best_add = None
                for add_v in valid_vertices:
                    if add_v in new_selected:
                        continue
                    can_add = True
                    for s in new_selected:
                        if s in conflict[add_v]:
                            can_add = False
                            break
                    if can_add:
                        best_gain = 1
                        best_add = add_v
                        break

                if best_gain > 0 and best_add is not None:
                    new_selected.add(best_add)
                    selected = new_selected
                    improved = True
                    break

                # Try adding two vertices
                for add_v1 in valid_vertices:
                    if add_v1 in new_selected:
                        continue
                    can_add1 = True
                    for s in new_selected:
                        if s in conflict[add_v1]:
                            can_add1 = False
                            break
                    if not can_add1:
                        continue
                    for add_v2 in valid_vertices:
                        if add_v2 in new_selected or add_v2 == add_v1:
                            continue
                        can_add2 = True
                        for s in new_selected:
                            if s in conflict[add_v2]:
                                can_add2 = False
                                break
                        if not can_add2:
                            continue
                        # Check add_v1 and add_v2 don't conflict
                        if add_v1 in conflict[add_v2]:
                            continue
                        # Also check they don't conflict with each other (already done above)
                        new_selected.add(add_v1)
                        new_selected.add(add_v2)
                        selected = new_selected
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
        return selected

    # Run multiple random restarts
    best_set = set()
    best_size = 0

    # First, deterministic greedy in natural order
    order = list(valid_vertices)
    sel = greedy_construct(order)
    sel = local_improve(sel)
    if len(sel) > best_size:
        best_set = set(sel)
        best_size = len(sel)

    # Random restarts
    num_restarts = 200
    for r in range(num_restarts):
        if (time.time() - start_time) > DEADLINE:
            break
        random.seed(r * 7919 + 13)
        order = list(valid_vertices)
        random.shuffle(order)
        sel = greedy_construct(order)
        sel = local_improve(sel)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = set(sel)

        # Local search on the best
        if len(sel) >= best_size - 2:
            sel = local_search(sel)
            if len(sel) > best_size:
                best_size = len(sel)
                best_set = set(sel)

    # Final local search on best
    best_set = local_search(best_set)

    # Convert to 1-indexed list
    result = sorted([v + 1 for v in best_set])
    return result if result else [1]
