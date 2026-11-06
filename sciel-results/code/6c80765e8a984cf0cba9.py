"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256
    # Perfect squares up to 512
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1

    # Build adjacency: conflict[a] = set of b where a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)

    # Diagonal check: 2a must not be a perfect square
    valid_nodes = set(range(1, N + 1))
    for a in range(1, N + 1):
        if 2 * a in squares:
            valid_nodes.discard(a)

    def is_conflict(a, b):
        return b in conflict[a]

    def greedy_is(nodes, seed=0):
        rng = random.Random(seed)
        node_list = list(nodes)
        # Compute degree within the set
        degree = {}
        for a in node_list:
            deg = 0
            for b in conflict[a]:
                if b in nodes and b >= a:
                    deg += 1
            degree[a] = deg
        # Sort by degree ascending with some randomness
        node_list.sort(key=lambda x: (degree[x], rng.random()))
        chosen = set()
        for a in node_list:
            can_add = True
            for b in chosen:
                if is_conflict(a, b):
                    can_add = False
                    break
            if can_add:
                chosen.add(a)
        return chosen

    # Try multiple greedy restarts with different seeds
    best_set = set()
    best_size = 0
    for seed in range(50):
        if time.time() - start_time > deadline - 5:
            break
        s = greedy_is(valid_nodes, seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local search: try to improve by swapping
    improved = True
    while improved and (time.time() - start_time) < deadline - 2:
        improved = False
        unselected = valid_nodes - best_set

        # Try adding each unselected node
        best_gain = 0
        best_swap = None
        for a in unselected:
            to_remove = set()
            for b in best_set:
                if is_conflict(a, b):
                    to_remove.add(b)
            gain = 1 - len(to_remove)
            if gain > best_gain:
                best_gain = gain
                best_swap = (a, to_remove)

        if best_swap is not None and best_gain > 0:
            a, to_remove = best_swap
            best_set = (best_set - to_remove) | {a}
            best_size = len(best_set)
            improved = True
        else:
            # Try 1-for-1 swaps: remove one, add one
            for b in best_set:
                if time.time() - start_time > deadline - 2:
                    break
                for a in unselected:
                    if is_conflict(a, b):
                        new_set = best_set - {b}
                        can_add_a = True
                        for c in new_set:
                            if is_conflict(a, c):
                                can_add_a = False
                                break
                        if can_add_a:
                            new_best = new_set | {a}
                            if len(new_best) > best_size:
                                best_set = new_best
                                best_size = len(best_set)
                                improved = True
                                break
                if improved:
                    break

            # Try 2-for-1: remove two, add one
            if not improved and (time.time() - start_time) < deadline - 5:
                best_list = list(best_set)
                for i in range(len(best_list)):
                    if time.time() - start_time > deadline - 5:
                        break
                    for j in range(i + 1, len(best_list)):
                        if time.time() - start_time > deadline - 5:
                            break
                        b1, b2 = best_list[i], best_list[j]
                        remove_set = {b1, b2}
                        for a in unselected:
                            if is_conflict(a, b1) and is_conflict(a, b2):
                                new_set = best_set - remove_set
                                can_add_a = True
                                for c in new_set:
                                    if is_conflict(a, c):
                                        can_add_a = False
                                        break
                                if can_add_a:
                                    new_best = new_set | {a}
                                    if len(new_best) > best_size:
                                        best_set = new_best
                                        best_size = len(best_set)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break

    return list(best_set)
# EVOLVE-BLOCK-END
