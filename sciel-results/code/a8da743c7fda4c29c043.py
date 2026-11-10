import time
from math import lcm, gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit

    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)

    # Precompute incompatibility: incompatible[a][b] = True if lcm(a,b) <= 256
    incompatible = {x: set() for x in candidates}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            if lcm(a, b) <= N:
                incompatible[a].add(b)
                incompatible[b].add(a)

    # Precompute reciprocal weights
    weights = {x: Fraction(1, x) for x in candidates}

    def can_add(x, selected_set):
        """Check if x is compatible with all in selected_set."""
        for y in selected_set:
            if y in incompatible[x]:
                return False
        return True

    def total_recip(selected):
        """Exact reciprocal sum."""
        return sum(weights[x] for x in selected)

    def greedy_construct(order):
        """Greedy construction following a given order."""
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected

    def local_search(current_set, max_iters=200):
        """Local search with add and swap moves."""
        current_set = set(current_set)
        current_score = total_recip(current_set)
        improved = True
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() - start_time > timeout - 5:
                break

            # Try adding a single element
            for x in candidates:
                if x not in current_set and can_add(x, current_set):
                    new_set = current_set | {x}
                    new_score = total_recip(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try 1-for-1 swaps
            if not improved:
                for x in list(current_set):
                    for y in candidates:
                        if y not in current_set:
                            test_set = current_set - {x}
                            if can_add(y, test_set):
                                new_set = test_set | {y}
                                new_score = total_recip(new_set)
                                if new_score > current_score:
                                    current_set = new_set
                                    current_score = new_score
                                    improved = True
                                    break
                    if improved:
                        break

            # Try 2-for-1 swaps (remove two, add one)
            if not improved:
                for i, x in enumerate(list(current_set)):
                    for y in list(current_set)[i+1:]:
                        test_set = current_set - {x, y}
                        for z in candidates:
                            if z not in current_set and can_add(z, test_set):
                                new_set = test_set | {z}
                                new_score = total_recip(new_set)
                                if new_score > current_score:
                                    current_set = new_set
                                    current_score = new_score
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

            # Try 1-for-2 swaps (remove one, add two)
            if not improved:
                for x in list(current_set):
                    test_set = current_set - {x}
                    # Try to add two elements
                    for y in candidates:
                        if y not in current_set and can_add(y, test_set):
                            test_set2 = test_set | {y}
                            for z in candidates:
                                if z not in current_set and z != y and can_add(z, test_set2):
                                    new_set = test_set2 | {z}
                                    new_score = total_recip(new_set)
                                    if new_score > current_score:
                                        current_set = new_set
                                        current_score = new_score
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break

        return current_set, current_score

    # Multiple greedy starting orders
    best_set = set()
    best_score = Fraction(0)

    # Order 1: ascending (smallest first = largest reciprocal)
    order1 = sorted(candidates, key=lambda x: x)
    s1 = greedy_construct(order1)
    sc1 = total_recip(s1)
    if sc1 > best_score:
        best_score = sc1
        best_set = set(s1)

    # Order 2: descending (largest first)
    order2 = sorted(candidates, key=lambda x: -x)
    s2 = greedy_construct(order2)
    sc2 = total_recip(s2)
    if sc2 > best_score:
        best_score = sc2
        best_set = set(s2)

    # Order 3: by conflict count (least conflicting first)
    conflict_counts = {x: len(incompatible[x]) for x in candidates}
    order3 = sorted(candidates, key=lambda x: (conflict_counts[x], x))
    s3 = greedy_construct(order3)
    sc3 = total_recip(s3)
    if sc3 > best_score:
        best_score = sc3
        best_set = set(s3)

    # Order 4: random shuffle (multiple times if time permits)
    rng = random.Random(42)
    for trial in range(5):
        if time.time() - start_time > timeout - 20:
            break
        order4 = candidates[:]
        rng.shuffle(order4)
        s4 = greedy_construct(order4)
        sc4 = total_recip(s4)
        if sc4 > best_score:
            best_score = sc4
            best_set = set(s4)

    # Local search from best greedy solution
    if time.time() - start_time < timeout - 10:
        best_set, best_score = local_search(best_set, max_iters=300)

    # Local search from other starting points if time permits
    if time.time() - start_time < timeout - 15:
        for start_set in [s1, s2, s3]:
            if start_set == best_set:
                continue
            ls_set, ls_score = local_search(start_set, max_iters=100)
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set
            if time.time() - start_time > timeout - 10:
                break

    return list(best_set)
