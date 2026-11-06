import time
from fractions import Fraction

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    def time_left():
        return deadline - time.time()

    # Precompute all nonconstant 3-APs as frozensets of 3 distinct elements
    aps = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > b and c <= N:
                aps.append((a, b, c))

    # For each element, list of AP indices it participates in
    elem_to_aps = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(aps):
        elem_to_aps[a].append(i)
        elem_to_aps[b].append(i)
        elem_to_aps[c].append(i)

    # Weights: Fraction(1, x)
    weights = [Fraction(0, 1)] + [Fraction(1, x) for x in range(1, N + 1)]

    # Sort elements by descending weight (ascending value) for B&B
    # Order: 1, 2, 3, ..., 256 (since 1/x is decreasing)
    order = list(range(1, N + 1))

    def greedy_construct():
        """Greedy: include x in ascending order if it doesn't create a 3-AP."""
        selected = set()
        for x in range(1, N + 1):
            ok = True
            for a in selected:
                b = 2 * x - a
                if b != a and b != x and b in selected:
                    ok = False
                    break
                b_end = 2 * a - x
                if b_end != a and b_end != x and b_end in selected:
                    ok = False
                    break
            if ok:
                selected.add(x)
        return selected

    def has_3ap_with_x(s, x):
        """Check if adding x to set s creates a 3-AP. s is a set, x not in s."""
        for a in s:
            b = 2 * x - a
            if b != a and b != x and b in s:
                return True
            b_end = 2 * a - x
            if b_end != a and b_end != x and b_end in s:
                return True
        return False

    def reciprocal_sum(s):
        total = Fraction(0, 1)
        for x in s:
            total += Fraction(1, x)
        return total

    # Phase 1: Greedy to get initial lower bound
    greedy_set = greedy_construct()
    best_score = reciprocal_sum(greedy_set)
    best_set = set(greedy_set)

    # Phase 2: Branch and Bound
    # We process elements in order (1, 2, 3, ..., 256).
    # At each step, we decide to include or exclude the next element.
    # Upper bound: current_score + sum of weights of all remaining elements
    #   that don't conflict with the current selection.
    # But a tighter bound: for each remaining element, check if it can be added.
    # Actually, the standard bound is: current_score + sum of all remaining weights.
    # That's too loose. Better: for each remaining element x, if it conflicts with
    # any selected element, it can't be added. So upper bound = current_score +
    # sum of weights of remaining elements that don't conflict.

    # Precompute: for each element, the set of elements that conflict with it
    # (i.e., form a 3-AP with it and one other element).
    # Actually, for B&B, we need: given current selected set S, which remaining
    # elements can still be added? An element x can be added if it doesn't form
    # a 3-AP with any two elements in S.

    # Let's use a different approach: process in order, maintain selected set.
    # Upper bound at node (i, S): score(S) + sum of 1/x for x in {i+1, ..., 256}
    #   such that x doesn't form a 3-AP with any pair in S.
    # This is O(|S| * (N-i)) per node, which is expensive.

    # Alternative: use a simpler bound. score(S) + sum of 1/x for x in {i+1,...,256}.
    # This is O(1) per node if we precompute suffix sums.
    # But it's very loose. Let's try it first and see if it's fast enough.

    # Precompute suffix sums of weights
    suffix_sum = [Fraction(0, 1)] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]

    # B&B state: (position, selected_set, current_score)
    # Use recursion with a time check at each node.

    import sys
    sys.setrecursionlimit(10000)

    nodes_visited = [0]

    def bb(pos, selected, score):
        """Branch and bound. pos: next element to consider (1-indexed).
        selected: set of selected elements so far.
        score: current sum of reciprocals.
        """
        if time.time() > deadline - 2.0:
            return

        nodes_visited[0] += 1
        if nodes_visited[0] % 10000 == 0 and time.time() > deadline - 2.0:
            return

        # Upper bound: current score + sum of all remaining weights
        ub = score + suffix_sum[pos]
        if ub <= best_score:
            return

        if pos > N:
            if score > best_score:
                nonlocal best_score, best_set
                best_score = score
                best_set = set(selected)
            return

        x = pos

        # Option 1: Include x if it doesn't create a 3-AP
        if not has_3ap_with_x(selected, x):
            new_selected = selected | {x}
            new_score = score + weights[x]
            # Tighter bound: check if new_score + suffix_sum[x+1] > best_score
            if new_score + suffix_sum[x + 1] > best_score:
                bb(x + 1, new_selected, new_score)

        # Option 2: Exclude x
        # Upper bound for this branch: score + suffix_sum[x+1]
        if score + suffix_sum[x + 1] > best_score:
            bb(x + 1, selected, score)

    # Run B&B
    bb(1, set(), Fraction(0, 1))

    # Phase 3: Local search to polish
    if time_left() > 5.0:
        current = set(best_set)
        current_score = best_score
        all_candidates = set(range(1, N + 1))

        while time_left() > 3.0:
            improved = False

            # Try adding elements
            to_add = sorted(all_candidates - current)
            for x in to_add:
                if time.time() >= deadline - 3.0:
                    break
                if not has_3ap_with_x(current, x):
                    current.add(x)
                    current_score += weights[x]
                    improved = True

            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current)
                continue

            # Try swaps
            to_remove = sorted(current, reverse=True)
            found_swap = False
            for rem in to_remove:
                if time.time() >= deadline - 3.0:
                    break

                best_add = None
                best_gain = Fraction(0, 1)
                for x in range(1, rem):
                    if time.time() >= deadline - 3.0:
                        break
                    if x in current:
                        continue
                    if not has_3ap_with_x(current - {rem}, x):
                        gain = weights[x] - weights[rem]
                        if gain > best_gain:
                            best_gain = gain
                            best_add = x

                if best_add is not None:
                    current.remove(rem)
                    current.add(best_add)
                    current_score += best_gain
                    found_swap = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current)
                    break

            if not found_swap:
                break

    return sorted(best_set)
