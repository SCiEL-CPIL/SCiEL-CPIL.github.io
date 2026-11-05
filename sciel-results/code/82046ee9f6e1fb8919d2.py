"""Optimized construction of 3-AP-free subset of [1,256] maximizing sum of reciprocals."""
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 256

    # Precompute: for each pair (a, b) with a < b, the third term c = 2*b - a
    # If c is in [1, n], then {a, b, c} forms a 3-AP.
    # We'll maintain a set of selected elements and track conflicts.

    # Strategy:
    # 1. Start with a greedy solution (add elements in order of 1/x, i.e., smallest x first)
    # 2. Local search: try removals, additions, and swaps to improve the reciprocal sum

    # Greedy initialization: add elements in increasing order of x (smallest reciprocals first? No, largest reciprocals first = smallest x)
    # Actually, to maximize sum of 1/x, we want small x. So try adding from 1 upward.
    # But we must avoid 3-APs.

    def has_3ap_with(selected_set, x):
        """Check if adding x to selected_set creates a 3-AP."""
        # For each pair (a, b) in selected_set with a < b:
        #   if 2*b - a == x, conflict
        #   if 2*a - b == x (i.e., b is middle, a is larger... wait a<b so 2a-b < a < b, so 2a-b < a, could be x if x < a)
        #   if a + b == 2*x, i.e., x is the middle term: a + b = 2x
        # More systematically: x forms a 3-AP with two existing elements if:
        #   Case 1: x is the largest: exists a, b in S with a < b and 2b - a = x
        #   Case 2: x is the middle: exists a, c in S with a < x < c and a + c = 2x
        #   Case 3: x is the smallest: exists b, c in S with b < c and 2b - c = x (i.e., c - b = b - x, so x = 2b - c)
        # But since we're checking against the full set, let's just check all pairs.

        sel = list(selected_set)
        m = len(sel)
        for i in range(m):
            a = sel[i]
            # x as third term: 2*b - a = x => b = (x + a) / 2
            b = (x + a) / 2
            if b == int(b) and int(b) in selected_set and int(b) > a:
                return True
            # x as middle: a + c = 2*x => c = 2*x - a
            c = 2 * x - a
            if c in selected_set and c > x:
                return True
            # x as first term: 2*a - b = x => b = 2*a - x
            b2 = 2 * a - x
            if b2 in selected_set and b2 > a:
                return True
        return False

    def has_3ap_fast(selected_set, x):
        """Faster check using set operations."""
        # x as largest term: need a, b in S, a < b, 2b - a = x => a = 2b - x
        # For each b in S where b > x/2 (so a = 2b-x > 0): check if 2b-x in S and 2b-x < b
        for b in selected_set:
            a = 2 * b - x
            if 0 < a < b and a in selected_set:
                return True
        # x as middle: need a, c in S, a < x < c, a + c = 2x
        for a in selected_set:
            if a < x:
                c = 2 * x - a
                if c > x and c in selected_set:
                    return True
        # x as smallest: need b, c in S, b < c, 2b - c = x => c = 2b - x
        for b in selected_set:
            c = 2 * b - x
            if b < c and c in selected_set:
                return True
        return False

    def reciprocal_sum(selected):
        return sum(1.0 / x for x in selected)

    # Greedy: try to add elements from 1 to 256
    selected = set()
    for x in range(1, n + 1):
        if not has_3ap_fast(selected, x):
            selected.add(x)

    current_sum = reciprocal_sum(selected)
    best_set = set(selected)
    best_sum = current_sum

    # Local search: try single removals and then try to add back different elements
    # Also try swaps: remove one, add another

    def try_add(selected_set, x):
        """Try to add x to selected_set, return new set if valid, else None."""
        if x in selected_set:
            return None
        if has_3ap_fast(selected_set, x):
            return None
        new_set = selected_set | {x}
        return new_set

    # Phase 1: Try removing each element and see if we can add multiple new ones
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for x in sorted(best_set):
            if time.time() > DEADLINE:
                break
            # Try removing x
            reduced = best_set - {x}
            reduced_sum = best_sum - 1.0 / x

            # Try adding each element not in reduced
            best_addition = None
            best_addition_sum = reduced_sum
            for y in range(1, n + 1):
                if y in reduced:
                    continue
                new_set = try_add(reduced, y)
                if new_set is not None:
                    s = reduced_sum + 1.0 / y
                    if s > best_addition_sum + 1e-15:
                        best_addition_sum = s
                        best_addition = new_set

            if best_addition is not None and best_addition_sum > best_sum + 1e-15:
                best_set = best_addition
                best_sum = best_addition_sum
                improved = True

        # Phase 2: Try simple additions to current best
        for y in range(1, n + 1):
            if y in best_set:
                continue
            new_set = try_add(best_set, y)
            if new_set is not None:
                s = best_sum + 1.0 / y
                if s > best_sum + 1e-15:
                    best_set = new_set
                    best_sum = s
                    improved = True

    # Phase 3: Try 2-for-1 swaps: remove two, add one or two
    if time.time() < DEADLINE:
        improved2 = True
        while improved2 and time.time() < DEADLINE:
            improved2 = False
            sel_list = sorted(best_set)
            for i in range(len(sel_list)):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, len(sel_list)):
                    if time.time() > DEADLINE:
                        break
                    x1, x2 = sel_list[i], sel_list[j]
                    reduced = best_set - {x1, x2}
                    reduced_sum = best_sum - 1.0 / x1 - 1.0 / x2

                    # Try adding one element
                    for y in range(1, n + 1):
                        if y in reduced:
                            continue
                        new_set = try_add(reduced, y)
                        if new_set is not None:
                            s = reduced_sum + 1.0 / y
                            if s > best_sum + 1e-15:
                                best_set = new_set
                                best_sum = s
                                improved2 = True
                                break  # restart with new best

                    if improved2:
                        break
                    # Try adding two elements
                    for y1 in range(1, n + 1):
                        if y1 in reduced:
                            continue
                        ns1 = try_add(reduced, y1)
                        if ns1 is None:
                            continue
                        for y2 in range(y1 + 1, n + 1):
                            if y2 in ns1:
                                continue
                            ns2 = try_add(ns1, y2)
                            if ns2 is not None:
                                s = reduced_sum + 1.0 / y1 + 1.0 / y2
                                if s > best_sum + 1e-15:
                                    best_set = ns2
                                    best_sum = s
                                    improved2 = True
                                    break
                        if improved2:
                            break
                    if improved2:
                        break

    # Final: try single additions one more time
    for y in range(1, n + 1):
        if y in best_set:
            continue
        new_set = try_add(best_set, y)
        if new_set is not None:
            s = best_sum + 1.0 / y
            if s > best_sum + 1e-15:
                best_set = new_set
                best_sum = s

    return sorted(best_set)
