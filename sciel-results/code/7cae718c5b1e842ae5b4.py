import time
from fractions import Fraction
import heapq
import math

def construct():
    """
    Find a subset of [1, 256] with no non-constant 3-term AP
    that maximizes the sum of reciprocals.
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    n = 256

    # Precompute conflicts: for each pair (a, b), determine if they form a 3-AP
    # Actually, we need: for each pair (i, j), is there a k such that {i, j, k} is a 3-AP?
    # A 3-AP is {a, a+d, a+2d}. So for two numbers i < j, they form a 3-AP with:
    #   k = i + 2*(j-i) = 2j - i (if k <= n and k != i, j)
    #   k = j + (j-i) = 2j - i ... wait, let me think more carefully.
    #
    # Three numbers a < b < c form an AP if b-a == c-b, i.e., a+c = 2b.
    # For a pair (x, y) with x < y, they can be:
    #   - first two: x, y, 2y-x (if 2y-x <= n and 2y-x != x and 2y-x != y)
    #   - first and third: x, (x+y)/2, y (if (x+y) even)
    #   - second and third: 2x-y, x, y (if 2x-y >= 1 and 2x-y != x and 2x-y != y)
    #
    # But the constraint is: NO three elements form an AP. So if we include x and y,
    # we must NOT include any z such that {x,y,z} is a 3-AP.
    #
    # For a pair (x, y), the conflicting third element z is:
    #   If x < y: z could be 2y - x (making x, y, z an AP with x first), 
    #             or (x+y)/2 if integer (making x, z, y an AP),
    #             or 2x - y (making z, x, y an AP with y third)
    #   Actually, any z such that one of {x,y,z} is the midpoint.
    #
    # Let me think differently: three numbers form an AP iff one is the average of the other two.
    # So for a pair (x, y), the forbidden z values are:
    #   z = 2y - x (if 1 <= z <= n, z != x, z != y)  -> x, y, z in AP
    #   z = 2x - y (if 1 <= z <= n, z != x, z != y)  -> z, x, y in AP
    #   z = (x + y) / 2 if (x+y) even and z != x and z != y  -> x, z, y in AP
    #
    # But actually, the simplest approach: for each potential AP {a, b, c} with a < b < c,
    # b = (a+c)/2, the three pairs (a,b), (a,c), (b,c) are all "conflicting" in the sense that
    # you can't have all three. But for an independent set formulation, we need edges between
    # pairs that CANNOT both be in the set.
    #
    # Actually, the standard formulation: two elements i, j conflict if there exists k such that
    # {i, j, k} is a 3-AP. But that's not quite right either. The constraint is that NO THREE
    # elements form an AP. This is a 3-uniform hypergraph constraint, not a simple graph constraint.
    #
    # Let me use a different approach: greedy with local search, using exact rational arithmetic.

    # Greedy initialization: add elements in order of 1/x (descending), skip if it creates an AP
    def creates_ap(existing, x):
        """Check if adding x to existing set creates a 3-AP."""
        s = set(existing)
        for a in s:
            # Check if there's b in s such that a, b, x is AP or a, x, b is AP or x, a, b is AP
            # a, b, x: b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b != a and b != x and b in s:
                    return True
            # a, x, b: b = 2x - a
            b = 2 * x - a
            if b != a and b != x and b in s:
                return True
            # x, a, b: b = 2a - x
            b = 2 * a - x
            if b != a and b != x and b in s:
                return True
        return False

    # Greedy: process in order of decreasing 1/x (i.e., increasing x)
    # Actually, to maximize sum of 1/x, we want small x's. So process 1, 2, 3, ...
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        if not creates_ap(selected, x):
            selected.append(x)
            selected_set.add(x)

    def total_weight(s):
        return sum(Fraction(1, x) for x in s)

    best_weight = total_weight(selected)
    best_set = list(selected)

    # Local search: 1-add and 1-swap
    def try_add(s, x):
        """Try to add x to s, return (success, new_set)"""
        if x in s:
            return False, s
        for a in s:
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b != a and b != x and b in s:
                    return False, s
            b = 2 * x - a
            if b != a and b != x and b in s:
                return False, s
            b = 2 * a - x
            if b != a and b != x and b in s:
                return False, s
        new_s = set(s)
        new_s.add(x)
        return True, new_s

    def try_swap(s, out, inn):
        """Try to remove 'out' and add 'inn' from s."""
        if out not in s or inn in s:
            return False, s
        s2 = set(s)
        s2.remove(out)
        s2.add(inn)
        # Check no 3-AP in s2
        s2_list = list(s2)
        for i in range(len(s2_list)):
            for j in range(i + 1, len(s2_list)):
                a, b = s2_list[i], s2_list[j]
                if (a + b) % 2 == 0:
                    c = (a + b) // 2
                    if c != a and c != b and c in s2:
                        return False, s
                c = 2 * b - a
                if c != a and c != b and c in s2:
                    return False, s
                c = 2 * a - b
                if c != a and c != b and c in s2:
                    return False, s
        return True, s2

    # Local search
    s = set(selected)
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        best_local_weight = total_weight(s)
        
        # 1-swap: try removing one element and adding another
        for out in list(s):
            for inn in range(1, n + 1):
                if inn in s:
                    continue
                success, new_s = try_swap(s, out, inn)
                if success:
                    w = total_weight(new_s)
                    if w > best_local_weight:
                        s = new_s
                        best_local_weight = w
                        improved = True
        
        # 1-add: try adding an element
        for x in range(1, n + 1):
            success, new_s = try_add(s, x)
            if success:
                w = total_weight(new_s)
                if w > best_local_weight:
                    s = new_s
                    best_local_weight = w
                    improved = True

        if total_weight(s) > best_weight:
            best_weight = total_weight(s)
            best_set = list(s)

    return sorted(best_set)
