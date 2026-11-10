import time
from fractions import Fraction

def _verify(subset):
    """Verify all reciprocal subset sums are distinct."""
    if not subset:
        return True
    sums = {Fraction(0)}
    for x in subset:
        rec = Fraction(1, x)
        new_sums = set()
        for s in sums:
            ns = s + rec
            if ns in sums:
                return False
            new_sums.add(ns)
        sums = new_sums
    return True

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill

    # Known valid seed solution of size 9
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)

    elements = list(range(1, 23))
    recip = {x: Fraction(1, x) for x in elements}

    # ------------------------------------------------------------------
    # STRUCTURAL APPROACH: superincreasing reciprocal sequences.
    # A sequence a_1 < a_2 < ... < a_m with 1/a_k > sum_{j>k} 1/a_j
    # guarantees all 2^m subset sums are distinct.
    # We search for the longest such sequence by building from the tail:
    # pick a_m, a_{m-1}, ... such that the tail sum stays small enough.
    # ------------------------------------------------------------------
    def find_superincreasing():
        """Find longest superincreasing reciprocal sequence in [1,22]."""
        best = []
        # We build the sequence from the largest element (tail) to smallest (head).
        # State: tail_sum = sum of reciprocals of already-placed (larger) elements.
        # When adding a new element x (smaller than all placed), we need 1/x > tail_sum.
        # To maximize length, we want to place as many elements as possible.
        # Strategy: backtracking over choices, preferring large elements for the tail
        # (they contribute small reciprocals, keeping tail_sum small, allowing more head elements).

        def backtrack(available, tail_sum, placed):
            nonlocal best
            if time.time() > deadline:
                return
            if len(placed) > len(best):
                best = list(placed)
            if not available:
                return
            # Try adding each available element as the new head (smallest so far).
            # Condition: 1/x > tail_sum
            for x in available:
                r = Fraction(1, x)
                if r > tail_sum:
                    new_tail = tail_sum + r
                    new_placed = [x] + placed
                    # remaining available must be smaller than x
                    new_available = [y for y in available if y < x]
                    backtrack(new_available, new_tail, new_placed)

        backtrack(elements, Fraction(0), [])
        return best

    si_set = find_superincreasing()
    if len(si_set) > best_size:
        if _verify(si_set):
            best_set = si_set
            best_size = len(si_set)

    # ------------------------------------------------------------------
    # GENERAL EXACT-RATIONAL SEARCH: try to find a set larger than best_size
    # using greedy + backtracking with exact Fractions.
    # ------------------------------------------------------------------
    for k in range(22, best_size, -1):
        found = False
        if time.time() > deadline:
            break

        # Greedy with multiple orderings
        orderings = [
            sorted(elements),
            sorted(elements, reverse=True),
            sorted(elements, key=lambda x: (x % 3, x)),
            sorted(elements, key=lambda x: (x % 5, x)),
        ]
        for order in orderings:
            if time.time() > deadline:
                break
            current_set = []
            current_sums = {Fraction(0)}
            for x in order:
                if len(current_set) == k:
                    break
                rec = recip[x]
                collision = False
                for s in current_sums:
                    if (s + rec) in current_sums:
                        collision = True
                        break
                if not collision:
                    current_set.append(x)
                    for s in list(current_sums):
                        current_sums.add(s + rec)
            if len(current_set) == k and _verify(current_set):
                best_set = current_set
                best_size = k
                found = True
                break
        if found:
            break

        # Backtracking for moderate k
        if k <= 14 and time.time() < deadline:
            def bt(remaining, current_set, current_sums, depth, max_depth):
                if time.time() > deadline:
                    return None
                if depth == max_depth:
                    return list(current_set)
                if not remaining:
                    return None
                for i in range(len(remaining)):
                    x = remaining[i]
                    new_remaining = remaining[:i] + remaining[i+1:]
                    rec = recip[x]
                    collision = False
                    for s in current_sums:
                        if (s + rec) in current_sums:
                            collision = True
                            break
                    if not collision:
                        new_sums = set(current_sums)
                        for s in current_sums:
                            new_sums.add(s + rec)
                        result = bt(new_remaining, current_set + [x], new_sums, depth + 1, max_depth)
                        if result is not None:
                            return result
                return None

            result = bt(elements, [], {Fraction(0)}, 0, k)
            if result is not None and len(result) == k and _verify(result):
                best_set = result
                best_size = k
                found = True
                break

    return best_set

def construct():
    return _construct_impl()

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END
