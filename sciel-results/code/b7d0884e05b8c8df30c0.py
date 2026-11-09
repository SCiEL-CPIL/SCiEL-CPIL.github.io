"""Rewritten program using constraint propagation for hypergraph Ramsey R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _triple_index(a, b, c, n):
    # Lexicographic index of (a,b,c) with 0<=a<b<c<n
    # Count triples (a',b',c') < (a,b,c)
    # a' < a: sum_{i=0}^{a-1} C(n-1-i, 2)
    # a' == a, b' < b: sum_{j=a+1}^{b-1} (n-1-j)
    # a' == a, b' == b, c' < c: (c - b - 1)
    idx = 0
    for i in range(a):
        idx += (n - 1 - i) * (n - 2 - i) // 2
    for j in range(a + 1, b):
        idx += (n - 1 - j)
    idx += (c - b - 1)
    return idx

def _build_color_array(n, colors):
    arr = [0] * (n * (n - 1) * (n - 2) // 6)
    for idx, col in enumerate(colors):
        arr[idx] = col
    return arr

def _check_valid(n, arr):
    # Check no red K4 (all 4 triples red) and no blue K5 (all 10 triples blue)
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        i1 = _triple_index(a, b, c, n)
        i2 = _triple_index(a, b, d, n)
        i3 = _triple_index(a, c, d, n)
        i4 = _triple_index(b, c, d, n)
        if arr[i1] == 0 and arr[i2] == 0 and arr[i3] == 0 and arr[i4] == 0:
            return False
    for quint in combinations(range(n), 5):
        all_blue = True
        for triple in combinations(quint, 3):
            a, b, c = triple
            idx = _triple_index(a, b, c, n)
            if arr[idx] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    return True

def _propagate_constraints(n, arr, unassigned):
    """
    Constraint propagation:
    - If 3 of 4 triples in a K4 are red, the 4th must be blue.
    - If 9 of 10 triples in a K5 are blue, the 10th must be red.
    Returns (arr, unassigned, changed) or None if contradiction.
    """
    changed = False
    # Red K4 constraint: 3 red => 4th must be blue
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        idxs = [
            _triple_index(a, b, c, n),
            _triple_index(a, b, d, n),
            _triple_index(a, c, d, n),
            _triple_index(b, c, d, n),
        ]
        reds = 0
        blue = 0
        unassigned_idx = None
        for idx in idxs:
            if arr[idx] == 0:
                reds += 1
            elif arr[idx] == 1:
                blue += 1
            else:
                unassigned_idx = idx
        if reds == 3 and unassigned_idx is not None and arr[unassigned_idx] == -1:
            arr[unassigned_idx] = 1
            unassigned.discard(unassigned_idx)
            changed = True
        if blue == 4:
            # Already have a blue in K4, that's fine, no constraint
            pass

    # Blue K5 constraint: 9 blue => 10th must be red
    # This is expensive; only do it if n is small enough or selectively
    if n <= 20:
        for quint in combinations(range(n), 5):
            triples = list(combinations(quint, 3))
            idxs = [_triple_index(a, b, c, n) for a, b, c in triples]
            blues = 0
            red = 0
            unassigned_idx = None
            for idx in idxs:
                if arr[idx] == 1:
                    blues += 1
                elif arr[idx] == 0:
                    red += 1
                else:
                    unassigned_idx = idx
            if blues == 9 and unassigned_idx is not None and arr[unassigned_idx] == -1:
                arr[unassigned_idx] = 0
                unassigned.discard(unassigned_idx)
                changed = True
            if red == 1:
                pass  # fine

    return arr, unassigned, changed

def _try_extend(n_target, initial_colors, deadline):
    """
    Try to extend a partial coloring to n_target vertices.
    initial_colors: list of 0/1/-1 for all triples of n_target (initially mostly -1)
    Returns valid coloring as list of 0/1, or None.
    """
    arr = list(initial_colors)
    unassigned = set(i for i, c in enumerate(arr) if c == -1)

    # Initial propagation
    for _ in range(50):
        result = _propagate_constraints(n_target, arr, unassigned)
        if result is None:
            return None
        arr, unassigned, changed = result
        if not changed:
            break

    if _check_valid(n_target, arr):
        return arr

    # Backtracking search with randomization
    best = None
    max_backtracks = 50000

    def backtrack(arr, unassigned, depth):
        nonlocal best, max_backtracks
        if time.time() > deadline:
            return None
        if not unassigned:
            if _check_valid(n_target, arr):
                return list(arr)
            return None
        if max_backtracks <= 0:
            return None
        max_backtracks -= 1

        # Propagate
        for _ in range(20):
            result = _propagate_constraints(n_target, arr, unassigned)
            if result is None:
                return None
            arr, unassigned, changed = result
            if not changed:
                break

        if not unassigned:
            if _check_valid(n_target, arr):
                return list(arr)
            return None

        # Choose a variable: prefer one involved in many constraints
        # Simple: pick first unassigned
        var = min(unassigned)

        # Try both colors
        for color in (0, 1):
            arr[var] = color
            new_unassigned = unassigned - {var}
            result = backtrack(arr, new_unassigned, depth + 1)
            if result is not None:
                return result
            arr[var] = -1

        return None

    result = backtrack(arr, unassigned, 0)
    if result is not None:
        return result
    return None

def _local_search(n, arr, deadline, max_iter=200000):
    """
    Local search: flip random triples to eliminate violations.
    arr: list of 0/1
    Returns improved arr or None.
    """
    arr = list(arr)
    best_arr = list(arr)
    best_violations = _count_violations(n, best_arr)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if best_violations == 0:
            return best_arr

        # Pick a random violation and fix it
        violations = _find_violations(n, arr)
        if not violations:
            return arr

        # Pick a random violation
        vtype, verts = random.choice(violations)
        if vtype == 'red_k4':
            a, b, c, d = verts
            idxs = [
                _triple_index(a, b, c, n),
                _triple_index(a, b, d, n),
                _triple_index(a, c, d, n),
                _triple_index(b, c, d, n),
            ]
            # Flip a random one to blue
            idx = random.choice(idxs)
            arr[idx] = 1
        elif vtype == 'blue_k5':
            quint = verts
            triples = list(combinations(quint, 3))
            idxs = [_triple_index(a, b, c, n) for a, b, c in triples]
            idx = random.choice(idxs)
            arr[idx] = 0

        v = _count_violations(n, arr)
        if v < best_violations:
            best_violations = v
            best_arr = list(arr)
            if v == 0:
                return best_arr

    return best_arr

def _count_violations(n, arr):
    count = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (arr[_triple_index(a, b, c, n)] == 0 and
            arr[_triple_index(a, b, d, n)] == 0 and
            arr[_triple_index(a, c, d, n)] == 0 and
            arr[_triple_index(b, c, d, n)] == 0):
            count += 1
    for quint in combinations(range(n), 5):
        all_blue = True
        for triple in combinations(quint, 3):
            a, b, c = triple
            if arr[_triple_index(a, b, c, n)] != 1:
                all_blue = False
                break
        if all_blue:
            count += 1
    return count

def _find_violations(n, arr):
    violations = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (arr[_triple_index(a, b, c, n)] == 0 and
            arr[_triple_index(a, b, d, n)] == 0 and
            arr[_triple_index(a, c, d, n)] == 0 and
            arr[_triple_index(b, c, d, n)] == 0):
            violations.append(('red_k4', quad))
    for quint in combinations(range(n), 5):
        all_blue = True
        for triple in combinations(quint, 3):
            a, b, c = triple
            if arr[_triple_index(a, b, c, n)] != 1:
                all_blue = False
                break
        if all_blue:
            violations.append(('blue_k5', quint))
    return violations

def _random_coloring(n, seed=42):
    rng = random.Random(seed)
    return [rng.randint(0, 1) for _ in range(n * (n - 1) * (n - 2) // 6)]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin

    best_n = 9
    best_colors = None

    # Start with the known valid coloring from the seed
    n0 = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n0), 3)]
    best_n = n0

    # Try to grow from n0 upward
    # Strategy: for each target n, try to find a valid coloring
    # using random initialization + local search + constraint propagation

    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        total_triples = target_n * (target_n - 1) * (target_n - 2) // 6

        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break

            # Initialize with the mod-3 pattern extended, then randomize some
            arr = []
            for a, b, c in combinations(range(target_n), 3):
                if (a + b + c) % 3 == 0:
                    arr.append(0)
                else:
                    arr.append(1)

            # Randomly flip some to break patterns
            rng = random.Random(attempt * 1000 + target_n)
            for i in range(len(arr)):
                if rng.random() < 0.1:
                    arr[i] = 1 - arr[i]

            # Local search to fix violations
            arr = _local_search(target_n, arr, deadline - 5, max_iter=50000)

            if arr is not None and _count_violations(target_n, arr) == 0:
                best_n = target_n
                best_colors = arr
                found = True
                break

        if found:
            # Try to extend further
            continue
        else:
            # Can't reach this n, stop
            break

    # Final check
    if best_colors is None:
        n0 = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n0), 3)]
        best_n = n0

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
