import time
import random
import math

N = 256
MAX_EACH = 64
DEADLINE_MARGIN = 10.0  # safety margin before 90s limit


def _is_sidon(S):
    """Check if S is a Sidon set (pair sums including a+a unique)."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(S):
    """Nonzero difference set of S."""
    d = set()
    for i in range(len(S)):
        for j in range(len(S)):
            if i != j:
                d.add(S[i] - S[j])
    return d


def _valid_pair(A, B):
    """Check both Sidon and disjoint nonzero differences."""
    if not A or not B:
        return False
    if not _is_sidon(A) or not _is_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return len(dA & dB) == 0


def _greedy_build(candidates_order, prefer_A_first=True, deadline=None):
    """Greedy build: for each candidate in order, try adding to A or B."""
    A = []
    B = []
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()

    def can_add_to_A(x):
        for a in A:
            s = x + a
            if s in A_sums:
                return False
        if 2 * x in A_sums:
            return False
        for a in A:
            d = x - a
            if d in B_diffs:
                return False
        return True

    def can_add_to_B(x):
        for b in B:
            s = x + b
            if s in B_sums:
                return False
        if 2 * x in B_sums:
            return False
        for b in B:
            d = x - b
            if d in A_diffs:
                return False
        return True

    for x in candidates_order:
        if deadline is not None and time.time() > deadline:
            break
        okA = can_add_to_A(x)
        okB = can_add_to_B(x)
        if okA and okB:
            # Prefer to balance sizes; add to smaller
            if len(A) <= len(B) and prefer_A_first:
                A.append(x)
                for a in A[:-1]:
                    A_sums.add(x + a)
                    A_diffs.add(x - a)
                    A_diffs.add(a - x)
                A_sums.add(2 * x)
            else:
                B.append(x)
                for b in B[:-1]:
                    B_sums.add(x + b)
                    B_diffs.add(x - b)
                    B_diffs.add(b - x)
                B_sums.add(2 * x)
        elif okA:
            A.append(x)
            for a in A[:-1]:
                A_sums.add(x + a)
                A_diffs.add(x - a)
                A_diffs.add(a - x)
            A_sums.add(2 * x)
        elif okB:
            B.append(x)
            for b in B[:-1]:
                B_sums.add(x + b)
                B_diffs.add(x - b)
                B_diffs.add(b - x)
            B_sums.add(2 * x)
    return A, B


def _build_state(A, B):
    """Build sum sets and diff sets for A and B."""
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            A_sums.add(A[i] + A[j])
    for i in range(len(B)):
        for j in range(i, len(B)):
            B_sums.add(B[i] + B[j])
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                A_diffs.add(A[i] - A[j])
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                B_diffs.add(B[i] - B[j])
    return A_sums, B_sums, A_diffs, B_diffs


def _can_add_to_set(x, S, S_sums, other_diffs):
    """Check if x can be added to set S given its sums and the other set's diffs."""
    for s in S:
        if x + s in S_sums:
            return False
    if 2 * x in S_sums:
        return False
    for s in S:
        if (x - s) in other_diffs:
            return False
    return True


def _add_to_set(x, S, S_sums, S_diffs):
    """Add x to set S, updating sums and diffs in place."""
    for s in S:
        S_sums.add(x + s)
        S_diffs.add(x - s)
        S_diffs.add(s - x)
    S_sums.add(2 * x)
    S.append(x)


def _remove_from_set(x, S, S_sums, S_diffs):
    """Remove x from set S, rebuilding sums and diffs."""
    S.remove(x)
    # Rebuild sums and diffs from scratch (simpler and still fast for small sets)
    new_sums = set()
    new_diffs = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            new_sums.add(S[i] + S[j])
    for i in range(len(S)):
        for j in range(len(S)):
            if i != j:
                new_diffs.add(S[i] - S[j])
    S_sums.clear()
    S_sums.update(new_sums)
    S_diffs.clear()
    S_diffs.update(new_diffs)


def _local_search(A, B, max_iters=2000, deadline=None):
    """Try to improve by swapping elements between sets or adding new ones."""
    A = list(A)
    B = list(B)

    def score():
        return len(A) + len(B)

    def valid():
        if not A or not B:
            return False
        A_sums, B_sums, A_diffs, B_diffs = _build_state(A, B)
        return len(A_diffs & B_diffs) == 0

    if not valid():
        return A, B

    best_A, best_B = list(A), list(B)
    best_score = score()
    cur_score = best_score

    all_vals = set(range(1, N + 1))
    used = set(A) | set(B)
    unused_list = list(all_vals - used)

    # Maintain incremental state
    A_sums, B_sums, A_diffs, B_diffs = _build_state(A, B)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        move_type = random.random()
        if move_type < 0.4:
            # Swap: remove a from A, add b from B to A, and vice versa
            if len(A) < 2 or len(B) < 2:
                continue
            a = random.choice(A)
            b = random.choice(B)
            if a == b:
                continue
            # Check if swap is valid
            # After swap: A' = A - {a} + {b}, B' = B - {b} + {a}
            # We need to check Sidon for A' and B', and disjoint diffs
            # Quick check: try the swap and validate
            _remove_from_set(a, A, A_sums, A_diffs)
            _remove_from_set(b, B, B_sums, B_diffs)
            if _can_add_to_set(b, A, A_sums, B_diffs) and _can_add_to_set(a, B, B_sums, A_diffs):
                _add_to_set(b, A, A_sums, A_diffs)
                _add_to_set(a, B, B_sums, B_diffs)
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                # Revert
                _add_to_set(a, A, A_sums, A_diffs)
                _add_to_set(b, B, B_sums, B_diffs)
        elif move_type < 0.7:
            # Move element from A to B
            if len(A) < 2:
                continue
            a = random.choice(A)
            _remove_from_set(a, A, A_sums, A_diffs)
            if _can_add_to_set(a, B, B_sums, A_diffs):
                _add_to_set(a, B, B_sums, B_diffs)
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                _add_to_set(a, A, A_sums, A_diffs)
        elif move_type < 0.9:
            # Move element from B to A
            if len(B) < 2:
                continue
            b = random.choice(B)
            _remove_from_set(b, B, B_sums, B_diffs)
            if _can_add_to_set(b, A, A_sums, B_diffs):
                _add_to_set(b, A, A_sums, A_diffs)
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                _add_to_set(b, B, B_sums, B_diffs)
        else:
            # Try adding a new unused element to A or B
            if not unused_list:
                continue
            x = random.choice(unused_list)
            target = A if random.random() < 0.5 else B
            if target is A:
                if _can_add_to_set(x, A, A_sums, B_diffs):
                    _add_to_set(x, A, A_sums, A_diffs)
                    unused_list.remove(x)
                    s = score()
                    if s > best_score:
                        best_score = s
                        best_A, best_B = list(A), list(B)
                        cur_score = s
            else:
                if _can_add_to_set(x, B, B_sums, A_diffs):
                    _add_to_set(x, B, B_sums, B_diffs)
                    unused_list.remove(x)
                    s = score()
                    if s > best_score:
                        best_score = s
                        best_A, best_B = list(A), list(B)
                        cur_score = s

    return best_A, best_B


def construct():
    deadline = time.time() + (90.0 - DEADLINE_MARGIN)
    best_size = 0
    best_sets = ([], [])

    # Strategy 1: Deterministic greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
        [x for x in range(N, 0, -1) if x % 2 == 0] + [x for x in range(N, 0, -1) if x % 2 == 1],
        list(range(1, 65)) + list(range(65, 129)) + list(range(129, 193)) + list(range(193, 257)),
        list(range(193, 257)) + list(range(129, 193)) + list(range(65, 129)) + list(range(1, 65)),
        [x for x in range(1, N + 1) if x % 3 != 0] + [x for x in range(1, N + 1) if x % 3 == 0],
        [x for x in range(N, 0, -1) if x % 3 != 0] + [x for x in range(N, 0, -1) if x % 3 == 0],
        # New: modular orderings to spread differences
        [x for x in range(1, N + 1) if x % 4 in (1, 2)] + [x for x in range(1, N + 1) if x % 4 in (0, 3)],
        [x for x in range(N, 0, -1) if x % 5 in (1, 2, 3)] + [x for x in range(N, 0, -1) if x % 5 in (0, 4)],
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        A, B = _greedy_build(order, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
        A2, B2 = _greedy_build(order, prefer_A_first=False, deadline=deadline)
        if _valid_pair(A2, B2):
            s = len(A2) + len(B2)
            if s > best_size:
                best_size = s
                best_sets = (list(A2), list(B2))

    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(42)
    num_random_tries = 500
    for t in range(num_random_tries):
        if time.time() > deadline:
            break
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        A, B = _greedy_build(candidates, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
            # Local search on promising solutions
            if s >= best_size - 3:
                A_ls, B_ls = _local_search(A, B, max_iters=800, deadline=deadline)
                if _valid_pair(A_ls, B_ls):
                    s2 = len(A_ls) + len(B_ls)
                    if s2 > best_size:
                        best_size = s2
                        best_sets = (list(A_ls), list(B_ls))

    # Strategy 3: Extended local search on the best found
    if best_sets[0] and best_sets[1]:
        A_ls, B_ls = _local_search(best_sets[0], best_sets[1], max_iters=10000, deadline=deadline)
        if _valid_pair(A_ls, B_ls):
            s = len(A_ls) + len(B_ls)
            if s > best_size:
                best_size = s
                best_sets = (list(A_ls), list(B_ls))

    # Fallback: ensure non-empty valid pair
    if not best_sets[0] or not best_sets[1]:
        best_sets = ([1, 10], [2, 20])
        best_size = 4

    # Final validation
    if not _valid_pair(best_sets[0], best_sets[1]):
        best_sets = ([1, 10], [2, 20])

    return [list(best_sets[0]), list(best_sets[1])]
