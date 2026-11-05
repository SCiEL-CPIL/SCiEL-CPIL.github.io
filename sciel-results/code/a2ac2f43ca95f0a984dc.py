import time
import random
import math

N = 256
MAX_EACH = 64
DEADLINE_MARGIN = 5.0  # safety margin before 90s limit


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


def _greedy_build(candidates_order, prefer_A_first=True):
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


def _local_search(A, B, max_iters=2000, deadline=None):
    """Try to improve by swapping elements between sets or adding new ones."""
    A = list(A)
    B = list(B)

    def score():
        return len(A) + len(B)

    def valid():
        if not A or not B:
            return False
        # Quick Sidon check
        for S in (A, B):
            sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        return False
                    sums.add(s)
        dA = _diffs(A)
        dB = _diffs(B)
        return len(dA & dB) == 0

    if not valid():
        return A, B

    best_A, best_B = list(A), list(B)
    best_score = score()
    cur_score = best_score

    all_vals = set(range(1, N + 1))
    used = set(A) | set(B)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Random move: swap an element from A with one from B, or move within
        move_type = random.random()
        if move_type < 0.4:
            # Swap: remove a from A, add b from B to A, and vice versa
            if len(A) < 2 or len(B) < 2:
                continue
            a = random.choice(A)
            b = random.choice(B)
            if a == b:
                continue
            A.remove(a)
            B.remove(b)
            A.append(b)
            B.append(a)
            if valid():
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                # Revert
                A.remove(b)
                B.remove(a)
                A.append(a)
                B.append(b)
        elif move_type < 0.7:
            # Move element from A to B
            if len(A) < 2:
                continue
            a = random.choice(A)
            A.remove(a)
            if a not in B:
                B.append(a)
            if valid():
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                B.remove(a)
                A.append(a)
        elif move_type < 0.9:
            # Move element from B to A
            if len(B) < 2:
                continue
            b = random.choice(B)
            B.remove(b)
            if b not in A:
                A.append(b)
            if valid():
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                A.remove(b)
                B.append(b)
        else:
            # Try adding a new unused element to A or B
            unused = all_vals - used
            if not unused:
                continue
            x = random.choice(list(unused))
            target = A if random.random() < 0.5 else B
            target.append(x)
            if valid():
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
                used.add(x)
            else:
                target.remove(x)

    return best_A, best_B


def construct():
    deadline = time.time() + (90.0 - DEADLINE_MARGIN)
    best_size = 0
    best_sets = ([], [])

    # Strategy 1: Deterministic greedy with different orderings
    # Try ascending, descending, and a few structured orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        # Interleaved: odds first then evens
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
        # Reverse interleaved
        [x for x in range(N, 0, -1) if x % 2 == 0] + [x for x in range(N, 0, -1) if x % 2 == 1],
        # Blocks: 1-64, 65-128, 129-192, 193-256
        list(range(1, 65)) + list(range(65, 129)) + list(range(129, 193)) + list(range(193, 257)),
        # Reverse blocks
        list(range(193, 257)) + list(range(129, 193)) + list(range(65, 129)) + list(range(1, 65)),
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        A, B = _greedy_build(order)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
        # Also try with prefer_A_first=False
        A2, B2 = _greedy_build(order, prefer_A_first=False)
        if _valid_pair(A2, B2):
            s = len(A2) + len(B2)
            if s > best_size:
                best_size = s
                best_sets = (list(A2), list(B2))

    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(42)
    num_random_tries = 200
    for t in range(num_random_tries):
        if time.time() > deadline:
            break
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        A, B = _greedy_build(candidates)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
            # Local search on the best
            if s >= best_size - 2:
                A_ls, B_ls = _local_search(A, B, max_iters=500, deadline=deadline)
                if _valid_pair(A_ls, B_ls):
                    s2 = len(A_ls) + len(B_ls)
                    if s2 > best_size:
                        best_size = s2
                        best_sets = (list(A_ls), list(B_ls))

    # Strategy 3: Local search on the best found
    if best_sets[0] and best_sets[1]:
        A_ls, B_ls = _local_search(best_sets[0], best_sets[1], max_iters=5000, deadline=deadline)
        if _valid_pair(A_ls, B_ls):
            s = len(A_ls) + len(B_ls)
            if s > best_size:
                best_size = s
                best_sets = (list(A_ls), list(B_ls))

    # Fallback: ensure non-empty valid pair
    if not best_sets[0] or not best_sets[1]:
        # A={1, 10}, B={2, 20}: Sidon, diffs {±9} vs {±18}, disjoint
        best_sets = ([1, 10], [2, 20])
        best_size = 4

    # Final validation
    if not _valid_pair(best_sets[0], best_sets[1]):
        best_sets = ([1, 10], [2, 20])

    return [list(best_sets[0]), list(best_sets[1])]
