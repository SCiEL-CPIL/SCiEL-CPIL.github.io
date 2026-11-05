"""Optimized program: MODEL-AND-OPTIMIZE for two Sidon sets with disjoint difference sets."""
import time
import random
from itertools import combinations_with_replacement, combinations
import sys

def _is_sidon(s):
    """Check if set s is Sidon (all pair sums a+b with a<=b are unique)."""
    if not s:
        return True
    sums = set()
    for i, a in enumerate(s):
        for b in s[i:]:
            s_val = a + b
            if s_val in sums:
                return False
            sums.add(s_val)
    return True

def _diffs(s):
    """Return set of nonzero absolute differences (actually signed: b-a for a<b)."""
    if len(s) < 2:
        return set()
    d = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            d.add(lst[j] - lst[i])
    return d

def _validate(A, B):
    """Validate both sets are Sidon and have disjoint nonzero differences. Return True if valid."""
    if not A or not B:
        return False
    if not _is_sidon(A):
        return False
    if not _is_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False
    return True

def _greedy_construct(n=256, max_size=64, rng=None, order=None):
    """Greedy construction: try to build two Sidon sets with disjoint diffs."""
    if rng is None:
        rng = random.Random()
    if order is None:
        order = list(range(1, n+1))
        rng.shuffle(order)

    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    for x in order:
        # Try adding to A (Sidon + diff-disjoint from B)
        if len(A) < max_size:
            canA = True
            for a in A:
                s_val = a + x
                if s_val in sumsA:
                    canA = False
                    break
            if canA and (x + x) in sumsA:
                canA = False
            if canA:
                for a in A:
                    d_val = abs(x - a)
                    if d_val != 0 and d_val in diffsB:
                        canA = False
                        break
            if canA:
                A.append(x)
                sumsA.add(x + x)
                for a in A[:-1]:
                    sumsA.add(a + x)
                    diffsA.add(abs(x - a))

        # Try adding to B (Sidon + diff-disjoint from A)
        if len(B) < max_size:
            canB = True
            for b in B:
                s_val = b + x
                if s_val in sumsB:
                    canB = False
                    break
            if canB and (x + x) in sumsB:
                canB = False
            if canB:
                for b in B:
                    d_val = abs(x - b)
                    if d_val != 0 and d_val in diffsA:
                        canB = False
                        break
            if canB:
                B.append(x)
                sumsB.add(x + x)
                for b in B[:-1]:
                    sumsB.add(b + x)
                    diffsB.add(abs(x - b))

    return A, B

def _local_search(A, B, n=256, max_size=64, rng=None, max_iters=5000, deadline=None):
    """Improve by full-candidate add attempts + occasional removal to escape local optima."""
    if rng is None:
        rng = random.Random()

    all_vals = list(range(1, n + 1))

    def try_add_full(side):
        """Try to add any single element to the given side; return True if added."""
        if side == 'A':
            cur, other = A, B
            sums, diffs, odiffs = sumsA, diffsA, diffsB
        else:
            cur, other = B, A
            sums, diffs, odiffs = sumsB, diffsB, diffsA
        if len(cur) >= max_size:
            return False
        cur_set = set(cur)
        for x in all_vals:
            if x in cur_set:
                continue
            ok = True
            for a in cur:
                s_val = a + x
                if s_val in sums:
                    ok = False
                    break
            if ok and (x + x) in sums:
                ok = False
            if ok:
                for a in cur:
                    d_val = abs(x - a)
                    if d_val != 0 and d_val in odiffs:
                        ok = False
                        break
            if ok:
                if side == 'A':
                    A.append(x); sumsA.add(x + x)
                    for a in A[:-1]: sumsA.add(a + x); diffsA.add(abs(x - a))
                else:
                    B.append(x); sumsB.add(x + x)
                    for b in B[:-1]: sumsB.add(b + x); diffsB.add(abs(x - b))
                return True
        return False

    # Maintain incremental state
    sumsA = set(); diffsA = set()
    for a in A:
        sumsA.add(a + a)
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            sumsA.add(A[i] + A[j])
            diffsA.add(abs(A[i] - A[j]))
    sumsB = set(); diffsB = set()
    for b in B:
        sumsB.add(b + b)
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            sumsB.add(B[i] + B[j])
            diffsB.add(abs(B[i] - B[j]))

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break

        # Full-candidate add to A
        if len(A) < max_size:
            try_add_full('A')
        # Full-candidate add to B
        if len(B) < max_size:
            try_add_full('B')

        # Occasionally remove a random element from the larger set to escape
        if rng.random() < 0.15:
            if len(A) > 1 and (len(A) >= len(B) or rng.random() < 0.5):
                idx = rng.randrange(len(A))
                A.pop(idx)
            elif len(B) > 1:
                idx = rng.randrange(len(B))
                B.pop(idx)
            # Rebuild incremental state after removal
            sumsA = set(); diffsA = set()
            for a in A:
                sumsA.add(a + a)
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    sumsA.add(A[i] + A[j]); diffsA.add(abs(A[i] - A[j]))
            sumsB = set(); diffsB = set()
            for b in B:
                sumsB.add(b + b)
            for i in range(len(B)):
                for j in range(i + 1, len(B)):
                    sumsB.add(B[i] + B[j]); diffsB.add(abs(B[i] - B[j]))

    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin

    best_A, best_B = [], []
    best_score = 0

    # Strategy 1: Many greedy runs with different random orderings
    for seed in range(300):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(1, 257))
        rng.shuffle(order)
        A, B = _greedy_construct(256, 64, rng, order)
        score = len(A) + len(B)
        if _validate(A, B) and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)

    # Strategy 2: Structured orderings
    for order_type in ['asc', 'desc', 'interleaved', 'evens_odds']:
        if time.time() > deadline - 10:
            break
        if order_type == 'asc':
            order = list(range(1, 257))
        elif order_type == 'desc':
            order = list(range(256, 0, -1))
        elif order_type == 'interleaved':
            order = []
            for i in range(128):
                order.append(i + 1)
                order.append(256 - i)
        else:
            order = list(range(2, 257, 2)) + list(range(1, 257, 2))

        rng = random.Random(12345)
        A, B = _greedy_construct(256, 64, rng, order)
        score = len(A) + len(B)
        if _validate(A, B) and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)

    # Strategy 3: Local search on best found
    if time.time() < deadline - 5 and best_score > 0:
        A, B = _local_search(best_A, best_B, 256, 64, random.Random(42),
                             max_iters=50000, deadline=deadline)
        if _validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A, best_B = list(A), list(B)

    # Fallback: ensure we return valid nonempty sets
    if not best_A or not best_B:
        best_A, best_B = _greedy_construct(256, 64, random.Random(0), list(range(1, 257)))
        if not _validate(best_A, best_B) or not best_A or not best_B:
            best_A = [1, 2, 4, 8]
            best_B = [3, 5, 7, 11]

    return [best_A, best_B]
