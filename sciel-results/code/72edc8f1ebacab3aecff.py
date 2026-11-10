import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for safety

    n = 256
    candidates = list(range(1, n + 1))

    best_size = 0
    best_A = []
    best_B = []

    def can_add(x, S, sums_S, diffs_S, diffs_other):
        # Check Sidon property for new sums in S
        # New sums are x+a for a in S, and 2x
        # We need to ensure these are not in sums_S and are unique among themselves.

        # 1. Check 2x
        if 2 * x in sums_S:
            return False

        # 2. Check x+a
        for a in S:
            if x + a in sums_S:
                return False

        # Check disjoint differences with other set
        for a in S:
            d = abs(x - a)
            if d in diffs_other:
                return False

        return True

    def add_to_set(x, S, sums_S, diffs_S):
        sums_S.add(2 * x)
        for a in S:
            sums_S.add(x + a)
        for a in S:
            diffs_S.add(abs(x - a))
        S.append(x)

    def greedy_construct(order, prefer_A_first=True):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x in order:
            if time.time() - start_time > time_limit:
                break
            if prefer_A_first:
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                elif can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
            else:
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                elif can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
        return A, B

    def try_improve(A, B):
        """Try local search: swap elements between sets or add/remove to improve balance"""
        nonlocal best_size, best_A, best_B
        A = A[:]
        B = B[:]
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        for a in A:
            sums_A.add(2 * a)
            diffs_A.add(0)
        for a in A:
            for b in A:
                if a != b:
                    sums_A.add(a + b)
                    diffs_A.add(abs(a - b))
        for b in B:
            sums_B.add(2 * b)
            diffs_B.add(0)
        for b in B:
            for c in B:
                if b != c:
                    sums_B.add(b + c)
                    diffs_B.add(abs(b - c))

        # Try to move elements from larger set to smaller if possible
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            if len(A) > len(B):
                src, dst = A, B
                src_sums, dst_sums = sums_A, sums_B
                src_diffs, dst_diffs = diffs_A, diffs_B
            else:
                src, dst = B, A
                src_sums, dst_sums = sums_B, sums_A
                src_diffs, dst_diffs = diffs_B, diffs_A

            for i, x in enumerate(src[:]):
                if time.time() - start_time > time_limit:
                    break
                # Try removing x from src and adding to dst
                # First check if x can be added to dst
                if can_add(x, dst, dst_sums, dst_diffs, src_diffs):
                    # Check if removing x from src keeps src Sidon
                    # We'd need to recompute, which is expensive. Skip for now.
                    pass
        return A, B

    # Strategy 1: Sorted greedy (deterministic baseline)
    A1, B1 = greedy_construct(sorted(candidates), True)
    size1 = len(A1) + len(B1)
    if size1 > best_size:
        best_size = size1
        best_A = A1[:]
        best_B = B1[:]

    # Strategy 2: Sorted greedy, prefer B first
    A2, B2 = greedy_construct(sorted(candidates), False)
    size2 = len(A2) + len(B2)
    if size2 > best_size:
        best_size = size2
        best_A = A2[:]
        best_B = B2[:]

    # Strategy 3: Reverse sorted greedy
    A3, B3 = greedy_construct(sorted(candidates, reverse=True), True)
    size3 = len(A3) + len(B3)
    if size3 > best_size:
        best_size = size3
        best_A = A3[:]
        best_B = B3[:]

    # Strategy 4: Random restarts
    max_restarts = 100000
    restart_count = 0

    while time.time() - start_time < time_limit and restart_count < max_restarts:
        restart_count += 1
        shuffled = candidates[:]
        random.shuffle(shuffled)

        # Alternate which set to prefer
        prefer_A = random.random() < 0.5
        A, B = greedy_construct(shuffled, prefer_A)

        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]

        # If we've found a good solution, do some targeted local search
        if best_size >= 24 and restart_count % 10 == 0:
            # Try to add more elements to the smaller set
            pass

    if best_size == 0:
        return [[1], [2]]

    return [best_A, best_B]
