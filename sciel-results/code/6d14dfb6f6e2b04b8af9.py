import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit (evaluator kills at 90s)
    n = 256
    max_size = 64

    def elapsed():
        return time.time() - start_time

    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False

        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)

        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)

        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                diffs_A.add(abs(A[j] - A[i]))
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                diffs_B.add(abs(B[j] - B[i]))

        if diffs_A & diffs_B:
            return False

        return True

    def compute_state(S):
        """Compute sums set and diff bitmask for a set S."""
        sums = set()
        diff_mask = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[j] - S[i])
                diff_mask |= (1 << d)
        return sums, diff_mask

    def can_add(x, S, sums, diff_mask, other_diff_mask):
        """
        Check if x can be added to S maintaining Sidon and diff disjointness.
        Returns True if valid.
        """
        # Check Sidon: new sums involving x must not collide with existing sums
        if (2 * x) in sums:
            return False
        for s in S:
            if (x + s) in sums:
                return False

        # Check differences: new diffs must not collide with other_diff_mask
        for s in S:
            d = abs(x - s)
            if d == 0:
                continue
            if (other_diff_mask >> d) & 1:
                return False

        return True

    def add_to_set(x, S, sums, diff_mask):
        """Add x to S, updating sums and diff_mask."""
        S.append(x)
        sums.add(2 * x)
        for s in S[:-1]:
            sums.add(x + s)
        for s in S[:-1]:
            d = abs(x - s)
            diff_mask |= (1 << d)
        return diff_mask

    def greedy_build(candidates, other_diff_mask, target_size, seed=None):
        """
        Greedily build a Sidon set with differences disjoint from other_diff_mask.
        Uses random ordering for diversity.
        """
        S = []
        sums = set()
        diff_mask = 0

        cand = candidates[:]
        random.shuffle(cand)

        for x in cand:
            if len(S) >= target_size:
                break
            if can_add(x, S, sums, diff_mask, other_diff_mask):
                diff_mask = add_to_set(x, S, sums, diff_mask)

        return S, sums, diff_mask

    def local_search(A, B, time_limit=3.0):
        """
        Try to improve A and B by adding elements.
        Returns (A, B, sums_A, sums_B, diff_mask_A, diff_mask_B).
        """
        end_time = time.time() + time_limit
        A = A[:]
        B = B[:]

        sums_A, diff_mask_A = compute_state(A)
        sums_B, diff_mask_B = compute_state(B)

        candidates = list(range(1, 257))
        random.shuffle(candidates)

        improved = True
        while improved and time.time() < end_time:
            improved = False

            # Try to add to A
            for x in candidates:
                if x in A:
                    continue
                if can_add(x, A, sums_A, diff_mask_A, diff_mask_B):
                    diff_mask_A = add_to_set(x, A, sums_A, diff_mask_A)
                    improved = True
                    break

            # Try to add to B
            for x in candidates:
                if x in B:
                    continue
                if can_add(x, B, sums_B, diff_mask_B, diff_mask_A):
                    diff_mask_B = add_to_set(x, B, sums_B, diff_mask_B)
                    improved = True
                    break

        return A, B, sums_A, sums_B, diff_mask_A, diff_mask_B

    def mis_conflict_graph(A, B, all_candidates):
        """
        Build a conflict graph for MIS approach.
        Vertices: candidate elements.
        Edges: conflicts (sum collision in A, sum collision in B, diff collision between A and B).
        This is a simplified version - we use greedy coloring heuristic.
        """
        # This is complex, so we use a simpler greedy with random restarts instead
        pass

    def fast_greedy_with_mis_insights(other_diff_mask, target_size):
        """
        Fast greedy that incorporates MIS insights:
        - Use random ordering
        - Prefer elements that have fewer conflicts (lower degree in conflict graph)
        """
        S = []
        sums = set()
        diff_mask = 0

        # Precompute conflict degrees for each candidate
        # Degree = number of other candidates that would conflict if both added
        # This is approximate and fast

        candidates = list(range(1, 257))
        random.shuffle(candidates)

        # Score each candidate by how "compatible" it is with others
        # Lower score = fewer conflicts = better to add first
        scores = {}
        for x in candidates:
            score = 0
            for y in candidates:
                if x == y:
                    continue
                # Check if x and y would conflict in differences with other_diff_mask
                d = abs(x - y)
                if (other_diff_mask >> d) & 1:
                    score += 1
            scores[x] = score

        # Sort by score (ascending) with random tie-breaking
        candidates.sort(key=lambda x: (scores[x], random.random()))

        for x in candidates:
            if len(S) >= target_size:
                break
            if can_add(x, S, sums, diff_mask, other_diff_mask):
                diff_mask = add_to_set(x, S, sums, diff_mask)

        return S, sums, diff_mask

    best_score = 0
    best_A = [1]
    best_B = [2]

    # Strategy:
    # 1. Random partition of differences to guide greedy build.
    # 2. Use MIS-inspired scoring to break ties.
    # 3. Local search to polish.
    # 4. Many restarts with bitset optimization.

    num_restarts = 5000  # Will be limited by time

    for iteration in range(num_restarts):
        if elapsed() > deadline - 5:
            break

        # Randomly assign each difference to A, B, or None
        p_a = 0.45
        p_b = 0.45

        mask_A = 0
        mask_B = 0

        for d in range(1, 256):
            r = random.random()
            if r < p_a:
                mask_A |= (1 << d)
            elif r < p_a + p_b:
                mask_B |= (1 << d)

        # Build A using MIS-inspired greedy
        A, sums_A, diff_mask_A = fast_greedy_with_mis_insights(mask_B, 16)

        # Build B, constrained by A's real diff mask
        B, sums_B, diff_mask_B = fast_greedy_with_mis_insights(diff_mask_A, 16)

        if A and B:
            # Local search to polish
            A2, B2, _, _, _, _ = local_search(A, B, time_limit=1.5)

            if is_valid_solution(A2, B2):
                score = len(A2) + len(B2)
                if score > best_score:
                    best_score = score
                    best_A = A2[:]
                    best_B = B2[:]

    # Fallback
    if best_score == 0:
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
