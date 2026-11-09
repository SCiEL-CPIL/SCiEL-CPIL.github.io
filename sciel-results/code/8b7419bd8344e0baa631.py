import random
import time

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 55.0

    best_sum = 0
    best_A = []
    best_B = []

    # Bitset-based helpers for speed
    def can_add_to_sidon(x, S, sums_used):
        """Check if x can be added to Sidon set S. sums_used is a set of used pair sums."""
        # Check cross sums
        for s in S:
            pair_sum = x + s
            if pair_sum in sums_used:
                return False
        # Check diagonal sum
        if 2 * x in sums_used:
            return False
        return True

    def add_to_sidon(x, S, sums_used):
        """Add x to S, update sums_used."""
        for s in S:
            sums_used.add(x + s)
        sums_used.add(2 * x)
        S.append(x)

    def get_diffs_set(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    def can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
        """Check if x can be added to B (Sidon + no diff conflict with A)."""
        for s in B:
            pair_sum = x + s
            if pair_sum in sums_B:
                return False
        if 2 * x in sums_B:
            return False
        for s in B:
            d1 = x - s
            d2 = s - x
            if d1 in diffs_A or d2 in diffs_A:
                return False
            if d1 in diffs_B or d2 in diffs_B:
                return False
        return True

    def add_to_B(x, B, sums_B, diffs_B):
        for s in B:
            sums_B.add(x + s)
            d1 = x - s
            d2 = s - x
            diffs_B.add(d1)
            diffs_B.add(d2)
        sums_B.add(2 * x)
        B.append(x)

    def polish(A, B, sums_A, diffs_A, sums_B, diffs_B, deadline):
        """Try to improve (A, B) by inserting skipped elements."""
        target_size = 16
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    return A, B, sums_A, diffs_A, sums_B, diffs_B
                if x in A or x in B:
                    continue
                if len(A) >= target_size:
                    break
                if can_add_to_sidon(x, A, sums_A):
                    add_to_sidon(x, A, sums_A)
                    improved = True
            for x in range(1, n + 1):
                if time.time() > deadline:
                    return A, B, sums_A, diffs_A, sums_B, diffs_B
                if x in A or x in B:
                    continue
                if len(B) >= target_size:
                    break
                if can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                    add_to_B(x, B, sums_B, diffs_B)
                    improved = True
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    for restart in range(2000):
        if time.time() > deadline:
            break

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        # Greedy for A
        for x in candidates:
            if len(A) >= 16:
                break
            if can_add_to_sidon(x, A, sums_A):
                add_to_sidon(x, A, sums_A)

        diffs_A = get_diffs_set(A)

        # Greedy for B
        for x in candidates:
            if len(B) >= 16:
                break
            if can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                add_to_B(x, B, sums_B, diffs_B)

        # Polish
        A, B, sums_A, diffs_A, sums_B, diffs_B = polish(A, B, sums_A, diffs_A, sums_B, diffs_B, deadline)

        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]

        if best_sum >= 32:
            break

    return [best_A, best_B]
