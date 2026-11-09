import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0

    def elapsed():
        return time.time() - start_time

    # Bitmask-based subset sum tracking.
    # S[k] is a Python int where bit s is 1 iff sum s is achievable with exactly k elements.
    # Max possible sum: 60 * 256 = 15360, so we need ~15361 bits.
    # Python ints handle arbitrary precision, so this is fine.

    def greedy_bitmask(order, time_limit=None):
        """Greedy construction using bitmask representation."""
        # S[k] = bitmask of achievable sums with exactly k elements
        S = [0] * 20
        S[0] = 1  # sum 0 with 0 elements
        current_size = 0
        A = []

        for x in order:
            if time_limit and elapsed() > time_limit:
                break

            valid = True
            # For each j from 1 to current_size+1:
            #   new sums for cardinality j: shift S[j-1] left by x bits
            #   These must not collide with S[i] for i != j
            # Also need: shifted S[j-1] must not collide with any other shifted S[i-1]
            # But as analyzed, (S[j-1]+x) ∩ (S[i-1]+x) = (S[j-1]∩S[i-1])+x = 0 if j≠i

            for j in range(1, current_size + 2):
                if S[j-1] == 0:
                    continue
                shifted = S[j-1] << x
                # Check against all i != j
                for i in range(current_size + 1):
                    if i == j:
                        continue
                    if (shifted & S[i]) != 0:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                A.append(x)
                current_size += 1
                # Update S from high to low to avoid using x twice
                for k in range(current_size, 0, -1):
                    S[k] |= (S[k-1] << x)

        return A

    def is_valid_set(A):
        """Verify a set satisfies cardinality-separated subset sums."""
        n = len(A)
        if n == 0:
            return True
        # Compute all subset sums by cardinality
        S = [set() for _ in range(n + 1)]
        S[0] = {0}
        for x in A:
            for k in range(n, 0, -1):
                for s in S[k-1]:
                    S[k].add(s + x)
        # Check disjointness across different cardinalities
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if S[i] & S[j]:
                    return False
        return True

    def local_search_improve(A, time_limit):
        """Try to improve set A by swapping elements or adding new ones."""
        A = list(A)
        improved = True
        while improved and elapsed() < time_limit:
            improved = False
            n = len(A)
            # Try adding a new element
            for x in range(1, 257):
                if x in A:
                    continue
                test_A = A + [x]
                if is_valid_set(test_A):
                    A = test_A
                    improved = True
                    break
            if elapsed() > time_limit:
                break
            # Try swapping: remove one element, add another
            for i in range(len(A)):
                if elapsed() > time_limit:
                    break
                for x in range(1, 257):
                    if x in A:
                        continue
                    test_A = [A[j] for j in range(len(A)) if j != i] + [x]
                    if is_valid_set(test_A):
                        # Only accept if it leads to same or larger size
                        # (same size here since we swap)
                        # Prefer swaps that allow future growth - just accept
                        A = test_A
                        improved = True
                        break
                if improved:
                    break
        return A

    # Phase 1: Deterministic greedy with bitmask
    best_set = greedy_bitmask(list(range(1, 257)))
    best_len = len(best_set)

    # Phase 2: Try multiple random orderings
    rng = random.Random(42)
    for trial in range(200):
        if elapsed() > deadline - 10:
            break
        order = list(range(1, 257))
        rng.shuffle(order)
        candidate = greedy_bitmask(order)
        if len(candidate) > best_len:
            best_set = candidate
            best_len = len(candidate)

    # Phase 3: Local search improvement
    if elapsed() < deadline - 5:
        improved = local_search_improve(best_set, deadline - 2)
        if len(improved) > best_len:
            best_set = improved
            best_len = len(improved)

    # Final verification
    if not is_valid_set(best_set):
        # Fallback: rebuild with greedy
        best_set = greedy_bitmask(list(range(1, 257)))

    return [int(x) for x in best_set]
