"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute coprime relationships for numbers in [2, 256]
    max_val = 256
    coprime = [[False] * (max_val + 1) for _ in range(max_val + 1)]
    for a in range(2, max_val + 1):
        for b in range(a, max_val + 1):
            if gcd(a, b) == 1:
                coprime[a][b] = True
                coprime[b][a] = True

    # Helper: check if all distinct sums are pairwise coprime
    def check_pairwise_coprime(sums):
        slist = sorted(sums)
        n = len(slist)
        if n <= 1:
            return True
        for i in range(n):
            for j in range(i + 1, n):
                if not coprime[slist[i]][slist[j]]:
                    return False
        return True

    # Helper: evaluate a pair (A, B) and return (count, valid)
    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        count = len(sums)
        if count <= 1:
            return count, True
        return count, check_pairwise_coprime(sums)

    best_count = 0
    best_A = None
    best_B = None

    # Strategy: CSP-style backtracking over the sum grid
    # The 9 sums s_ij = A[i] + B[j] must satisfy:
    #   s_ij - s_ik = s_jk - s_jk'  (rank-1 additive structure)
    # Equivalently: s_00 + s_11 = s_01 + s_10, etc.
    # 
    # We enumerate A = {a0, a1, a2} and B = {b0, b1, b2} with a0 < a1 < a2, b0 < b1 < b2
    # and use the structure to prune.
    #
    # Key insight for pruning: if we fix A, then for each B, the sums are determined.
    # But we can also think of it as: pick 3 values for A, then for each choice of B,
    # check the 9 sums.
    #
    # Smarter: enumerate A (C(128,3) ~ 340k), and for each A, do a constrained search over B.
    # For a fixed A, the sums are a_i + b_j. We need all distinct sums to be pairwise coprime.
    #
    # Let's try: for each A (sorted), enumerate B (sorted) but prune early.
    # To make this feasible, we limit the search and use randomization + local improvement.

    import random
    random.seed(123)

    # Phase 1: Random search with time budget
    # Try many random pairs
    for iteration in range(2000000):
        if time.time() > deadline - 10:  # leave 10s for local search
            break

        A = tuple(sorted(random.sample(range(1, 129), 3)))
        B = tuple(sorted(random.sample(range(1, 129), 3)))

        count, valid = evaluate(A, B)
        if valid and count > best_count:
            best_count = count
            best_A = list(A)
            best_B = list(B)

    # Phase 2: Local search / hill climbing
    if best_A is not None and time.time() < deadline - 2:
        current_A = best_A[:]
        current_B = best_B[:]
        current_count = best_count

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False

            # Try modifying one element of A
            for idx in range(3):
                if improved:
                    break
                # Try nearby values first, then wider range
                center = current_A[idx]
                for offset in range(0, 129):
                    if improved:
                        break
                    for delta in ([0] if offset == 0 else [-offset, offset]):
                        new_val = center + delta
                        if new_val < 1 or new_val > 128:
                            continue
                        if new_val == current_A[idx]:
                            continue
                        # Check distinctness
                        if any(new_val == current_A[k] for k in range(3) if k != idx):
                            continue

                        new_A = current_A[:]
                        new_A[idx] = new_val
                        new_A.sort()

                        count, valid = evaluate(new_A, current_B)
                        if valid and count > current_count:
                            current_A = new_A
                            current_count = count
                            improved = True
                            break

            # Try modifying one element of B
            for idx in range(3):
                if improved:
                    break
                center = current_B[idx]
                for offset in range(0, 129):
                    if improved:
                        break
                    for delta in ([0] if offset == 0 else [-offset, offset]):
                        new_val = center + delta
                        if new_val < 1 or new_val > 128:
                            continue
                        if new_val == current_B[idx]:
                            continue
                        if any(new_val == current_B[k] for k in range(3) if k != idx):
                            continue

                        new_B = current_B[:]
                        new_B[idx] = new_val
                        new_B.sort()

                        count, valid = evaluate(current_A, new_B)
                        if valid and count > current_count:
                            current_B = new_B
                            current_count = count
                            improved = True
                            break

            if current_count > best_count:
                best_count = current_count
                best_A = current_A[:]
                best_B = current_B[:]

    # Phase 3: Targeted search for 9 distinct pairwise-coprime sums
    # Try to find A, B such that all 9 sums are distinct and pairwise coprime.
    # This is the maximum possible (3x3 grid = 9 cells).
    if best_count < 9 and time.time() < deadline - 1:
        # Try structured search: pick A with small differences to maximize spread
        # and B similarly
        # Focus on values that produce sums in ranges with many coprime pairs
        for a0 in range(1, 30):
            if time.time() > deadline - 1:
                break
            for a1 in range(a0 + 1, 50):
                if time.time() > deadline - 1:
                    break
                for a2 in range(a1 + 1, 80):
                    if time.time() > deadline - 1:
                        break
                    A = [a0, a1, a2]
                    for b0 in range(1, 30):
                        if time.time() > deadline - 1:
                            break
                        for b1 in range(b0 + 1, 50):
                            if time.time() > deadline - 1:
                                break
                            for b2 in range(b1 + 1, 80):
                                if time.time() > deadline - 1:
                                    break
                                B = [b0, b1, b2]
                                count, valid = evaluate(A, B)
                                if valid and count > best_count:
                                    best_count = count
                                    best_A = A[:]
                                    best_B = B[:]
                                    if best_count == 9:
                                        break
                            if best_count == 9:
                                break
                        if best_count == 9:
                            break
                    if best_count == 9:
                        break
                if best_count == 9:
                    break
            if best_count == 9:
                break

    # Fallback: known good solution
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    seed_count, seed_valid = evaluate(seed_A, seed_B)

    if best_A is None or best_count < seed_count:
        return [seed_A, seed_B]

    return [sorted(best_A), sorted(best_B)]
# EVOLVE-BLOCK-END
