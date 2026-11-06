# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_sidon(S):
    """Check Sidon property: all pair sums (with replacement) are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    """Return set of absolute nonzero differences."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    """Validate solution. Returns (is_valid, score)."""
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def construct():
    """
    Main entry point. Uses a combined approach:
    1. Build initial sets via greedy with difference partitioning.
    2. Improve via local search (add/swap).
    3. Return the best valid pair found.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    best_A, best_B = [1], [3]
    best_score = 2

    # Precompute all possible differences
    max_n = 256
    max_size = 64

    def try_add_to_set(S, sums, used_diffs, x, pool, max_size=64):
        """
        Try adding x to set S.
        Returns (success, new_S, new_sums, new_used_diffs)
        """
        if len(S) >= max_size:
            return False, S, sums, used_diffs

        # Check Sidon sums
        for a in S:
            if x + a in sums:
                return False, S, sums, used_diffs
        if 2 * x in sums:
            return False, S, sums, used_diffs

        # Check differences
        for a in S:
            d = abs(x - a)
            if d > 0:
                if d not in pool or d in used_diffs:
                    return False, S, sums, used_diffs

        # All good, add it
        new_S = S + [x]
        new_sums = set(sums)
        for a in S:
            new_sums.add(x + a)
        new_sums.add(2 * x)

        new_used_diffs = set(used_diffs)
        for a in S:
            d = abs(x - a)
            if d > 0:
                new_used_diffs.add(d)

        return True, new_S, new_sums, new_used_diffs

    def local_search(A, B, sums_A, sums_B, used_diffs_A, used_diffs_B, time_budget, rng, max_n=256, max_size=64):
        """
        Local search to improve A and B.
        """
        start_ls = time.time()
        A = list(A)
        B = list(B)
        sums_A = set(sums_A)
        sums_B = set(sums_B)
        used_diffs_A = set(used_diffs_A)
        used_diffs_B = set(used_diffs_B)
        all_A = set(A)
        all_B = set(B)

        improved = True
        while improved and time.time() - start_ls < time_budget:
            improved = False

            # Try adding to A or B
            candidates = [x for x in range(1, max_n + 1) if x not in all_A and x not in all_B]
            rng.shuffle(candidates)

            for x in candidates:
                if time.time() - start_ls > time_budget:
                    break

                # Try A
                if len(A) < max_size:
                    ok, nA, nsA, ndA = try_add_to_set(A, sums_A, used_diffs_A, x, used_diffs_A | set(range(1, max_n + 1)), max_size)
                    if ok:
                        # Check difference disjointness
                        new_diffs = _diffs(nA)
                        if not (new_diffs & used_diffs_B):
                            A = nA
                            sums_A = nsA
                            used_diffs_A = ndA
                            all_A.add(x)
                            improved = True
                            break

                # Try B
                if len(B) < max_size:
                    ok, nB, nsB, ndB = try_add_to_set(B, sums_B, used_diffs_B, x, used_diffs_B | set(range(1, max_n + 1)), max_size)
                    if ok:
                        new_diffs = _diffs(nB)
                        if not (new_diffs & used_diffs_A):
                            B = nB
                            sums_B = nsB
                            used_diffs_B = ndB
                            all_B.add(x)
                            improved = True
                            break

            if not improved:
                # Try swaps: remove from A, add to B or vice versa
                if len(A) > 1:
                    idx = rng.randint(0, len(A) - 1)
                    removed = A[idx]
                    A_temp = A[:idx] + A[idx+1:]
                    sums_temp = set()
                    for i in range(len(A_temp)):
                        for j in range(i, len(A_temp)):
                            sums_temp.add(A_temp[i] + A_temp[j])
                    used_diffs_temp = _diffs(A_temp)
                    all_A_temp = set(A_temp)

                    for x in range(1, max_n + 1):
                        if x in all_A_temp or x in all_B:
                            continue
                        ok, nB, nsB, ndB = try_add_to_set(B, sums_B, used_diffs_B, x, used_diffs_B | set(range(1, max_n + 1)), max_size)
                        if ok:
                            new_diffs = _diffs(nB)
                            if not (new_diffs & used_diffs_temp):
                                B = nB
                                sums_B = nsB
                                used_diffs_B = ndB
                                all_B.add(x)
                                A = A_temp
                                sums_A = sums_temp
                                used_diffs_A = used_diffs_temp
                                all_A = all_A_temp
                                improved = True
                                break

                    if not improved:
                        # Restore
                        A = A_temp + [removed]
                        A.sort()
                        all_A = set(A)
                        sums_A = set()
                        for i in range(len(A)):
                            for j in range(i, len(A)):
                                sums_A.add(A[i] + A[j])
                        used_diffs_A = _diffs(A)

                if len(B) > 1 and not improved:
                    idx = rng.randint(0, len(B) - 1)
                    removed = B[idx]
                    B_temp = B[:idx] + B[idx+1:]
                    sums_temp = set()
                    for i in range(len(B_temp)):
                        for j in range(i, len(B_temp)):
                            sums_temp.add(B_temp[i] + B_temp[j])
                    used_diffs_temp = _diffs(B_temp)
                    all_B_temp = set(B_temp)

                    for x in range(1, max_n + 1):
                        if x in all_A or x in all_B_temp:
                            continue
                        ok, nA, nsA, ndA = try_add_to_set(A, sums_A, used_diffs_A, x, used_diffs_A | set(range(1, max_n + 1)), max_size)
                        if ok:
                            new_diffs = _diffs(nA)
                            if not (new_diffs & used_diffs_temp):
                                A = nA
                                sums_A = nsA
                                used_diffs_A = ndA
                                all_A.add(x)
                                B = B_temp
                                sums_B = sums_temp
                                used_diffs_B = used_diffs_temp
                                all_B = all_B_temp
                                improved = True
                                break

                    if not improved:
                        B = B_temp + [removed]
                        B.sort()
                        all_B = set(B)
                        sums_B = set()
                        for i in range(len(B)):
                            for j in range(i, len(B)):
                                sums_B.add(B[i] + B[j])
                        used_diffs_B = _diffs(B)

        return A, B, sums_A, sums_B, used_diffs_A, used_diffs_B

    # Main search loop
    num_rounds = 20
    for rnd in range(num_rounds):
        if time.time() > deadline - 5.0:
            break

        remaining = deadline - 5.0 - time.time()
        if remaining <= 3.0:
            break

        rng = random.Random(rnd * 7919 + 12345)

        # Create a random partition of differences
        pool_A = set()
        pool_B = set()
        for d in range(1, max_n):
            r = rng.random()
            if r < 0.45:
                pool_A.add(d)
            elif r < 0.90:
                pool_B.add(d)

        if not pool_A:
            pool_A.add(1)
        if not pool_B:
            pool_B.add(2)

        # Greedy construction: alternating order
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        used_diffs_A = set()
        used_diffs_B = set()

        # Try adding in random order
        order = list(range(1, max_n + 1))
        rng.shuffle(order)

        for x in order:
            if time.time() > deadline - 3.0:
                break

            # Try A first, then B, with some randomness
            if len(A) < max_size and (len(A) < len(B) or rng.random() < 0.5):
                ok, nA, nsA, ndA = try_add_to_set(A, sums_A, used_diffs_A, x, pool_A, max_size)
                if ok:
                    new_diffs = _diffs(nA)
                    if not (new_diffs & used_diffs_B):
                        A = nA
                        sums_A = nsA
                        used_diffs_A = ndA
                        continue

            if len(B) < max_size:
                ok, nB, nsB, ndB = try_add_to_set(B, sums_B, used_diffs_B, x, pool_B, max_size)
                if ok:
                    new_diffs = _diffs(nB)
                    if not (new_diffs & used_diffs_A):
                        B = nB
                        sums_B = nsB
                        used_diffs_B = ndB
                        continue

        if A and B:
            # Local search
            ls_budget = min(remaining - (time.time() - (start_time + rnd * 0)), 5.0)
            if ls_budget > 1.0:
                A, B, sums_A, sums_B, used_diffs_A, used_diffs_B = local_search(
                    A, B, sums_A, sums_B, used_diffs_A, used_diffs_B, ls_budget, rng
                )

            valid, score = _validate(A, B)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A), list(B)

        if best_score >= 64:
            break

    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]

    return [best_A, best_B]

# EVOLVE-BLOCK-END
