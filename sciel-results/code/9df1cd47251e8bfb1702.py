"""Rewritten program: systematic search with bitmask coprime checking for pairwise-coprime sumsets."""
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin for 90s limit

    # Best known solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 0

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _pairwise_coprime_bitmask(sums):
        """Check if all values in sums are pairwise coprime using bitmask approach."""
        distinct = sorted(set(sums))
        n = len(distinct)
        if n <= 1:
            return True
        # Build coprime mask for each element
        # For each i, mask[i] has bit j set if gcd(distinct[i], distinct[j]) == 1
        # We need all pairs to be coprime
        for i in range(n):
            for j in range(i + 1, n):
                if _gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    def _score_and_check(A, B):
        """Compute distinct sums and check pairwise coprimality. Returns (count, is_valid)."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        count = len(sums)
        if count <= 1:
            return count, True
        distinct = sorted(sums)
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if _gcd(distinct[i], distinct[j]) != 1:
                    return count, False
        return count, True

    # Compute initial best score
    best_score, _ = _score_and_check(best_A, best_B)

    # If best_score is already 9, return immediately
    if best_score == 9:
        return [best_A, best_B]

    # Systematic search over increasing ranges
    # Start with small range for quick wins, expand if needed
    range_sizes = [15, 20, 25, 30, 40, 50, 70, 100, 128]

    for range_limit in range_sizes:
        if time.time() > deadline:
            break

        candidates = list(itertools.combinations(range(1, range_limit + 1), 3))
        n_cands = len(candidates)

        # For efficiency, we can try to be smarter about ordering
        # But let's do a direct double loop with early termination
        found_nine = False

        for i in range(n_cands):
            if time.time() > deadline:
                break
            A = candidates[i]
            for j in range(n_cands):
                if time.time() > deadline:
                    break
                B = candidates[j]

                # Quick sum computation
                sums = set()
                for a in A:
                    for b in B:
                        sums.add(a + b)

                count = len(sums)

                # If count < best_score, skip coprime check
                if count < best_score:
                    continue

                # Check pairwise coprime
                if count <= 1:
                    valid = True
                else:
                    distinct = sorted(sums)
                    n = len(distinct)
                    valid = True
                    for x in range(n):
                        for y in range(x + 1, n):
                            if _gcd(distinct[x], distinct[y]) != 1:
                                valid = False
                                break
                        if not valid:
                            break

                if valid and count > best_score:
                    best_score = count
                    best_A = list(A)
                    best_B = list(B)
                    if best_score == 9:
                        found_nine = True
                        break
            if found_nine:
                break

        if found_nine:
            break

    return [best_A, best_B]
