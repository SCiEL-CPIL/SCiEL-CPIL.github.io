import time
import random

def construct():
    deadline = time.time() + 55
    max_val = 256
    max_size = 60
    max_sum_global = max_size * max_val + 1
    mask_global = (1 << max_sum_global) - 1

    def incremental_validate_add(sums, result, x):
        """Check if x can be added to result given current sums bitsets.
        Returns (new_sums, valid)."""
        n = len(result)
        valid = True
        new_bits_per_k = []
        for k in range(1, n + 2):
            nb = (sums[k - 1] << x) & mask_global
            new_bits_per_k.append(nb)

        for k in range(1, n + 2):
            nb = new_bits_per_k[k - 1]
            if nb == 0:
                continue
            for j in range(max_size + 1):
                if j == k:
                    continue
                if nb & sums[j]:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            return None, False

        new_sums = [0] * (max_size + 1)
        new_sums[0] = sums[0]
        for k in range(1, n + 2):
            new_sums[k] = sums[k] | new_bits_per_k[k - 1]
        return new_sums, True

    def build_from_ordering(candidates, deadline):
        """Greedily build set from candidate ordering using incremental bitset DP."""
        result = []
        sums = [0] * (max_size + 1)
        sums[0] = 1

        for x in candidates:
            if time.time() > deadline:
                break
            if x < 1 or x > 256:
                continue
            if x in result:
                continue

            new_sums, valid = incremental_validate_add(sums, result, x)
            if valid:
                result.append(x)
                sums = new_sums
                if len(result) >= max_size:
                    break

        return result

    def try_extend_fast(best, sums, deadline):
        """Try to add remaining elements using incremental validation."""
        if len(best) >= max_size:
            return best, sums
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in remaining:
                if time.time() > deadline:
                    break
                new_sums, valid = incremental_validate_add(sums, best, x)
                if valid:
                    best.append(x)
                    sums = new_sums
                    used.add(x)
                    remaining.remove(x)
                    improved = True
                    if len(best) >= max_size:
                        break
        return best, sums

    def rebuild_sums(s):
        """Rebuild sums bitsets from scratch for set s."""
        n = len(s)
        sums = [0] * (max_size + 1)
        sums[0] = 1
        for val in s:
            new_sums = [0] * (max_size + 1)
            new_sums[0] = sums[0]
            for k in range(1, n + 1):
                new_sums[k] = sums[k] | ((sums[k - 1] << val) & mask_global)
            sums = new_sums
        return sums

    def local_search_incremental(best, sums, deadline):
        """Try swaps using incremental validation, then extend."""
        if len(best) >= max_size:
            return best, sums
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]

        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline - 3:
                    break
                old_val = best[i]
                for new_val in remaining:
                    if new_val == old_val:
                        continue
                    # Try swap: remove old_val, add new_val
                    # Rebuild sums without old_val, then try adding new_val
                    test_set = best[:i] + best[i+1:]
                    test_sums = rebuild_sums(test_set)
                    new_sums, valid = incremental_validate_add(test_sums, test_set, new_val)
                    if valid:
                        test_set.append(new_val)
                        # Now try to extend
                        extended, ext_sums = try_extend_fast(test_set, new_sums, deadline)
                        if len(extended) > len(best):
                            best = extended
                            sums = ext_sums
                            used = set(best)
                            remaining = [x for x in range(1, 257) if x not in used]
                            improved = True
                            break
                if improved:
                    break
        return best, sums

    random.seed(42)
    results = []

    # Multiple structured orderings
    orderings = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        sorted(range(129, 257), reverse=True) + sorted(range(1, 129)),
        sorted(range(1, 257), key=lambda x: (x % 8, x // 8)),  # lattice-like
        [x for x in range(1, 257) if x % 2 == 1] + [x for x in range(1, 257) if x % 2 == 0],
        [x for x in range(1, 257) if x % 3 != 0] + [x for x in range(1, 257) if x % 3 == 0],
        # Geometric-ish spacing
        [max(1, min(256, round(256 * (i / 256.0) ** 0.7))) for i in range(1, 257)],
        [max(1, min(256, round(256 * (i / 256.0) ** 1.3))) for i in range(1, 257)],
    ]

    # Deduplicate orderings
    seen_orderings = set()
    unique_orderings = []
    for o in orderings:
        key = tuple(o)
        if key not in seen_orderings:
            seen_orderings.add(key)
            unique_orderings.append(o)

    for ordering in unique_orderings:
        if time.time() > deadline - 10:
            break
        r = build_from_ordering(ordering, deadline)
        results.append(r)

    # Random orderings
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        results.append(build_from_ordering(candidates, deadline))

    # Pick best
    best = max(results, key=len)
    if len(best) == 0:
        return [1]

    # Rebuild sums for best
    sums = rebuild_sums(best)

    # Fast extension
    best, sums = try_extend_fast(best, sums, deadline)

    # Local search if time permits
    if time.time() < deadline - 5:
        best, sums = local_search_incremental(best, sums, deadline)

    # Final extension
    best, sums = try_extend_fast(best, sums, deadline)

    if len(best) == 0:
        return [1]

    best = [int(x) for x in best]
    return best
