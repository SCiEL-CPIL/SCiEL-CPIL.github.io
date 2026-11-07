"""Rewritten program: structured lattice + bitwise DP."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return a list of distinct ints in 1..256 such that subset sums at different
    cardinalities are disjoint. Maximize size via bitwise DP.
    """
    max_val = 256
    max_size = 60

    def check_valid(sums_by_k, max_k):
        # sums_by_k: list of bitsets, index = cardinality
        # Check disjointness for all pairs
        for k1 in range(max_k + 1):
            if sums_by_k[k1] == 0:
                continue
            for k2 in range(k1 + 1, max_k + 1):
                if sums_by_k[k2] == 0:
                    continue
                if sums_by_k[k1] & sums_by_k[k2]:
                    return False
        return True

    def can_add(sums_by_k, x, current_size):
        # Try adding x to a set tracked by sums_by_k (for cardinalities 0..current_size)
        # Return (new_sums_by_k, True) if valid, else (None, False)
        # We only need to check up to current_size + 1
        new_sums = [0] * (current_size + 2)
        new_sums[0] = 1
        for k in range(1, current_size + 2):
            # sums from cardinality k in new set:
            # either don't take x (sums_by_k[k]) or take x (sums_by_k[k-1] << x)
            if k <= current_size:
                new_sums[k] = sums_by_k[k] | (sums_by_k[k - 1] << x)
            else:
                # k == current_size + 1
                new_sums[k] = sums_by_k[k - 1] << x
        # Check disjointness only for pairs where at least one is new
        # Full check is O(current_size^2) which is fine
        for k1 in range(current_size + 2):
            if new_sums[k1] == 0:
                continue
            for k2 in range(k1 + 1, current_size + 2):
                if new_sums[k2] == 0:
                    continue
                if new_sums[k1] & new_sums[k2]:
                    return None, False
        return new_sums, True

    # Candidate pool: structured lattice
    # Powers of 2 up to 128, then fill in with other values
    candidates = []
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128
    for i in range(8):
        candidates.append(1 << i)
    # Then add other values in descending order to try to maximize separation
    for v in range(256, 0, -1):
        if v not in candidates:
            candidates.append(v)
    # Actually, let's try multiple orderings and pick the best

    def run_greedy(order):
        sums_by_k = [0] * (max_size + 1)
        sums_by_k[0] = 1
        result = []
        for x in order:
            if len(result) >= max_size:
                break
            new_sums, ok = can_add(sums_by_k, x, len(result))
            if ok:
                result.append(x)
                sums_by_k = new_sums
        return result

    # Order 1: powers of 2 first (ascending), then rest descending
    order1 = [1, 2, 4, 8, 16, 32, 64, 128] + [v for v in range(256, 127, -1) if v not in (1, 2, 4, 8, 16, 32, 64, 128)]
    # Ensure no duplicates and all in 1..256
    seen = set()
    order1_clean = []
    for v in order1:
        if 1 <= v <= 256 and v not in seen:
            seen.add(v)
            order1_clean.append(v)
    result1 = run_greedy(order1_clean)

    # Order 2: descending from 256
    order2 = list(range(256, 0, -1))
    result2 = run_greedy(order2)

    # Order 3: powers of 2 descending, then rest descending
    order3 = [128, 64, 32, 16, 8, 4, 2, 1] + [v for v in range(256, 0, -1) if v not in (1, 2, 4, 8, 16, 32, 64, 128)]
    seen3 = set()
    order3_clean = []
    for v in order3:
        if 1 <= v <= 256 and v not in seen3:
            seen3.add(v)
            order3_clean.append(v)
    result3 = run_greedy(order3_clean)

    # Order 4: geometric progression 1, 3, 9, 27, 81, 243, then fill
    geo = [1, 3, 9, 27, 81, 243]
    order4 = geo + [v for v in range(256, 0, -1) if v not in geo]
    seen4 = set()
    order4_clean = []
    for v in order4:
        if 1 <= v <= 256 and v not in seen4:
            seen4.add(v)
            order4_clean.append(v)
    result4 = run_greedy(order4_clean)

    # Order 5: 1, 2, 4, 8, 16, 32, 64, 128, then 192, 256, then rest
    order5 = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256] + [v for v in range(255, 1, -1) if v not in (1, 2, 4, 8, 16, 32, 64, 128, 192, 256)]
    seen5 = set()
    order5_clean = []
    for v in order5:
        if 1 <= v <= 256 and v not in seen5:
            seen5.add(v)
            order5_clean.append(v)
    result5 = run_greedy(order5_clean)

    best = max([result1, result2, result3, result4, result5], key=len)

    # Ensure return type is list of Python ints
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
