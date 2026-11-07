import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144

    # Precompute: for each i in 1..12 and each residue r in 0..i-1,
    # the bitmask of d in 1..max_L where d % i == r
    full_mask = {}
    for i in range(1, n + 1):
        full_mask[i] = [0] * i
        for d in range(1, max_L + 1):
            rem = d % i
            full_mask[i][rem] |= (1 << d)

    # Precompute bit_count for masks up to 144 bits
    # We'll use int.bit_count() which is very fast in Python 3.8+

    def has_perfect_matching(m, L):
        """Check if (m, m+L] has a perfect matching for distinct multiples."""
        if L < n:
            return False

        # For each i, compute the valid d's up to L as a bitmask
        adj_masks = [0] * n  # adj_masks[i-1] = bitmask of valid d's for left node i
        counts = [0] * n
        union_mask = 0

        for i in range(1, n + 1):
            rem = (-m) % i
            mask = full_mask[i][rem] & ((1 << (L + 1)) - 1)  # mask d in 1..L
            adj_masks[i - 1] = mask
            cnt = mask.bit_count()
            counts[i - 1] = cnt
            union_mask |= mask

        # Quick check: any node with 0 options?
        for c in counts:
            if c == 0:
                return False

        # Hall's condition quick check: union of all neighbors must have size >= n
        if union_mask.bit_count() < n:
            return False

        # Stronger Hall checks: for subsets of size 2 and 3
        # Check all pairs
        for a in range(n):
            for b in range(a + 1, n):
                if (adj_masks[a] | adj_masks[b]).bit_count() < 2:
                    return False

        # Order left nodes by count (ascending) - most constrained first
        order = sorted(range(n), key=lambda idx: counts[idx])

        # Use DFS matching with bitset-based neighbor iteration
        # matchR[d] = left node index (0-based) that d is matched to, or -1
        matchR = [-1] * (L + 1)  # 1-indexed for d

        def dfs(idx, seen_mask):
            # idx is the original left node index (0-based)
            # seen_mask is a bitmask of already-tried d's in this DFS call
            mask = adj_masks[idx] & ~seen_mask
            while mask:
                # Get the lowest set bit
                d = (mask & (-mask)).bit_length() - 1
                mask &= mask - 1  # clear the lowest set bit
                if matchR[d] == -1:
                    matchR[d] = idx
                    return True
                # Recurse on the node currently matched to d
                if dfs(matchR[d], seen_mask | (1 << d)):
                    matchR[d] = idx
                    return True
            return False

        for orig_idx in order:
            seen_mask = 0
            if not dfs(orig_idx, seen_mask):
                return False
        return True

    best_m = 0
    best_L = 0

    # Phase 1: Scan all m values in one period with binary search
    for m in range(0, period):
        if time.time() - start_time > deadline - 10.0:
            break

        # Binary search for minimum L
        lo, hi = 1, max_L
        min_L_for_m = max_L

        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                min_L_for_m = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if min_L_for_m > best_L:
            best_L = min_L_for_m
            best_m = m

        # Early termination if we hit the theoretical max
        if best_L >= max_L:
            break

    # Phase 2: Local refinement around best_m with wider window
    if time.time() - start_time < deadline - 2.0:
        window = 200
        for dm in range(-window, window + 1):
            if dm == 0:
                continue
            m2 = (best_m + dm) % period
            if m2 < 0:
                m2 += period

            lo, hi = 1, max_L
            min_L_for_m2 = max_L
            while lo <= hi:
                mid = (lo + hi) // 2
                if has_perfect_matching(m2, mid):
                    min_L_for_m2 = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            if min_L_for_m2 > best_L:
                best_L = min_L_for_m2
                best_m = m2

    # Phase 3: Final polish - check a few more candidates if time allows
    if time.time() - start_time < deadline - 1.0 and best_L < max_L:
        # Try some structurally interesting m values
        candidates = set()
        for offset in range(0, period, 720):  # 27720/38.5 ~ every 720
            candidates.add(offset)
        for offset in range(0, period, 360):
            candidates.add(offset)
        candidates.add(best_m)

        for m2 in sorted(candidates):
            if time.time() - start_time > deadline - 0.5:
                break
            if m2 == best_m:
                continue

            lo, hi = 1, max_L
            min_L_for_m2 = max_L
            while lo <= hi:
                mid = (lo + hi) // 2
                if has_perfect_matching(m2, mid):
                    min_L_for_m2 = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            if min_L_for_m2 > best_L:
                best_L = min_L_for_m2
                best_m = m2

    # Verify best_m is in range
    if best_m < 0 or best_m > 1000000:
        best_m = 0

    return int(best_m)
