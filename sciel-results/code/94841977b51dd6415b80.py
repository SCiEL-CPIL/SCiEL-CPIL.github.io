"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    DEADLINE = 80.0

    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)

    # dp[i] = list of (last_gap, length) for Pareto-optimal subsequences ending at primes[i]
    # We also need to reconstruct the path, so we store parent info
    # For efficiency, store for each i a sorted list by last_gap with max length

    # dp[i][k] = (last_gap, length, prev_index) where prev_index is index of previous prime in subsequence
    # Actually, let's store: for each i, a list of entries (gap, length, prev_j)
    # where gap is the last gap, length is total length, prev_j is the index of the previous prime

    # To make reconstruction work, we need to track the full path.
    # Let's use: best[i] = list of (gap, length, prev_j) Pareto-optimal
    # Then to reconstruct, we follow prev_j pointers.

    best = [[] for _ in range(n)]
    parent = [[None] * n for _ in range(n)]  # parent[i][j] = index of prev prime when best[i] entry came from j? No...

    # Simpler: store for each i, a list of dicts or tuples
    # Let me use: seqs[i] = list of (last_gap, length, prev_idx)
    # where prev_idx is the index in primes of the previous element

    seqs = [[] for _ in range(n)]
    # For reconstruction, we need to know for each (i, entry) what the previous prime index is
    # Let's store: for each i, a list of (last_gap, length, prev_prime_idx)

    # To extend: for prime i, for each j < i:
    #   gap = primes[i] - primes[j]
    #   For each entry in seqs[j] with last_gap <= gap, we can extend to get length+1
    #   Also, we can start a new subsequence of length 2 (just primes[j], primes[i])

    # Optimization: for each j, pre-sort seqs[j] by last_gap and compute prefix max length
    # Then for a given gap, binary search to find max length with last_gap <= gap

    from bisect import bisect_right

    # For each j, maintain sorted list of (last_gap, max_length_so_far)
    # Actually, let's just process and keep Pareto frontier

    # Initialize: each prime alone is a subsequence of length 1 with no gap
    # But we need at least 2 elements to have a gap. Let's handle length 1 separately.

    # Let's redefine: seqs[i] contains entries for subsequences of length >= 2 ending at i
    # Each entry: (last_gap, length, prev_idx)

    # For length 1, we don't store anything special; we just know it exists.

    # Process primes in order
    for i in range(n):
        if time.time() - t0 >= DEADLINE:
            break
        pi = primes[i]
        new_entries = []
        for j in range(i):
            if time.time() - t0 >= DEADLINE:
                break
            pj = primes[j]
            gap = pi - pj
            # Can start new subsequence of length 2
            new_entries.append((gap, 2, j))
            # Can extend existing subsequences ending at j
            for (lg, ln, _) in seqs[j]:
                if lg <= gap:
                    new_entries.append((gap, ln + 1, j))
        # Now reduce new_entries to Pareto frontier:
        # Sort by last_gap, then for each gap keep only max length
        # Then remove entries where a later entry has >= length and >= gap (dominated)
        if new_entries:
            # Group by gap, keep max length for each gap
            from collections import defaultdict
            gap_to_maxlen = {}
            gap_to_prev = {}
            for (g, l, pidx) in new_entries:
                if g not in gap_to_maxlen or l > gap_to_maxlen[g]:
                    gap_to_maxlen[g] = l
                    gap_to_prev[g] = pidx
            # Sort by gap
            sorted_gaps = sorted(gap_to_maxlen.keys())
            # Pareto frontier: increasing gap, decreasing length (keep only non-dominated)
            # An entry (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2
            # After sorting by gap ascending, we want strictly decreasing lengths
            frontier = []
            max_len_so_far = 0
            # Process in reverse order of gap to find non-dominated
            # Actually: sort by gap ascending, then traverse and keep entry if its length > all lengths of entries with larger gap
            # Better: sort by gap ascending, then from right to left, keep if length > max seen to the right
            # Let's do: sort by gap ascending, then compute suffix max of lengths
            # Keep entry at position k if length[k] > suffix_max[k+1]
            lengths = [gap_to_maxlen[g] for g in sorted_gaps]
            prevs = [gap_to_prev[g] for g in sorted_gaps]
            m = len(sorted_gaps)
            suffix_max = [0] * (m + 1)
            for k in range(m - 1, -1, -1):
                suffix_max[k] = max(lengths[k], suffix_max[k + 1])
            for k in range(m):
                # Keep if this length is strictly greater than all lengths to its right
                # i.e., length[k] > suffix_max[k+1]
                if k == m - 1 or lengths[k] > suffix_max[k + 1]:
                    frontier.append((sorted_gaps[k], lengths[k], prevs[k]))
            seqs[i] = frontier

    # Find the best: max length over all i
    best_len = 1
    best_i = 0
    best_entry = None
    for i in range(n):
        for (g, l, pidx) in seqs[i]:
            if l > best_len:
                best_len = l
                best_i = i
                best_entry = (g, l, pidx)

    # Reconstruct the sequence
    if best_len == 1:
        return [primes[0]]

    # Follow the chain
    result = [primes[best_i]]
    cur_i = best_i
    cur_entry = best_entry
    while cur_entry is not None and len(result) < best_len:
        prev_idx = cur_entry[2]
        result.append(primes[prev_idx])
        # Find the entry in seqs[prev_idx] that was used
        # We need to know which entry in seqs[prev_idx] has last_gap == the gap before this one
        # The gap before was: primes[cur_i] - primes[prev_idx]
        # But we need to find the specific entry. This is tricky.

        # Let me restructure: store more info. Actually, let's store for each entry in seqs[i]
        # the specific entry index in seqs[prev_idx] that it came from.

        # This is getting complex. Let me restart with a cleaner approach.
        break

    # Given complexity, let me use a simpler DP with full reconstruction tracking
    # Let me redo this more carefully

    # Reset
    # dp[i] = list of (last_gap, length, prev_prime_idx, prev_entry_idx)
    # where prev_entry_idx is the index into dp[prev_prime_idx] of the entry we extended

    dp = [[] for _ in range(n)]

    for i in range(n):
        if time.time() - t0 >= DEADLINE:
            break
        pi = primes[i]
        new_entries = []
        for j in range(i):
            if time.time() - t0 >= DEADLINE:
                break
            gap = pi - primes[j]
            # New subsequence of length 2
            new_entries.append((gap, 2, j, -1))  # -1 means no previous entry
            # Extend existing
            for k, (lg, ln, pj, pk) in enumerate(dp[j]):
                if lg <= gap:
                    new_entries.append((gap, ln + 1, j, k))
        # Reduce to Pareto frontier
        if new_entries:
            gap_to_best = {}
            for (g, l, pidx, eidx) in new_entries:
                if g not in gap_to_best or l > gap_to_best[g][0]:
                    gap_to_best[g] = (l, pidx, eidx)
            sorted_gaps = sorted(gap_to_best.keys())
            lengths = [gap_to_best[g][0] for g in sorted_gaps]
            m = len(sorted_gaps)
            suffix_max = [0] * (m + 1)
            for k in range(m - 1, -1, -1):
                suffix_max[k] = max(lengths[k], suffix_max[k + 1])
            frontier = []
            for k in range(m):
                if k == m - 1 or lengths[k] > suffix_max[k + 1]:
                    g = sorted_gaps[k]
                    l, pidx, eidx = gap_to_best[g]
                    frontier.append((g, l, pidx, eidx))
            dp[i] = frontier

    # Find best
    best_len = 1
    best_i = 0
    best_eidx = -1
    for i in range(n):
        for k, (g, l, pidx, eidx) in enumerate(dp[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_eidx = k

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]

    result = []
    cur_i = best_i
    cur_k = best_eidx
    while True:
        result.append(primes[cur_i])
        if cur_k == -1:
            break
        g, l, pidx, eidx = dp[cur_i][cur_k]
        cur_i = pidx
        cur_k = eidx
    result.reverse()
    return result

# EVOLVE-BLOCK-END
