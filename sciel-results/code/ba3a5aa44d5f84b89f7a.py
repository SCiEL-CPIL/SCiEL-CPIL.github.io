import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115
    N = 1000
    MAXC = 2  # max representations allowed

    # ---------- bit utilities ----------
    # We represent the set S as a Python int bitmask: bit i set means i in S.
    # For a candidate x, the "conflict mask" tells which existing y would make
    # counts[x+y] exceed MAXC if x were added.
    # Precompute for each x: for each existing y, the sum s=x+y.
    # We need: for all y in S, counts[x+y] < MAXC (i.e., <= MAXC-1).
    # This is equivalent to: the set of y in S where counts[x+y] >= MAXC must be empty.
    # Represent "forbidden y for candidate x" as a bitmask forbidden[x].
    # Then x is addable iff (S_mask & forbidden[x]) == 0 AND counts[2x] < MAXC.

    # We'll maintain counts as a numpy array for speed in batch ops,
    # but the feasibility check uses the bitmask trick.

    def make_forbidden(x, counts):
        """Return bitmask of y in 1..N such that counts[x+y] >= MAXC."""
        # y ranges 1..N, sum x+y ranges x+1 .. x+N
        # forbidden bit y is set if counts[x+y] >= MAXC
        # Build as numpy bool array then pack into int
        sums = x + np.arange(1, N + 1)  # x+1 .. x+N
        bad = counts[sums] >= MAXC
        # Pack bool array into int bitmask
        # bit i corresponds to value i+1 (so bit 0 = value 1)
        # Use numpy to pack: treat as bytes
        # For N=1000, we need 1000 bits = 125 bytes
        # Simple approach: iterate (slow) or use numpy pack
        # Faster: convert to uint8 bytes and use int.from_bytes
        # But we need bit i to represent value i+1
        # bad[i] corresponds to y=i+1, so bit position i
        # int.from_bytes expects little-endian by default
        # Let's use a simple loop-free approach with numpy
        # Actually for 1000 bits, let's just use a list comprehension or numpy pack
        # numpy doesn't have direct bool->int bitmask, so:
        # Convert to uint8, then to bytes, then to int
        # bad is shape (N,), bool
        # We want bit i of the result to be bad[i]
        # int.from_bytes(b, 'little') gives bit j = b[j//8] & (1 << (j%8))
        # So we need b[i//8] to have bit (i%8) set iff bad[i]
        # This is exactly what numpy's packbits does (MSB first per byte)
        # Let's just do it with a simple approach:
        # For 1000 elements, this is fast enough
        if not bad.any():
            return 0
        # Pack: create uint8 array of length ceil(N/8)
        n_bytes = (N + 7) // 8
        # bad is bool array of length N
        # Reshape to (n_bytes, 8) with padding
        padded = np.zeros(n_bytes * 8, dtype=np.uint8)
        padded[:N] = bad.astype(np.uint8)
        padded = padded.reshape(n_bytes, 8)
        # numpy packbits packs MSB first: bit 7 of byte = first element
        # We want bit 0 of byte = first element (value 1)
        # So we need to reverse bits in each byte
        # Reverse the 8 elements in each row
        padded = padded[:, ::-1]
        # Now bit 0 of each byte = original element 0
        # Use int.from_bytes on the flat array
        b = padded.flatten().tobytes()
        return int.from_bytes(b, 'little')

    def bit_count(x):
        return bin(x).count('1')

    # ---------- greedy construction ----------
    def greedy_random(seed, time_limit=110):
        rng = random.Random(seed)
        mask = np.zeros(N + 1, dtype=bool)
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        S_mask = 0  # Python int bitmask
        size = 0

        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)

        for x in candidates:
            if time.time() > time_limit:
                break
            # Check if x is addable
            # Forbidden: y in S where counts[x+y] >= MAXC
            # Also check counts[2x] < MAXC
            if counts[2 * x] >= MAXC:
                continue
            # Get forbidden mask for x
            # For speed, we check: for each y in S, is counts[x+y] >= MAXC?
            # Using bitmask:
            forbidden = make_forbidden(x, counts)
            if (S_mask & forbidden) != 0:
                continue
            # Add x
            # Update counts: for each y in S, counts[x+y] += 1; counts[2x] += 1
            present = np.where(mask)[0]
            present = present[present >= 1]
            if len(present) > 0:
                counts[x + present] += 1
            counts[2 * x] += 1
            mask[x] = True
            S_mask |= (1 << x)
            size += 1

        return mask, counts, size

    # ---------- local search with bitmask feasibility ----------
    def local_search(mask, counts, seed, time_budget):
        rng = random.Random(seed)
        size = int(np.sum(mask))
        S_mask = 0
        for i in range(1, N + 1):
            if mask[i]:
                S_mask |= (1 << i)

        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < time_budget:
            improved = False

            # Try adds
            absent = [x for x in range(1, N + 1) if not mask[x]]
            if absent:
                rng.shuffle(absent)
                for x in absent:
                    if counts[2 * x] >= MAXC:
                        continue
                    forbidden = make_forbidden(x, counts)
                    if (S_mask & forbidden) != 0:
                        continue
                    # Add x
                    present = np.where(mask)[0]
                    present = present[present >= 1]
                    if len(present) > 0:
                        counts[x + present] += 1
                    counts[2 * x] += 1
                    mask[x] = True
                    S_mask |= (1 << x)
                    size += 1
                    improved = True
                    break

            if improved:
                continue

            # Try swaps
            present_list = [x for x in range(1, N + 1) if mask[x]]
            absent_list = [x for x in range(1, N + 1) if not mask[x]]
            if not present_list or not absent_list:
                break

            # Try a few random swaps
            for _ in range(100):
                a = present_list[rng.randrange(len(present_list))]
                b = absent_list[rng.randrange(len(absent_list))]

                # Check swap validity
                # After removing a: counts decrease for a+y (y in S\{a}) and 2a
                # After adding b: counts increase for b+y (y in S\{a}) and 2b
                # Net: for y in S\{a}: count[a+y]-1, count[b+y]+1; count[2a]-1, count[2b]+1
                # Need all resulting counts <= MAXC
                # Only increases matter: b+y for y in S\{a}, and 2b
                # But also: when we remove a, counts[a+y] decrease, which is fine.
                # However, we need to check that b+y doesn't exceed MAXC AFTER the a removal
                # Since a removal only decreases counts, the max count for b+y after
                # removal is at most the current count. So checking current counts[b+y] < MAXC
                # is sufficient (since removal can only help).
                # Wait: counts[b+y] currently includes contributions from a if a is in S.
                # When we remove a, counts[a+b] decreases by 1 (if a != b, which it isn't since a in S, b not).
                # So counts[b+a] = counts[a+b] decreases by 1.
                # For y != a: counts[b+y] is unchanged by removing a (unless b+y == a+y' for some y',
                # but that's a different sum).
                # Actually, removing a only affects sums involving a: a+y for y in S, and 2a.
                # Adding b affects sums b+y for y in S\{a} (since a is removed), and 2b.
                # The sum a+b: when a is removed, count[a+b] -= 1. When b is added, count[b+a] += 1
                # but a is no longer in S, so this doesn't happen.
                # So net for sum a+b: -1 (from removing a) + 0 (b added, a not in S) = -1. Fine.
                # For sum b+y where y in S\{a}: +1 from adding b. No change from removing a
                # (unless y=a, but y in S\{a}).
                # So we need: for all y in S\{a}, counts[b+y] + 1 <= MAXC, i.e., counts[b+y] < MAXC.
                # And counts[2b] + 1 <= MAXC, i.e., counts[2b] < MAXC.
                # Also need to ensure that after removal, no count goes negative (impossible since
                # we only remove valid contributions).

                valid = True
                # Check counts[2b]
                if counts[2 * b] >= MAXC:
                    valid = False
                else:
                    # Check counts[b+y] for y in S\{a}
                    present_no_a = [y for y in present_list if y != a]
                    for y in present_no_a:
                        if counts[b + y] >= MAXC:
                            valid = False
                            break

                if valid:
                    # Perform swap
                    # Remove a
                    present_no_a = [y for y in present_list if y != a]
                    if present_no_a:
                        counts[a + np.array(present_no_a)] -= 1
                    counts[2 * a] -= 1
                    mask[a] = False
                    S_mask &= ~(1 << a)

                    # Add b
                    if present_no_a:
                        counts[b + np.array(present_no_a)] += 1
                    counts[2 * b] += 1
                    mask[b] = True
                    S_mask |= (1 << b)

                    improved = True
                    break

        return mask, counts, size

    # ---------- main ----------
    best_size = 0
    best_mask = None
    best_counts = None

    # Deterministic greedy baseline
    mask_det = np.zeros(N + 1, dtype=bool)
    counts_det = np.zeros(2 * N + 1, dtype=np.int32)
    size_det = 0
    S_mask_det = 0
    for x in range(1, N + 1):
        if counts_det[2 * x] >= MAXC:
            continue
        forbidden = make_forbidden(x, counts_det)
        if (S_mask_det & forbidden) != 0:
            continue
        present = np.where(mask_det)[0]
        present = present[present >= 1]
        if len(present) > 0:
            counts_det[x + present] += 1
        counts_det[2 * x] += 1
        mask_det[x] = True
        S_mask_det |= (1 << x)
        size_det += 1

    best_size = size_det
    best_mask = mask_det.copy()
    best_counts = counts_det.copy()

    # Multi-restart random greedy + local search
    seed = 0
    while time.time() < deadline - 10:
        time_left = deadline - time.time() - 5
        if time_left < 3:
            break
        mask_r, counts_r, size_r = greedy_random(seed, time.time() + min(10, time_left * 0.4))

        # Local search with remaining time
        ls_time = min(30, time_left * 0.6)
        if ls_time > 1:
            mask_r, counts_r, size_r = local_search(mask_r, counts_r, seed, ls_time)

        if size_r > best_size:
            best_size = size_r
            best_mask = mask_r.copy()
            best_counts = counts_r.copy()

        seed += 1

    # Final local search on best
    time_left = deadline - time.time() - 2
    if time_left > 1:
        best_mask, best_counts, best_size = local_search(best_mask, best_counts, seed, time_left)

    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result
