def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    p = 1009
    n = 1008

    # Precompute modular inverses mod p for x in [1, p-1]
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)

    # For each x, the conflict threshold is max(x, inv[x])
    # For a given start s, we need: e < min{max(x, inv[x]) : x in [s,n], inv[x] >= s}
    #
    # As s increases from 1 to n:
    # - Elements x drop out when x < s (but we iterate s upward, so x >= s is maintained)
    # - Elements x drop out when inv[x] < s
    #
    # We can use a min-heap of (threshold, x) and remove entries where inv[x] < s
    # or x < s (though x < s won't happen since we add x >= current s)
    #
    # Actually, let me think differently. For each s from 1 to n:
    #   Valid x are those with x >= s AND inv[x] >= s
    #   The constraint is e < min{max(x, inv[x]) : valid x}
    #
    # As s increases, some x become invalid (when inv[x] < s).
    # We can maintain a heap and lazily remove invalid entries.

    import heapq

    # Initialize: for s=1, all x in [1,n] are valid (inv[x] >= 1 always)
    # Build initial heap with (max(x, inv[x]), x) for all x
    heap = []
    for x in range(1, n + 1):
        threshold = max(x, inv[x])
        heapq.heappush(heap, (threshold, x))

    best_start = 2
    best_length = 0

    for s in range(1, n + 1):
        if time.time() > deadline:
            break

        # Remove entries where inv[x] < s (x is no longer valid for this s)
        # Also remove entries where x < s (shouldn't happen but be safe)
        while heap:
            threshold, x = heap[0]
            if x < s or inv[x] < s:
                heapq.heappop(heap)
            else:
                break

        # The minimum threshold in the heap gives the constraint
        if heap:
            min_threshold = heap[0][0]
            max_valid_e = min_threshold - 1
        else:
            max_valid_e = n  # no constraint

        if max_valid_e >= s:
            length = max_valid_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    # Verify the result
    e_final = best_start + best_length - 1
    valid = True
    for x in range(best_start, e_final + 1):
        if best_start <= inv[x] <= e_final:
            valid = False
            break

    if not valid:
        # Fallback: find a valid interval by brute force
        for s in range(1, n + 1):
            for e in range(n, s - 1, -1):
                ok = True
                for x in range(s, e + 1):
                    if s <= inv[x] <= e:
                        ok = False
                        break
                if ok:
                    best_start = s
                    best_length = e - s + 1
                    break
            if best_length > 0:
                break

    return [best_start, best_length]
