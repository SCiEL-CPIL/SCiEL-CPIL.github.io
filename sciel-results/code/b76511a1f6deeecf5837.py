"""Cochromatic partition solver with bitmask DP, greedy, and iterative deepening backtracking."""
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute for every subset whether it is a clique and/or independent set.
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    for v in range(n):
        m = 1 << v
        is_clique[m] = True
        is_indep[m] = True

    # Iterate by popcount to ensure smaller subsets are computed first.
    for popcount in range(2, n + 1):
        for mask in range(1 << n):
            if mask.bit_count() != popcount:
                continue
            if is_clique[mask] or is_indep[mask]:
                continue

            temp = mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                rest = mask ^ lsb

                if is_clique[rest]:
                    if (rest & ~adj[v]) == 0:
                        is_clique[mask] = True
                        break

                if not is_clique[mask] and is_indep[rest]:
                    if (rest & adj[v]) == 0:
                        is_indep[mask] = True
                        break

                temp ^= lsb

    # Precompute valid_part for speed
    valid_part = [False] * (1 << n)
    for mask in range(1, 1 << n):
        valid_part[mask] = is_clique[mask] or is_indep[mask]

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s margin for 90s limit

    # Memoized DP: min_parts[mask] = minimum number of valid parts to partition 'mask'
    # Use -1 as sentinel for uncomputed
    min_parts = [-1] * (1 << n)
    min_parts[0] = 0

    # Precompute list of valid submasks for each mask to speed up iteration
    # For each mask, we need to iterate over valid submasks that contain the least significant bit
    # to avoid duplicate work (symmetry breaking: the part containing the smallest vertex is chosen first)

    def compute_min_parts(mask):
        """Compute minimum number of parts to partition 'mask' using valid parts."""
        if time.time() > deadline:
            return None
        if min_parts[mask] != -1:
            return min_parts[mask]

        # Find the least significant set bit - this vertex must be in some part
        lsb = mask & (-mask)

        best = n + 1  # Upper bound

        # Iterate over all valid submasks of mask that contain lsb
        # A submask S of mask contains lsb if S & lsb != 0
        # We need S to be a valid part (clique or independent set)
        sub = mask
        while sub:
            if sub & lsb:
                if valid_part[sub]:
                    rest = mask ^ sub
                    if rest == 0:
                        candidate = 1
                    else:
                        r = compute_min_parts(rest)
                        if r is None:
                            return None
                        candidate = 1 + r
                    if candidate < best:
                        best = candidate
                        if best == 1:
                            break
            sub = (sub - 1) & mask

        min_parts[mask] = best
        return best

    # Compute the answer
    answer = compute_min_parts(full_mask)

    if answer is None or answer > n:
        # Fallback to greedy if DP timed out or failed
        def greedy_partition():
            parts = []
            for v in range(n):
                vmask = 1 << v
                placed = False
                for i, part in enumerate(parts):
                    trial = part | vmask
                    if valid_part[trial]:
                        parts[i] = trial
                        placed = True
                        break
                if not placed:
                    parts.append(vmask)
            return parts

        initial_parts = greedy_partition()
        best_num_parts = len(initial_parts)
        best_partition = initial_parts[:]
    else:
        # Reconstruct the partition from the DP table
        best_num_parts = answer
        # We need to reconstruct which parts were chosen
        # Store parent pointers during DP
        parent = [-1] * (1 << n)
        parent[0] = 0

        # Recompute with parent tracking
        min_parts2 = [-1] * (1 << n)
        min_parts2[0] = 0

        def compute_min_parts_with_parent(mask):
            if time.time() > deadline:
                return None
            if min_parts2[mask] != -1:
                return min_parts2[mask]

            lsb = mask & (-mask)
            best = n + 1
            best_sub = -1

            sub = mask
            while sub:
                if sub & lsb:
                    if valid_part[sub]:
                        rest = mask ^ sub
                        if rest == 0:
                            candidate = 1
                        else:
                            r = compute_min_parts_with_parent(rest)
                            if r is None:
                                return None
                            candidate = 1 + r
                        if candidate < best:
                            best = candidate
                            best_sub = sub
                            if best == 1:
                                break
                sub = (sub - 1) & mask

            min_parts2[mask] = best
            if best_sub != -1:
                parent[mask] = best_sub
            return best

        compute_min_parts_with_parent(full_mask)

        # Reconstruct
        best_partition = []
        mask = full_mask
        while mask:
            sub = parent[mask]
            if sub == -1:
                # Fallback: just make each vertex its own part
                break
            # Extract vertices from sub
            verts = []
            temp = sub
            while temp:
                lsb2 = temp & (-temp)
                verts.append(lsb2.bit_length() - 1)
                temp ^= lsb2
            best_partition.append(verts)
            mask ^= sub

        # If reconstruction failed, use greedy
        if not best_partition:
            def greedy_partition():
                parts = []
                for v in range(n):
                    vmask = 1 << v
                    placed = False
                    for i, part in enumerate(parts):
                        trial = part | vmask
                        if valid_part[trial]:
                            parts[i] = trial
                            placed = True
                            break
                    if not placed:
                        parts.append(vmask)
                return parts
            best_partition = greedy_partition()

    # Ensure types are correct: Python int, list
    final_result = []
    for part in best_partition:
        if isinstance(part, int):
            # It's a bitmask, convert to list
            verts = []
            temp = part
            while temp:
                lsb = temp & (-temp)
                verts.append(int(lsb.bit_length() - 1))
                temp ^= lsb
            final_result.append(verts)
        else:
            # Already a list
            final_result.append([int(v) for v in part])

    return final_result

# EVOLVE-BLOCK-END
