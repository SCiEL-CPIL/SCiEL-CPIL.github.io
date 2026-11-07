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
    # is_clique[mask]: mask is a non-empty clique
    # is_indep[mask]: mask is a non-empty independent set
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    for v in range(n):
        m = 1 << v
        is_clique[m] = True
        is_indep[m] = True

    # For a mask, pick the least significant set bit.
    # A set S is a clique iff for some v in S, S\{v} is a clique and v is adjacent to all in S\{v}.
    # A set S is an independent set iff for some v in S, S\{v} is an independent set and v is not adjacent to any in S\{v}.
    # We iterate by popcount to ensure smaller subsets are computed first.
    for popcount in range(2, n + 1):
        for mask in range(1 << n):
            if mask.bit_count() != popcount:
                continue
            if is_clique[mask] or is_indep[mask]:
                continue

            # Try removing each vertex
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

    # Greedy initial solution
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

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 75  # 75 seconds, leaving 15s margin for 90s limit

    # Vertex ordering by degree (descending) for better pruning
    vertex_order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))

    # Backtracking search for cochromatic number
    def backtrack_search(k_max):
        """Try to find a valid partition with at most k_max parts."""
        parts = [0] * k_max
        assigned_mask = 0

        # Precompute for each vertex and each part slot, which vertices can be added
        # Actually, we'll check on the fly using valid_part

        def search(idx):
            nonlocal assigned_mask
            if time.time() > deadline:
                return None

            if idx == n:
                # All vertices assigned
                result = []
                for p in parts:
                    if p:
                        verts = []
                        temp = p
                        while temp:
                            lsb = temp & (-temp)
                            verts.append(lsb.bit_length() - 1)
                            temp ^= lsb
                        result.append(verts)
                return result

            v = vertex_order[idx]
            vmask = 1 << v

            # Try placing v in each existing part
            for i in range(k_max):
                if parts[i] == 0:
                    # Empty part slot - only try the first one to avoid symmetry
                    parts[i] = vmask
                    res = search(idx + 1)
                    if res is not None:
                        return res
                    parts[i] = 0
                else:
                    trial = parts[i] | vmask
                    if valid_part[trial]:
                        parts[i] = trial
                        res = search(idx + 1)
                        if res is not None:
                            return res
                        parts[i] = trial ^ vmask

            # Try creating a new part (use first empty slot to break symmetry)
            for i in range(k_max):
                if parts[i] == 0:
                    parts[i] = vmask
                    res = search(idx + 1)
                    if res is not None:
                        return res
                    parts[i] = 0
                    break

            return None

        return search(0)

    # Try smaller numbers of parts, starting from best_num_parts - 1
    for k in range(best_num_parts - 1, 0, -1):
        if time.time() > deadline:
            break
        result = backtrack_search(k)
        if result is not None:
            best_num_parts = k
            best_partition = result
            break

    # Convert best_partition to list of lists of Python ints
    final_parts = []
    for p in best_partition:
        verts = []
        temp = p
        while temp:
            lsb = temp & (-temp)
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
        final_parts.append(verts)

    # Ensure types are correct: Python int, list
    final_result = []
    for part in final_parts:
        final_result.append([int(v) for v in part])

    return final_result

# EVOLVE-BLOCK-END
