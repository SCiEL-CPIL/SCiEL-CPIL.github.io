"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85.0

    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Non-adjacency (excluding self)
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask ^ adj_mask[v] ^ (1 << v)

    # Incremental DP for clique and independent set detection
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest] and (rest & ~adj_mask[v]) == 0:
            is_clique[mask] = True

        if is_indep[rest] and (rest & ~non_adj[v]) == 0:
            is_indep[mask] = True

    # Valid part: clique or independent set
    valid = [False] * (1 << n)
    for mask in range(1 << n):
        valid[mask] = is_clique[mask] or is_indep[mask]

    # Max size of any valid part
    max_part_size = 0
    for mask in range(1, 1 << n):
        if valid[mask]:
            sz = bin(mask).count('1')
            if sz > max_part_size:
                max_part_size = sz

    # For each vertex, precompute valid subsets containing it (as masks), sorted by size desc
    valid_subsets_for_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if not valid[mask]:
            continue
        sz = bin(mask).count('1')
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            valid_subsets_for_vertex[v].append((mask, sz))
            m ^= lsb

    for v in range(n):
        valid_subsets_for_vertex[v].sort(key=lambda x: -x[1])

    best_parts = None
    best_count = n + 1
    nodes = [0]

    def solve(remaining, current_parts):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return
        nodes[0] += 1

        if remaining == 0:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = list(current_parts)
            return

        # Pruning: need at least 1 more part
        if len(current_parts) + 1 >= best_count:
            return

        # Pruning: remaining vertices must fit in (best_count - 1 - len(current_parts)) parts
        remaining_count = bin(remaining).count('1')
        max_parts_allowed = best_count - 1 - len(current_parts)
        if max_parts_allowed < 1:
            return
        if remaining_count > max_parts_allowed * max_part_size:
            return

        # Pick vertex with fewest valid subset options within remaining
        best_vertex = -1
        best_options = None
        best_count_opts = float('inf')

        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            options = [mask for mask, sz in valid_subsets_for_vertex[v] if (mask & ~remaining) == 0]
            cnt = len(options)
            if cnt < best_count_opts:
                best_count_opts = cnt
                best_vertex = v
                best_options = options
                if cnt == 0:
                    break
            m ^= lsb

        if best_count_opts == 0:
            return

        # Try each valid subset containing best_vertex that fits in remaining
        for mask in best_options:
            current_parts.append(mask)
            solve(remaining & ~mask, current_parts)
            current_parts.pop()
            if time.time() > deadline:
                return

    solve(full_mask, [])

    if best_parts is not None:
        result = []
        for part_mask in best_parts:
            vertices = [i for i in range(n) if part_mask & (1 << i)]
            result.append(vertices)
        return result

    # Fallback: each vertex as its own part
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
