"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute validity for all subsets using efficient bit operations
    # A subset is a clique if for every vertex v in the subset, all other vertices in the subset are adjacent to v
    # A subset is an independent set if for every vertex v in the subset, no other vertices in the subset are adjacent to v

    is_valid = [False] * (1 << n)
    is_valid[0] = True  # empty set is trivially valid

    for mask in range(1, 1 << n):
        # Check if clique: for each vertex v in mask, (mask & ~adj_mask[v] & ~(1<<v)) must be 0
        is_clique = True
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # All other vertices in mask must be adjacent to v
            if (mask & ~(1 << v) & ~adj_mask[v]) != 0:
                is_clique = False
                break
            m ^= lsb

        if is_clique:
            is_valid[mask] = True
        else:
            # Check if independent set: for each vertex v in mask, (mask & ~(1<<v) & adj_mask[v]) must be 0
            is_indep = True
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                # No other vertex in mask should be adjacent to v
                if (mask & ~(1 << v) & adj_mask[v]) != 0:
                    is_indep = False
                    break
                m ^= lsb
            if is_indep:
                is_valid[mask] = True

    # For each vertex, precompute the list of valid subsets that contain it, sorted by size descending
    valid_subsets_for_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if not is_valid[mask]:
            continue
        size = bin(mask).count('1')
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            valid_subsets_for_vertex[v].append((mask, size))
            m ^= lsb

    # Sort each vertex's valid subsets by size descending
    for v in range(n):
        valid_subsets_for_vertex[v].sort(key=lambda x: -x[1])

    best_parts = None
    best_count = n + 1
    nodes_explored = [0]

    def solve(remaining_mask, current_parts):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return

        nodes_explored[0] += 1

        if remaining_mask == 0:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = list(current_parts)
            return

        # Lower bound: at least 1 more part if remaining is non-empty
        if len(current_parts) + 1 >= best_count:
            return

        # Pick the vertex in remaining_mask with the fewest valid subset options
        # This reduces branching factor
        best_vertex = -1
        best_options = None
        best_option_count = float('inf')

        m = remaining_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            options = [mask for mask, size in valid_subsets_for_vertex[v] if (mask & ~remaining_mask) == 0]
            if len(options) < best_option_count:
                best_option_count = len(options)
                best_vertex = v
                best_options = options
                if best_option_count == 0:
                    break
            m ^= lsb

        if best_option_count == 0:
            return

        # Try each valid subset containing best_vertex that is a subset of remaining_mask
        for mask in best_options:
            current_parts.append(mask)
            solve(remaining_mask & ~mask, current_parts)
            current_parts.pop()

            if time.time() > deadline:
                return

    # Start with full_mask
    solve(full_mask, [])

    if best_parts is not None:
        # Convert bitmask parts to vertex lists
        result = []
        for part_mask in best_parts:
            vertices = [i for i in range(n) if part_mask & (1 << i)]
            result.append(vertices)
        return result

    # Fallback: each vertex as its own part (always valid since single vertex is both clique and independent)
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
