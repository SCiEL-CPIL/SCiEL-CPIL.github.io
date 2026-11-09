"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    import time
    n = P['n']
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True

    deadline = time.time() + 85.0

    # Precompute for each subset whether it is a clique or independent set
    # Using bitmask representation
    # For each vertex, adjacency bitmask
    adj_mask = [0] * n
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)

    # Check if subset (bitmask) is clique: for all u in subset, all other v in subset are neighbors of u
    # Check if subset is independent set: for all u in subset, no other v in subset is neighbor of u

    def is_clique(mask):
        while mask:
            u_bit = mask & (-mask)
            u = u_bit.bit_length() - 1
            mask ^= u_bit
            # All remaining vertices must be neighbors of u
            if mask & ~adj_mask[u]:
                return False
        return True

    def is_independent(mask):
        while mask:
            u_bit = mask & (-mask)
            u = u_bit.bit_length() - 1
            mask ^= u_bit
            # No remaining vertex should be a neighbor of u
            if mask & adj_mask[u]:
                return False
        return True

    def is_valid_part(mask):
        return is_clique(mask) or is_independent(mask)

    # Backtracking search
    best_parts = None
    best_count = n  # worst case: each vertex its own part

    # We'll assign vertices in order 0, 1, ..., n-1
    # State: list of parts, each part is a bitmask
    parts = []  # list of bitmasks

    def backtrack(idx, current_parts, count):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return

        if idx == n:
            if count < best_count:
                best_count = count
                best_parts = [list(current_parts)]
            return

        v = idx
        v_bit = 1 << v

        # Try adding v to each existing part
        for i in range(len(current_parts)):
            new_mask = current_parts[i] | v_bit
            if is_valid_part(new_mask):
                current_parts[i] = new_mask
                backtrack(idx + 1, current_parts, count)
                current_parts[i] ^= v_bit  # undo

        # Try starting a new part with v
        # Only do this if count + 1 < best_count (pruning)
        if count + 1 < best_count:
            current_parts.append(v_bit)
            backtrack(idx + 1, current_parts, count + 1)
            current_parts.pop()

    backtrack(0, [], 0)

    if best_parts is None:
        # Fallback: greedy or individual parts
        return [[v] for v in range(n)]

    # Convert bitmasks back to lists of vertex indices
    result = []
    for mask in best_parts[0]:
        vertices = []
        m = mask
        while m:
            bit = m & (-m)
            v = bit.bit_length() - 1
            vertices.append(v)
            m ^= bit
        result.append(vertices)

    return result

# EVOLVE-BLOCK-END
