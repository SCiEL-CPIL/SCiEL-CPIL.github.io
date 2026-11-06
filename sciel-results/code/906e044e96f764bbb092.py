"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140.0  # safety margin

    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    # Base: n=32 construction
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    best_n = 32
    best_colors = base_colors[:]

    def edge_index(i, j, n):
        """Index of edge (i,j) in lex-ordered list for n vertices, i<j."""
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def verify(colors, n):
        """Verify no monochromatic triangle exists."""
        if time.time() > deadline:
            return False
        # Build adjacency bitmasks for each color
        adj = [0] * (4 * n)  # adj[c*n + i] is bitmask of neighbors of i in color c
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
                idx += 1
        for c in range(4):
            for i in range(n):
                neighbors = adj[c * n + i]
                # Check if any two neighbors of i in color c are also connected in color c
                for j in range(n):
                    if neighbors & (1 << j):
                        # Check if j's color-c neighbors include any vertex > j that is also neighbor of i
                        common = neighbors & adj[c * n + j] & (~((1 << (j + 1)) - 1))
                        if common:
                            return False
        return True

    def try_add_vertex(colors, n):
        """Try to add a new vertex n with backtracking."""
        # Precompute for each existing vertex i and color c:
        # which vertices j (j>i) have edge (i,j) in color c
        # For new vertex v=n, we assign colors to edges (v, 0), (v, 1), ..., (v, n-1)
        # Constraint: for each color c, the set S_c = {i : edge(v,i)=c} must be independent in color-c graph
        # i.e., for any i<j in S_c, edge(i,j) != c

        # Precompute color-c adjacency as bitmasks
        adj_c = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_c[c][i] |= (1 << j)
                adj_c[c][j] |= (1 << i)
                idx += 1

        # Backtracking: assign colors to edges (v, 0), (v, 1), ..., (v, n-1)
        # State: for each color c, the set of already-assigned vertices in S_c
        # When assigning color c to edge (v, i), check that i is not connected in color c to any vertex already in S_c

        assignment = [-1] * n  # assignment[i] = color of edge (v, i)
        S = [0] * 4  # bitmask of vertices assigned to color c

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            # Try each color
            for c in range(4):
                # Check if adding pos to S[c] is valid: pos must not be connected in color c to any vertex in S[c]
                if adj_c[c][pos] & S[c]:
                    continue
                # Assign
                assignment[pos] = c
                S[c] |= (1 << pos)
                if backtrack(pos + 1):
                    return True
                # Undo
                S[c] &= ~(1 << pos)
            return False

        if backtrack(0):
            return assignment
        return None

    # Try to extend from n=32
    current_n = 32
    current_colors = base_colors[:]

    while current_n < 62:
        if time.time() > deadline:
            break
        new_vertex_colors = try_add_vertex(current_colors, current_n)
        if new_vertex_colors is None:
            break
        # Add the new vertex
        for i in range(current_n):
            current_colors.append(new_vertex_colors[i])
        current_n += 1
        if current_n > best_n:
            best_n = current_n
            best_colors = current_colors[:]

    # Final verification
    if not verify(best_colors, best_n):
        # Fallback to base
        best_n = 32
        best_colors = base_colors[:]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
