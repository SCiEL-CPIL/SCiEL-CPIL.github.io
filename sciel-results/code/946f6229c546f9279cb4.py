import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s

    def time_exceeded():
        return time.time() - start_time > deadline

    # Precompute for q=29 (Paley coloring)
    q = 29
    squares_29 = {1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28}

    def build_paley_colors(n):
        """Build Paley coloring for prime n ≡ 1 (mod 4)."""
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)

    def edge_index(n, i, j):
        """Get the index of edge (i,j) in the colors list, where i < j."""
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adj(n, colors):
        """Build bitmask adjacency for each color."""
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def mask_to_list(mask):
        """Convert a bitmask to a list of vertex indices."""
        result = []
        while mask:
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            result.append(v)
            mask ^= lsb
        return result

    def find_k222(n, colors):
        """
        Find a K_{2,2,2} in either color class using bitmasks.
        Returns (color, vertices) if found, else (None, None).
        vertices is (i, j, k, l, m, o) where:
          - (i,j) is pair 1 (no edge between them)
          - (k,l) is pair 2 (no edge between them)
          - (m,o) is pair 3 (no edge between them)
          - All cross-pair edges exist in the given color
        """
        adj = build_adj(n, colors)

        def has_k222(adj_color):
            """
            Check if adj_color contains K_{2,2,2}.
            Returns vertices tuple if found, else None.
            """
            for i in range(n):
                for j in range(i + 1, n):
                    # (i,j) must be a non-edge (intra-pair)
                    if adj_color[i] & (1 << j):
                        continue

                    # Common neighbors of i and j
                    cn_mask = (adj_color[i] & adj_color[j]) & ~(1 << i) & ~(1 << j)

                    cn_list = mask_to_list(cn_mask)
                    if len(cn_list) < 4:
                        continue

                    # Find pairs (k,l) in cn_list with no edge
                    for a_idx in range(len(cn_list)):
                        k = cn_list[a_idx]
                        for b_idx in range(a_idx + 1, len(cn_list)):
                            l = cn_list[b_idx]
                            if adj_color[k] & (1 << l):
                                continue

                            # Common neighbors of i, j, k, l
                            cn2_mask = (adj_color[i] & adj_color[j] & adj_color[k] & adj_color[l])
                            cn2_mask &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)

                            cn2_list = mask_to_list(cn2_mask)
                            if len(cn2_list) < 2:
                                continue

                            for c_idx in range(len(cn2_list)):
                                m = cn2_list[c_idx]
                                for d_idx in range(c_idx + 1, len(cn2_list)):
                                    o = cn2_list[d_idx]
                                    if not (adj_color[m] & (1 << o)):
                                        return (i, j, k, l, m, o)

            return None

        # Check color 0
        result = has_k222(adj[0])
        if result is not None:
            return (0, result)

        # Check color 1
        result = has_k222(adj[1])
        if result is not None:
            return (1, result)

        return (None, None)

    # Verify Paley(29) is valid
    color, vertices = find_k222(q, colors_29)

    if color is None:
        # Paley(29) is valid!
        best_n = q
        best_colors = colors_29

        if not time_exceeded():
            random.seed(42)

            # Strategy: Try to extend to n=30 with smart initialization
            # and targeted local search using bitmask-based violation detection

            max_attempts = 30
            for attempt in range(max_attempts):
                if time_exceeded():
                    break

                # Create initial coloring for n=30
                colors_30 = colors_29[:]

                # Smart initialization for the 30th vertex
                # Use different strategies based on attempt number
                if attempt == 0:
                    # All zeros
                    for i in range(29):
                        colors_30.append(0)
                elif attempt == 1:
                    # All ones
                    for i in range(29):
                        colors_30.append(1)
                elif attempt == 2:
                    # Alternating
                    for i in range(29):
                        colors_30.append(i % 2)
                elif attempt == 3:
                    # Based on distance pattern
                    for i in range(29):
                        colors_30.append((i * 3) % 2)
                elif attempt < 8:
                    # Random
                    for i in range(29):
                        colors_30.append(random.randint(0, 1))
                else:
                    # Start from previous attempt's result if we had one
                    # For now, random with different seed
                    random.seed(42 + attempt)
                    for i in range(29):
                        colors_30.append(random.randint(0, 1))

                # Local search with targeted fixes
                max_iter = 300
                for iteration in range(max_iter):
                    if time_exceeded():
                        break

                    color, vertices = find_k222(30, colors_30)
                    if color is None:
                        # Found valid n=30 coloring!
                        return {'n': 30, 'colors': colors_30}

                    # We have a K_{2,2,2} in `color` with vertices
                    # (i, j, k, l, m, o) where:
                    # (i,j) is pair 1, (k,l) is pair 2, (m,o) is pair 3
                    # All cross-pair edges have color `color`
                    # To break it, flip one cross-pair edge to the other color

                    pairs = [(vertices[0], vertices[1]),
                             (vertices[2], vertices[3]),
                             (vertices[4], vertices[5])]

                    # Collect all cross-pair edges
                    cross_edges = []
                    for p1 in range(3):
                        for p2 in range(p1 + 1, 3):
                            for v1 in pairs[p1]:
                                for v2 in pairs[p2]:
                                    cross_edges.append((v1, v2))

                    # Try flipping each cross-pair edge and see which reduces violations
                    # For efficiency, just pick one randomly or try all
                    if len(cross_edges) <= 6:
                        # Try all, pick best
                        best_edge = None
                        best_count = float('inf')
                        for v1, v2 in cross_edges:
                            idx = edge_index(30, v1, v2)
                            old_color = colors_30[idx]
                            colors_30[idx] = 1 - old_color

                            # Quick check: just count violations (approximate)
                            # For speed, just check if this specific K222 is broken
                            # and count new violations roughly
                            # Actually, let's just flip and recheck
                            c2, v2_found = find_k222(30, colors_30)
                            if c2 is None:
                                # Perfect! This flip solved it
                                return {'n': 30, 'colors': colors_30}

                            # Count how many K222 violations exist (approximate by checking a few)
                            # For now, just use a simple heuristic: prefer flips that
                            # don't create new violations in the same color
                            # This is expensive, so let's just pick randomly
                            colors_30[idx] = old_color  # revert

                        # Just pick randomly
                        v1, v2 = random.choice(cross_edges)
                        idx = edge_index(30, v1, v2)
                        colors_30[idx] = 1 - colors_30[idx]
                    else:
                        v1, v2 = random.choice(cross_edges)
                        idx = edge_index(30, v1, v2)
                        colors_30[idx] = 1 - colors_30[idx]

                # After local search, check if we succeeded
                color, vertices = find_k222(30, colors_30)
                if color is None:
                    return {'n': 30, 'colors': colors_30}

            # If we couldn't find n=30, return the best valid coloring
            return {'n': best_n, 'colors': best_colors}
        else:
            return {'n': q, 'colors': colors_29}
    else:
        # Paley(29) is NOT valid - need to find another construction
        # Try local search on n=29
        random.seed(42)

        # Start with Paley(29) and do local search
        colors = colors_29[:]
        max_iter = 3000
        for iteration in range(max_iter):
            if time_exceeded():
                break

            color, vertices = find_k222(29, colors)
            if color is None:
                return {'n': 29, 'colors': colors}

            # Fix the violation by flipping a cross-pair edge
            pairs = [(vertices[0], vertices[1]),
                     (vertices[2], vertices[3]),
                     (vertices[4], vertices[5])]

            cross_edges = []
            for p1 in range(3):
                for p2 in range(p1 + 1, 3):
                    for v1 in pairs[p1]:
                        for v2 in pairs[p2]:
                            cross_edges.append((v1, v2))

            v1, v2 = random.choice(cross_edges)
            idx = edge_index(29, v1, v2)
            colors[idx] = 1 - colors[idx]

        # If we couldn't find valid coloring, return best effort
        return {'n': 29, 'colors': colors}
