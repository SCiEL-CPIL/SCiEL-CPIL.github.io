import numpy as np
import itertools
import time

def construct():
        start_time = time.time()
        deadline = start_time + 145.0
        D = 11

        # 1. Generate all supports (combinations of 4 indices)
        supports = list(itertools.combinations(range(D), 4))
        N_S = len(supports)  # 330

        # Convert supports to bitmasks for fast intersection check
        masks = []
        for s in supports:
            m = 0
            for idx in s:
                m |= (1 << idx)
            masks.append(m)

        # 2. Build COMPATIBILITY graph: edge if intersection size <= 2
        # (i.e., NOT conflicting; conflict is intersection == 3 or 4)
        # For 4-element subsets, intersection can be 0,1,2,3,4
        # Two 4-subsets of [11] cannot have intersection 4 unless identical
        # We want pairs with intersection <= 2 to be compatible
        compat = np.zeros((N_S, N_S), dtype=bool)
        for i in range(N_S):
            for j in range(i + 1, N_S):
                inter = (masks[i] & masks[j]).bit_count()
                if inter <= 2:
                    compat[i][j] = True
                    compat[j][i] = True

        # 3. Tomita-style branch-and-bound maximum clique
        # P = candidate set, X = excluded set, C = current clique
        best_clique = []
        best_size = 0

        # Precompute neighbor lists as sorted arrays for efficiency
        neighbors = [np.where(compat[i])[0] for i in range(N_S)]

        def branch_and_bound(P, X, C):
            """
            Tomita's MCQ algorithm for maximum clique.
            P: list of candidate vertices (sorted)
            X: list of excluded vertices (sorted)
            C: current clique (list)
            """
            nonlocal best_clique, best_size

            if len(P) == 0:
                if len(C) > best_size:
                    best_size = len(C)
                    best_clique = C.copy()
                return

            # Greedy coloring on P (Tomita's approach)
            # Process P in order, assign colors greedily
            # color[v] = color index for vertex v in P
            # The key insight: vertices with the same color form an independent set
            # within the subgraph induced by P, so at most one per color can be in a clique.

            # We'll use a greedy coloring where we process P in reverse order
            # (higher degree first tends to give better bounds)
            # Actually, Tomita processes in the given order. Let's use degree ordering.

            # For efficiency, sort P by degree within P (descending) for better coloring
            # But this is expensive. Let's just use the natural order for now.

            P_list = list(P)
            n = len(P_list)

            # Greedy coloring: assign colors 1, 2, 3, ...
            # color[i] = color assigned to P_list[i]
            color = [0] * n
            # Color classes: list of lists, color_classes[c] = indices in P_list with color c+1
            color_classes = []
            # For each color, we need to track which vertices have that color
            # and ensure they form an independent set in the subgraph induced by P

            # Standard greedy coloring for clique bound:
            # Process vertices in order. For vertex v, assign the smallest color c
            # such that no already-colored neighbor of v has color c.

            # To make this efficient, we track for each color which vertices have it.
            # A vertex v can get color c if none of its neighbors in P have color c.

            # Alternative: use the "reverse greedy coloring" which is standard in Tomita.
            # Process P in the given order. For each vertex v in P:
            #   find the smallest color c such that v is not adjacent to any vertex
            #   in P that has already been assigned color c.
            #   assign color c to v.

            # For efficiency with numpy, let's do this with sets.
            # Given the small size (n <= 330), Python loops are acceptable.

            # Actually, let's use a simpler approach: 
            # The bound is: max possible clique size from P is at most the number of colors.
            # So if len(C) + num_colors <= best_size, we can prune.

            # Let's implement proper greedy coloring:
            # We process P_list in order. For each vertex v at position i:
            #   Check which colors are "available" (no neighbor of v in P has that color)
            #   Assign the smallest available color.

            # For speed, maintain: for each color c, the set of vertices (by index in P_list) with color c.
            # And for each vertex v, we need to check if any neighbor in P has color c.

            # This is getting complex. Let me use a simpler but effective approach:
            # Use the bound: |P| as upper bound (trivial), but that's weak.
            # Better: use greedy coloring properly.

            # Let me implement a cleaner version:
            # Order P by degree (descending) for better coloring quality.
            # Then do greedy coloring.

            # Compute degrees within P
            P_set = set(P_list)
            deg = [len([u for u in P_list if u != v and compat[v][u]]) for v in P_list]

            # Sort by degree descending
            order = sorted(range(n), key=lambda i: -deg[i])
            # Reorder P_list
            P_sorted = [P_list[i] for i in order]
            # We'll work with P_sorted

            n2 = len(P_sorted)
            color = [0] * n2
            # color_avail[c] = set of vertices (by position in P_sorted) with color c
            # We need to check for each new vertex if it conflicts with any vertex of color c
            # that has already been placed.

            # For each color, maintain a bitmask or set of vertices with that color.
            # Since n2 <= 330, we can use sets.

            color_vertices = {}  # color -> set of positions in P_sorted with that color
            max_color = 0

            for pos in range(n2):
                v = P_sorted[pos]
                # Find the smallest color c such that no neighbor of v (in P_sorted, already placed) has color c
                # Neighbors of v in P: neighbors[v] intersect P_set
                # But we only care about already-placed neighbors (positions < pos)
                # For each color c, check if any vertex with color c is a neighbor of v.

                # Get all neighbors of v that are in P
                v_neighbors_in_P = set(neighbors[v]) & P_set

                # Find available colors
                c = 1
                while c in color_vertices:
                    # Check if any vertex in color_vertices[c] is a neighbor of v
                    if v_neighbors_in_P & color_vertices[c]:
                        c += 1
                    else:
                        break

                if c > max_color:
                    max_color = c

                color[pos] = c
                if c not in color_vertices:
                    color_vertices[c] = set()
                color_vertices[c].add(pos)

            # Now, the upper bound on clique size from P is max_color
            if len(C) + max_color <= best_size:
                return

            # Branch: process vertices in reverse order of coloring (highest color first)
            # For each color c from max_color down to 1:
            #   For each vertex v with color c (in some order):
            #     Add v to C, recurse with P' = P ∩ N(v) ∩ {vertices with color <= c}
            #     Remove v from P

            # We need to process in a specific order. Tomita's algorithm:
            # For c from max_color down to 1:
            #   For each vertex v in P with color c (in reverse order of placement):
            #     if len(C) + c <= best_size: return (prune)
            #     P' = {u in P : u is neighbor of v and color[u] <= c}
            #     X' = {u in X : u is neighbor of v and color[u] <= c}  # not needed for basic version
            #     recurse(P', X', C + [v])
            #     remove v from P

            # Let's implement this:
            # We need to track which vertices are still in P (not yet branched on)
            in_P = [True] * n2  # in_P[pos] = True if P_sorted[pos] is still in P

            for c in range(max_color, 0, -1):
                if len(C) + c <= best_size:
                    return

                # Get vertices with color c, in reverse order
                if c not in color_vertices:
                    continue
                # Get positions with color c
                verts_c = list(color_vertices[c])
                # Process in reverse order
                for pos in reversed(verts_c):
                    if not in_P[pos]:
                        continue

                    v = P_sorted[pos]

                    # New P: neighbors of v that are in P and have color <= c
                    new_P = []
                    for pos2 in range(n2):
                        if not in_P[pos2]:
                            continue
                        if pos2 == pos:
                            continue
                        if color[pos2] > c:
                            continue
                        v2 = P_sorted[pos2]
                        if compat[v][v2]:
                            new_P.append(v2)

                    # Recurse
                    C.append(v)
                    branch_and_bound(new_P, [], C)
                    C.pop()

                    # Remove v from P
                    in_P[pos] = False

                    if time.time() > deadline - 10.0:
                        return

        # Initial call: P = all vertices, X = empty, C = empty
        # Sort initial P by degree descending for better performance
        all_degrees = [len(neighbors[i]) for i in range(N_S)]
        initial_order = sorted(range(N_S), key=lambda i: -all_degrees[i])
        initial_P = initial_order

        branch_and_bound(initial_P, [], [])

        # 4. Construct vectors from best clique
        vectors = []
        for s_idx in best_clique:
            support = supports[s_idx]
            # Generate all 16 sign combinations
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int8)
                for idx, sign in zip(support, signs):
                    v[idx] = sign
                vectors.append(v)

        if not vectors:
            return np.array([], dtype=np.int8).reshape(0, D)

        final_vectors = np.array(vectors, dtype=np.int8)

        return final_vectors
