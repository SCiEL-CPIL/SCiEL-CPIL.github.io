import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n_base = 29
    exponent = (n_base - 1) // 2

    # Precompute Paley graph adjacency for 29 vertices
    adj = [[0] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % n_base
            res = pow(diff, exponent, n_base)
            if res == 1:
                adj[i][j] = 1
                adj[j][i] = 1
            else:
                adj[i][j] = 0
                adj[j][i] = 0

    # Convert to numpy arrays
    adj_np = np.array(adj, dtype=np.int8)
    adj_comp_np = np.zeros((n_base, n_base), dtype=np.int8)
    np.fill_diagonal(adj_comp_np, 0)
    adj_comp_np = 1 - adj_np
    np.fill_diagonal(adj_comp_np, 0)

    # Precompute bitmasks
    bit_masks = [1 << v for v in range(n_base)]

    # Precompute common neighbors for all pairs
    common_adj = [0] * (n_base * n_base)
    common_comp = [0] * (n_base * n_base)

    for u in range(n_base):
        for v in range(u + 1, n_base):
            mask_common = 0
            mask_common_comp = 0
            for w in range(n_base):
                if w != u and w != v:
                    if adj[u][w] and adj[v][w]:
                        mask_common |= bit_masks[w]
                    if adj_comp_np[u][w] and adj_comp_np[v][w]:
                        mask_common_comp |= bit_masks[w]
            common_adj[u * n_base + v] = mask_common
            common_adj[v * n_base + u] = mask_common
            common_comp[u * n_base + v] = mask_common_comp
            common_comp[v * n_base + u] = mask_common_comp

    def count_k22_fast(mask, common_arr):
        count = 0
        for u in range(n_base):
            if not (mask & bit_masks[u]):
                continue
            for v in range(u + 1, n_base):
                if not (mask & bit_masks[v]):
                    continue
                cn_mask = common_arr[u * n_base + v] & mask
                c = bin(cn_mask).count('1')
                if c >= 2:
                    count += c * (c - 1) // 2
        return count

    def get_cost(mask):
        cost1 = count_k22_fast(mask, common_adj)
        mask_inv = ((1 << n_base) - 1) ^ mask
        cost2 = count_k22_fast(mask_inv, common_comp)
        return cost1 + cost2

    # Local Search
    q_set = {1, 2, 4, 8, 16, 7, 14, 28, 27, 25, 21, 13, 9, 18}
    init_mask = 1
    for v in q_set:
        init_mask |= bit_masks[v]

    best_mask = init_mask
    best_cost = get_cost(best_mask)

    if best_cost > 0:
        current_mask = init_mask
        current_cost = best_cost
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            bits = list(range(n_base))
            random.shuffle(bits)
            for b in bits:
                new_mask = current_mask ^ bit_masks[b]
                new_cost = get_cost(new_mask)
                if new_cost < current_cost:
                    current_mask = new_mask
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        break
            if current_cost == 0:
                break
            if not improved:
                current_mask = random.getrandbits(n_base)
                current_cost = get_cost(current_mask)

        if current_cost < best_cost:
            best_mask = current_mask
            best_cost = current_cost

    # New: Greedy backtracking search for n=30
    # Try to add vertex 29 with optimal edge colors to avoid K2,2,2 in either color
    n30 = 30
    all_mask_29 = (1 << n_base) - 1

    # For vertex 29, we need to choose colors for edges (29, i) for i=0..28
    # such that no K2,2,2 is formed in either color class
    # A K2,2,2 in color c requires two vertices u,v with at least 2 common neighbors in color c
    # among the other vertices, and both u and v connected to those neighbors in color c

    # Strategy: For each possible color assignment of edges from 29 to 0..28,
    # check if it creates a K2,2,2 in either color
    # We can use symmetry breaking: fix edge (29,0) = 0

    # Precompute for each pair (u,v) in 0..28, the number of common neighbors
    # in each color class within the existing 29-vertex graph
    # If we add vertex 29 with certain edge colors, we need to ensure
    # no new K2,2,2 is formed

    # A new K2,2,2 involving vertex 29 would require:
    # - Two other vertices u,v
    # - At least 2 common neighbors w1,w2 of u and v
    # - Edges (29,u), (29,v), (29,w1), (29,w2) all in the same color
    # - And (u,w1), (u,w2), (v,w1), (v,w2) all in that color (already satisfied in base)

    # So for a given color assignment c[29][i], we need:
    # For color 0: No pair (u,v) with c[29][u]=c[29][v]=0 and at least 2 common neighbors
    #               w1,w2 with c[29][w1]=c[29][w2]=0 and (u,w1),(u,w2),(v,w1),(v,w2) all color 0
    # Similarly for color 1

    # Let's precompute for each pair (u,v) in 0..28:
    # common0[u,v] = bitmask of w such that (u,w) and (v,w) are both color 0
    # common1[u,v] = bitmask of w such that (u,w) and (v,w) are both color 1

    common0 = [0] * (n_base * n_base)
    common1 = [0] * (n_base * n_base)
    for u in range(n_base):
        for v in range(u + 1, n_base):
            m0 = 0
            m1 = 0
            for w in range(n_base):
                if w != u and w != v:
                    if adj[u][w] == 0 and adj[v][w] == 0:
                        m0 |= bit_masks[w]
                    else:
                        m1 |= bit_masks[w]
            common0[u * n_base + v] = m0
            common0[v * n_base + u] = m0
            common1[u * n_base + v] = m1
            common1[v * n_base + u] = m1

    # Now search for a valid assignment of colors for edges (29, i) for i=0..28
    # Use backtracking with pruning
    # Symmetry breaking: fix c[29][0] = 0

    found_30 = False
    best_30_mask = None

    # Try multiple random orderings of vertices to assign colors to
    for attempt in range(50):
        if time.time() > deadline:
            break

        # Random permutation of vertices 1..28 (vertex 0 is fixed to color 0)
        vertices = list(range(1, n_base))
        random.shuffle(vertices)

        # Backtracking search
        assignment = [-1] * n_base  # assignment[i] = color of edge (29, i)
        assignment[0] = 0  # symmetry breaking

        def backtrack(idx):
            nonlocal found_30, best_30_mask
            if found_30 or time.time() > deadline:
                return

            if idx == len(vertices):
                # Check if this assignment is valid
                # For each color c in {0,1}, check no K2,2,2 involving vertex 29
                for c in [0, 1]:
                    # Get set of vertices i with assignment[i] = c
                    vertices_c = [i for i in range(n_base) if assignment[i] == c]
                    # Check all pairs in vertices_c
                    for ii in range(len(vertices_c)):
                        u = vertices_c[ii]
                        for jj in range(ii + 1, len(vertices_c)):
                            v = vertices_c[jj]
                            # Get common neighbors in color c
                            if c == 0:
                                cn_mask = common0[u * n_base + v]
                            else:
                                cn_mask = common1[u * n_base + v]
                            # Count common neighbors that also have assignment = c
                            cn_count = 0
                            for w in range(n_base):
                                if w != u and w != v and (cn_mask & bit_masks[w]) and assignment[w] == c:
                                    cn_count += 1
                            if cn_count >= 2:
                                return  # Invalid, prune

                # Valid!
                found_30 = True
                mask = 0
                for i in range(n_base):
                    if assignment[i] == 0:
                        mask |= bit_masks[i]
                best_30_mask = mask
                return

            v = vertices[idx]
            for color in [0, 1]:
                assignment[v] = color
                backtrack(idx + 1)
                assignment[v] = -1
                if found_30 or time.time() > deadline:
                    return

        backtrack(0)
        if found_30:
            break

    if found_30:
        n = 30
        colors = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(adj[i][j])
        for i in range(n_base):
            # Edge (29, i): if best_30_mask has bit i set, color 0, else color 1
            if (best_30_mask >> i) & 1:
                colors.append(0)
            else:
                colors.append(1)
        return {'n': n, 'colors': colors}
    else:
        # Fallback to n=29
        n = 29
        colors = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(adj[i][j])
        return {'n': n, 'colors': colors}
