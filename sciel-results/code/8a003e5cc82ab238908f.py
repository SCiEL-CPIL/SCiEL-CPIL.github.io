import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    n_base = 29

    # Generate Paley(29) coloring
    squares = set()
    for x in range(1, n_base):
        squares.add((x * x) % n_base)

    # Build adjacency matrix for n=29
    adj29 = np.zeros((n_base, n_base), dtype=np.int8)
    for i in range(n_base):
        for j in range(i + 1, n_base):
            d = (j - i) % n_base
            c = 1 if d in squares else 0
            adj29[i, j] = c
            adj29[j, i] = c

    # Helper: build neighbor bitmasks for a given color
    def build_neigh(adj, color_val, n):
        neigh = np.zeros(n, dtype=np.int64)
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and adj[i, j] == color_val:
                    mask |= (1 << j)
            neigh[i] = mask
        return neigh

    # Helper: count K2,2,2 violations for a given color (fast bitmask)
    def count_violations(adj, color_val, n):
        neigh = build_neigh(adj, color_val, n)
        count = 0
        for a in range(n):
            for b in range(a + 1, n):
                S = neigh[a] & neigh[b]
                pop = bin(S).count('1')
                if pop < 4:
                    continue
                # Collect vertices in S
                verts = []
                for v in range(n):
                    if S & (1 << v):
                        verts.append(v)
                lv = len(verts)
                for ci in range(lv):
                    c = verts[ci]
                    for di in range(ci + 1, lv):
                        d = verts[di]
                        T = neigh[c] & neigh[d] & S
                        if bin(T).count('1') >= 2:
                            count += 1
        return count

    # Helper: total violations (both colors)
    def total_violations(adj, n):
        return count_violations(adj, 0, n) + count_violations(adj, 1, n)

    # Verify n=29 is valid
    v29 = total_violations(adj29, n_base)
    if v29 != 0:
        # Shouldn't happen, but fallback
        colors = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(int(adj29[i, j]))
        return {'n': n_base, 'colors': colors}

    # Build colors list for n=29 (as fallback)
    colors29 = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors29.append(int(adj29[i, j]))

    # Try to extend to n=30 via local search
    n_target = 30
    adj = np.zeros((n_target, n_target), dtype=np.int8)
    adj[:n_base, :n_base] = adj29

    # Initial assignment for vertex 29 (index 29): use quadratic residue pattern
    # For vertex 29, edge to vertex i: color based on (29 - i) % 29 in squares
    for i in range(n_base):
        d = (29 - i) % n_base
        adj[29, i] = 1 if d in squares else 0
        adj[i, 29] = adj[29, i]

    # Check initial violation count
    v_init = total_violations(adj, n_target)

    # If already valid (unlikely), return
    if v_init == 0:
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(int(adj[i, j]))
        return {'n': n_target, 'colors': colors}

    # Local search: flip edges to reduce violations
    # Precompute all edges as a list of (i, j) with i < j
    edges = []
    for i in range(n_target):
        for j in range(i + 1, n_target):
            edges.append((i, j))

    best_v = v_init
    best_adj = adj.copy()
    np.random.seed(12345)

    # Simulated annealing / hill climbing
    current_v = v_init
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break

        # Pick a random edge to flip
        idx = np.random.randint(0, len(edges))
        i, j = edges[idx]

        # Flip the edge
        old_val = adj[i, j]
        adj[i, j] = 1 - old_val
        adj[j, i] = adj[i, j]

        # Recompute violations
        new_v = total_violations(adj, n_target)

        if new_v <= current_v:
            current_v = new_v
            if new_v < best_v:
                best_v = new_v
                best_adj = adj.copy()
                if best_v == 0:
                    break
        else:
            # Revert (strict hill climbing for reliability)
            adj[i, j] = old_val
            adj[j, i] = old_val

        # Periodic random restarts from best if stuck
        if it % 5000 == 4999 and it > 0:
            # Try a few random flips from best_adj
            adj_tmp = best_adj.copy()
            for _ in range(5):
                ri = np.random.randint(0, n_target)
                rj = np.random.randint(0, n_target)
                if ri != rj:
                    adj_tmp[ri, rj] = 1 - adj_tmp[ri, rj]
                    adj_tmp[rj, ri] = adj_tmp[ri, rj]
            tv = total_violations(adj_tmp, n_target)
            if tv < current_v:
                adj = adj_tmp
                current_v = tv
                if tv < best_v:
                    best_v = tv
                    best_adj = adj.copy()

    # Use the best result found
    if best_v == 0:
        adj = best_adj
    else:
        # Check if best is better than n=29 baseline (it won't be for n=30 if invalid)
        # Fall back to n=29
        adj = adj29
        n_final = n_base
    if best_v == 0:
        n_final = n_target
    else:
        n_final = n_base

    # Build final colors list
    colors = []
    for i in range(n_final):
        for j in range(i + 1, n_final):
            if n_final == n_base:
                colors.append(int(adj29[i, j]))
            else:
                colors.append(int(adj[i, j]))

    return {'n': n_final, 'colors': colors}
