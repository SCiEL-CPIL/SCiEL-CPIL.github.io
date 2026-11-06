import time

# EVOLVE-BLOCK-START

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0

    # Build adjacency: good_adj[u] is bitmask of v where (u*v)%(u+v) != 0
    good_adj = [0] * (N + 1)
    for u in range(1, N + 1):
        mask = 0
        for v in range(1, N + 1):
            if u == v:
                continue
            if (u * v) % (u + v) != 0:
                mask |= (1 << (v - 1))
        good_adj[u] = mask

    best_clique_mask = 0
    best_size = 0

    # Greedy initial solution
    P = (1 << N) - 1
    R = 0
    while P:
        best_v = -1
        best_deg = -1
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            v = lsb.bit_length() - 1
            deg = (good_adj[v + 1] & P).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            p_temp ^= lsb
        if best_v == -1:
            break
        R |= (1 << best_v)
        P &= good_adj[best_v + 1]
        P &= ~(1 << best_v)

    if R.bit_count() > best_size:
        best_size = R.bit_count()
        best_clique_mask = R

    def color_bound(P):
        """Fast greedy coloring: order by degree desc, first-fit."""
        verts_in_P = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts_in_P.append(v)
            temp ^= lsb

        # Sort by degree in P descending
        verts_in_P.sort(key=lambda v: (good_adj[v + 1] & P).bit_count(), reverse=True)

        color_of = {}
        num_colors = 0
        for v in verts_in_P:
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            adj_mask = good_adj[v + 1] & P
            am = adj_mask
            while am:
                lsb = am & -am
                u = lsb.bit_length() - 1
                if u in color_of:
                    neighbor_colors.add(color_of[u])
                am ^= lsb
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            if c + 1 > num_colors:
                num_colors = c + 1

        # Order vertices by color descending (high color first for branching)
        ordered = sorted(verts_in_P, key=lambda v: color_of[v], reverse=True)
        return num_colors, ordered, color_of

    def expand(P, R):
        nonlocal best_size, best_clique_mask

        if time.time() > deadline:
            return

        if not P:
            if R.bit_count() > best_size:
                best_size = R.bit_count()
                best_clique_mask = R
            return

        k, ordered_vertices, color_of = color_bound(P)

        if R.bit_count() + k <= best_size:
            return

        for v in ordered_vertices:
            if time.time() > deadline:
                return
            if not (P & (1 << v)):
                continue

            new_R = R | (1 << v)
            new_P = P & good_adj[v + 1]

            expand(new_P, new_R)

            P &= ~(1 << v)
            k -= 1

            if R.bit_count() + k <= best_size:
                break

    # Run Tomita B&B
    P = (1 << N) - 1
    R = 0
    expand(P, R)

    # Local search polish: try to add vertices to the best clique
    best_set = set()
    for i in range(N):
        if best_clique_mask & (1 << i):
            best_set.add(i + 1)

    # Try adding vertices that don't conflict with any in best_set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(1, N + 1):
            if v in best_set:
                continue
            # Check if v can be added: v must be good with all in best_set
            can_add = True
            for u in best_set:
                if (u * v) % (u + v) == 0:
                    can_add = False
                    break
            if can_add:
                best_set.add(v)
                improved = True

    # Also try removing one vertex and adding two others
    for v in list(best_set):
        if time.time() > deadline:
            break
        trial_set = best_set - {v}
        # Find candidates that can be added to trial_set
        candidates = []
        for u in range(1, N + 1):
            if u in trial_set:
                continue
            can_add = True
            for w in trial_set:
                if (u * w) % (u + w) == 0:
                    can_add = False
                    break
            if can_add:
                candidates.append(u)
        # Try to find two candidates that are also good with each other
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if (a * b) % (a + b) != 0:
                    new_set = trial_set | {a, b}
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break

    result = sorted(best_set)
    if not result:
        result = [1]

    return result

# EVOLVE-BLOCK-END
