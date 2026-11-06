"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s wall-clock kill

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Diagonal constraint: 2*a must not be a square
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build conflict graph: i and j conflict if valid_numbers[i] + valid_numbers[j] is a square
    # (including i == j already excluded by diagonal filter, but a+b with a!=b)
    adj = [0] * num_valid  # bitmask of neighbors
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b == a:
                continue
            j = val_to_idx.get(b)
            if j is not None:
                adj[i] |= (1 << j)

    # Symmetrize (should already be symmetric, but be safe)
    for i in range(num_valid):
        for j in range(num_valid):
            if (adj[i] >> j) & 1:
                adj[j] |= (1 << i)

    def popcount(x):
        return bin(x).count('1')

    def is_valid_set(indices):
        """Verify a set of indices is a valid independent set (no pair sums to square, incl diagonal)."""
        vals = [valid_numbers[i] for i in indices]
        s = set(vals)
        for a in s:
            if 2 * a in squares:
                return False
            for b in s:
                if b >= a and (a + b) in squares:
                    return False
        return True

    # Greedy coloring upper bound on the remaining candidate set
    # Colors the subgraph induced by 'cand' (bitmask); number of colors is an upper bound on independence number.
    def color_upper_bound(cand):
        # Simple greedy coloring: repeatedly pick a vertex of max degree in remaining, assign smallest color
        remaining = cand
        color_of = {}
        next_color = 0
        # To make it fast and somewhat deterministic, process vertices in order of index
        # But for a tighter bound, use a heuristic: order by descending degree within remaining
        # We'll do a simple sequential coloring which is fast; the bound may be looser but that's ok.
        # For better bound, use DSATUR-like: pick uncolored vertex with max saturation degree.
        # Given time constraints, use a quick greedy: iterate vertices, assign smallest available color.
        colors_used = 0
        # We'll collect vertices
        verts = []
        c = cand
        while c:
            lsb = c & (-c)
            verts.append(lsb.bit_length() - 1)
            c ^= lsb
        # Order by descending degree within cand for tighter bound
        verts.sort(key=lambda v: popcount(adj[v] & cand), reverse=True)
        color_assignment = {}
        for v in verts:
            # Determine which colors are used by colored neighbors
            used = set()
            nb = adj[v] & cand
            while nb:
                lsb = nb & (-nb)
                u = lsb.bit_length() - 1
                if u in color_assignment:
                    used.add(color_assignment[u])
                nb ^= lsb
            # Assign smallest non-used color
            col = 0
            while col in used:
                col += 1
            color_assignment[v] = col
            if col + 1 > colors_used:
                colors_used = col + 1
        return colors_used

    # Greedy independent set heuristic (min-degree)
    def greedy_min_degree():
        remaining = (1 << num_valid) - 1
        selected = []
        while remaining:
            # Pick vertex with minimum degree in remaining
            best = -1
            best_deg = float('inf')
            c = remaining
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best = v
                c ^= lsb
            selected.append(best)
            remaining &= ~(1 << best)
            remaining &= ~adj[best]
        return selected

    def greedy_random():
        remaining = (1 << num_valid) - 1
        selected = []
        while remaining:
            # Pick among min-degree vertices randomly
            c = remaining
            min_deg = float('inf')
            cands = []
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    cands = [v]
                elif deg == min_deg:
                    cands.append(v)
                c ^= lsb
            v = random.choice(cands)
            selected.append(v)
            remaining &= ~(1 << v)
            remaining &= ~adj[v]
        return selected

    # Branch and bound maximum independent set
    def bb_search(cand, current, best_size, best_set, time_limit):
        """
        cand: bitmask of remaining candidates
        current: bitmask of currently selected
        Returns (best_size, best_set_mask)
        """
        nonlocal best_global_size, best_global_mask
        # Upper bound: len(current) + color_upper_bound(cand)
        if popcount(current) + color_upper_bound(cand) <= best_global_size:
            return
        if time.time() > time_limit:
            return
        if cand == 0:
            if popcount(current) > best_global_size:
                best_global_size = popcount(current)
                best_global_mask = current
            return

        # Pick a vertex to branch on: choose one with max degree in cand (to reduce branching)
        c = cand
        best_v = -1
        best_deg = -1
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            deg = popcount(adj[v] & cand)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb

        v = best_v
        # Branch 1: include v
        new_current = current | (1 << v)
        new_cand = cand & ~(1 << v) & ~adj[v]
        bb_search(new_cand, new_current, best_global_size, best_global_mask, time_limit)

        # Branch 2: exclude v
        new_cand2 = cand & ~(1 << v)
        bb_search(new_cand2, current, best_global_size, best_global_mask, time_limit)

    # Local search with 1-1 and 2-2 exchanges
    def local_search(initial_mask, time_limit):
        nonlocal best_global_size, best_global_mask
        current = initial_mask
        cur_size = popcount(current)
        best_local = current
        best_local_size = cur_size
        no_improve = 0

        while time.time() < time_limit and no_improve < 50:
            improved = False

            # 1-1 exchange: remove one, add one
            # Find a vertex in current whose removal allows adding a non-neighbor that isn't already in
            c = current
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                # If we remove v, which new vertices become addable?
                without_v = current & ~(1 << v)
                # A vertex u can be added if u not in current and (adj[u] & without_v) == 0
                # and u != v
                addable = []
                all_mask = (1 << num_valid) - 1
                potential = all_mask & ~current  # not in current
                # For each potential u, check adj[u] & without_v == 0
                p = potential
                while p:
                    lsb2 = p & (-p)
                    u = lsb2.bit_length() - 1
                    if (adj[u] & without_v) == 0:
                        addable.append(u)
                    p ^= lsb2
                if addable:
                    # Try adding each addable; pick one that doesn't conflict with itself (trivially ok)
                    # We just add one
                    u = addable[0]
                    new_mask = without_v | (1 << u)
                    if popcount(new_mask) > best_local_size:
                        current = new_mask
                        best_local = new_mask
                        best_local_size = popcount(new_mask)
                        improved = True
                        break
                c ^= lsb

            # 2-2 exchange: remove two, add two
            if not improved and cur_size >= 2:
                # Try pairs
                verts_in = []
                c = current
                while c:
                    lsb = c & (-c)
                    verts_in.append(lsb.bit_length() - 1)
                    c ^= lsb
                # Sample some pairs to limit work
                if len(verts_in) > 30:
                    pairs_to_try = random.sample(list(range(len(verts_in))), 30)
                else:
                    pairs_to_try = list(range(len(verts_in)))
                found = False
                for i in pairs_to_try:
                    if found:
                        break
                    for j in range(i + 1, len(verts_in)):
                        v1, v2 = verts_in[i], verts_in[j]
                        without = current & ~(1 << v1) & ~(1 << v2)
                        potential = ((1 << num_valid) - 1) & ~current
                        addable = []
                        p = potential
                        while p:
                            lsb2 = p & (-p)
                            u = lsb2.bit_length() - 1
                            if (adj[u] & without) == 0:
                                addable.append(u)
                            p ^= lsb2
                        if len(addable) >= 2:
                            # Try to find two addable that don't conflict with each other
                            for a_idx in range(min(len(addable), 10)):
                                u1 = addable[a_idx]
                                for b_idx in range(a_idx + 1, min(len(addable), 10)):
                                    u2 = addable[b_idx]
                                    if (adj[u1] & (1 << u2)) == 0:
                                        new_mask = without | (1 << u1) | (1 << u2)
                                        if popcount(new_mask) > best_local_size:
                                            current = new_mask
                                            best_local = new_mask
                                            best_local_size = popcount(new_mask)
                                            improved = True
                                            found = True
                                            break
                        if found:
                            break
                    if found:
                        break

            cur_size = popcount(current)
            if not improved:
                no_improve += 1
            else:
                no_improve = 0

            if best_local_size > best_global_size:
                best_global_size = best_local_size
                best_global_mask = best_local

        return best_local, best_local_size

    # Main search
    best_global_size = 0
    best_global_mask = 0

    # Phase 1: Generate several initial solutions
    inits = []
    inits.append(greedy_min_degree())
    for _ in range(20):
        inits.append(greedy_random())

    best_init = max(inits, key=len)
    best_init_mask = 0
    for v in best_init:
        best_init_mask |= (1 << v)
    best_global_size = len(best_init)
    best_global_mask = best_init_mask

    # Phase 2: Run branch-and-bound from a few starts with time budgets
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5:
        # B&B from best initial's complement? Actually B&B searches from scratch with cand = all.
        # But we can use the best known as lower bound.
        # Run B&B with time limit
        bnb_time = min(remaining * 0.5, 20.0)
        bb_search((1 << num_valid) - 1, 0, best_global_size, best_global_mask, start_time + bnb_time)

    # Phase 3: Local search from best and several others
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 3:
        # Run local search from best_global_mask
        ls_time = min(remaining * 0.5, 15.0)
        local_search(best_global_mask, start_time + ls_time)

        # Run local search from a few other initial solutions
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 2:
            for init in inits[:5]:
                if time.time() > start_time + deadline - 1.0:
                    break
                mask = 0
                for v in init:
                    mask |= (1 << v)
                local_search(mask, start_time + deadline - 1.0)

    # Convert best mask to values
    best_indices = []
    m = best_global_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        best_indices.append(v)
        m ^= lsb

    best_values = [valid_numbers[i] for i in best_indices]

    # Verify
    if best_values and is_valid_set(set(best_values)):
        return sorted(best_values)
    else:
        # Fallback: simple greedy
        fb = greedy_min_degree()
        fb_vals = [valid_numbers[i] for i in fb]
        return sorted(fb_vals)

# EVOLVE-BLOCK-END
