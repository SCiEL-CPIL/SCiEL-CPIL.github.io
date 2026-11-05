import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflict graph using bitsets for speed
    # conflict_mask[v] has bit u set iff (v*u)%(v+u)==0 (i.e., they conflict)
    conflict_mask = [0] * (N + 1)

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            a, b = i, j
            while b:
                a, b = b, a % b
            g = a
            u_prime = i // g
            v_prime = j // g
            if g % (u_prime + v_prime) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    def can_add(v, s_mask):
        """Check if vertex v can be added to set represented by s_mask."""
        return (conflict_mask[v] & s_mask) == 0

    def popcount(mask):
        return bin(mask).count('1')

    def mask_to_set(mask):
        return {v for v in vertices if (mask >> v) & 1}

    def greedy_by_order(order, s_mask=0):
        result_mask = s_mask
        for v in order:
            if (result_mask >> v) & 1:
                continue
            if can_add(v, result_mask):
                result_mask |= (1 << v)
        return result_mask

    # ---------- Spectral core heuristic (inspiration) ----------
    # Compute the largest-magnitude eigenvector of the conflict adjacency matrix
    # via power iteration. Vertices with the largest |eigenvector| are the most
    # "central" (high-degree) in the conflict graph; we use the *opposite*
    # (smallest |value|) as a low-conflict core to seed a large independent set.
    def spectral_order():
        # Build adjacency as list of bitsets (symmetric, no self loops)
        adj = conflict_mask[:]  # adj[i] bitset of neighbors of i
        # Power iteration on a random start vector
        n = N
        vec = [random.random() for _ in range(n + 1)]  # index 1..N
        vec[0] = 0.0
        for _ in range(200):
            new_vec = [0.0] * (n + 1)
            for i in range(1, n + 1):
                # dot(adj[i], vec)
                m = adj[i]
                s = 0.0
                # iterate set bits
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    s += vec[idx]
                    m ^= lb
                new_vec[i] = s
            # normalize
            norm = math.sqrt(sum(x * x for x in new_vec))
            if norm > 1e-12:
                for i in range(1, n + 1):
                    new_vec[i] /= norm
            vec = new_vec
        # Order vertices by |vec| ascending (low-conflict core first)
        order = sorted(vertices, key=lambda v: abs(vec[v]))
        return order

    # ---------- Initial candidates ----------
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N + 1)]
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]
    order_odd_first = sorted(vertices, key=lambda x: (x % 2 == 0, degrees[x]))
    order_value = list(range(1, N + 1))
    order_value_desc = list(range(N, 0, -1))
    order_spectral = spectral_order()
    order_spectral_rev = list(reversed(order_spectral))

    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        greedy_by_order(order_odd),
        greedy_by_order(order_even),
        greedy_by_order(order_odd_first),
        greedy_by_order(order_value),
        greedy_by_order(order_value_desc),
        greedy_by_order(order_spectral),
        greedy_by_order(order_spectral_rev),
    ]

    # Multi-start random shuffles
    for _ in range(25):
        r = vertices[:]
        random.shuffle(r)
        candidates.append(greedy_by_order(r))

    best_mask = 0
    for c in candidates:
        if popcount(c) > popcount(best_mask):
            best_mask = c

    # ---------- Local search: add + 1-for-2 swaps ----------
    def local_search(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Try adding each excluded vertex
            excluded = [v for v in vertices if not (s_mask >> v) & 1]
            random.shuffle(excluded)
            for v in excluded:
                if time.time() >= time_limit:
                    break
                if can_add(v, s_mask):
                    s_mask |= (1 << v)
                    improved = True

            # Phase 2: Try 1-for-2 swaps (remove one, add two non-conflicting)
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() >= time_limit:
                        break
                    temp_mask = s_mask & ~(1 << u)
                    addable = []
                    for v in vertices:
                        if v != u and not (s_mask >> v) & 1 and can_add(v, temp_mask):
                            addable.append(v)
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        found = False
                        limit = min(len(addable), 25)
                        for i in range(limit):
                            for j in range(i + 1, limit):
                                v1, v2 = addable[i], addable[j]
                                if not ((conflict_mask[v1] >> v2) & 1):
                                    s_mask = temp_mask | (1 << v1) | (1 << v2)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
        return s_mask

    time_limit = deadline - 25.0
    best_mask = local_search(best_mask, time_limit)

    # ---------- Simulated annealing ----------
    def simulated_annealing(s_mask, time_limit):
        current_mask = s_mask
        current_size = popcount(current_mask)
        best_mask_sa = current_mask
        best_size_sa = current_size

        t0 = 50.0
        t_end = 0.1
        cooling = 0.995
        temp = t0
        no_improve = 0

        while time.time() < time_limit and temp > t_end:
            move_type = random.random()

            if move_type < 0.4:
                # Try adding a random excluded vertex
                excluded = [v for v in vertices if not (current_mask >> v) & 1]
                if not excluded:
                    temp *= cooling
                    continue
                v = random.choice(excluded)
                if can_add(v, current_mask):
                    new_mask = current_mask | (1 << v)
                    new_size = current_size + 1
                    delta = 1
                else:
                    conflicts = [u for u in vertices if (current_mask >> u) & 1 and ((conflict_mask[v] >> u) & 1)]
                    if not conflicts:
                        temp *= cooling
                        continue
                    u = random.choice(conflicts)
                    temp_mask = current_mask & ~(1 << u)
                    if can_add(v, temp_mask):
                        new_mask = temp_mask | (1 << v)
                        new_size = current_size
                        delta = 0
                    else:
                        temp *= cooling
                        continue
            elif move_type < 0.7:
                # Try removing a random included vertex
                included = [v for v in vertices if (current_mask >> v) & 1]
                if not included:
                    temp *= cooling
                    continue
                v = random.choice(included)
                new_mask = current_mask & ~(1 << v)
                new_size = current_size - 1
                delta = -1
            else:
                # Try a swap: remove one, add one
                included = [v for v in vertices if (current_mask >> v) & 1]
                excluded = [v for v in vertices if not (current_mask >> v) & 1]
                if not included or not excluded:
                    temp *= cooling
                    continue
                u = random.choice(included)
                v = random.choice(excluded)
                temp_mask = current_mask & ~(1 << u)
                if can_add(v, temp_mask):
                    new_mask = temp_mask | (1 << v)
                    new_size = current_size
                    delta = 0
                else:
                    temp *= cooling
                    continue

            # Accept/reject
            if delta >= 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size_sa:
                    best_mask_sa = current_mask
                    best_size_sa = current_size
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if random.random() < math.exp(delta / temp):
                    current_mask = new_mask
                    current_size = new_size
                no_improve += 1

            temp *= cooling

            if no_improve > 250:
                break

        return best_mask_sa

    sa_time_limit = deadline - 12.0
    best_mask = simulated_annealing(best_mask, sa_time_limit)

    # ---------- Final local search polish ----------
    if time.time() < deadline - 6.0:
        best_mask = local_search(best_mask, deadline - 4.0)

    # ---------- Branch-and-bound polish with coloring upper bound ----------
    # Try to improve the current solution by exploring whether a strictly larger
    # independent set exists, using a greedy-coloring upper bound on the
    # complement graph (clique cover of the conflict graph).
    def color_bound(remaining_mask):
        """Greedy coloring of the complement subgraph induced by remaining_mask.
        Returns the number of colors used, which upper-bounds the independent
        set size of the conflict graph on remaining_mask."""
        if remaining_mask == 0:
            return 0
        colors = 0
        uncolored = remaining_mask
        while uncolored:
            colors += 1
            # Pick a vertex with highest degree in the uncolored complement subgraph
            # (i.e., fewest conflicts among uncolored) to keep color classes large
            best_v = -1
            best_deg = -1
            m = uncolored
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                # degree in complement within uncolored = popcount(uncolored & ~conflict_mask[v]) - 1
                comp_deg = popcount(uncolored & ~conflict_mask[v]) - 1
                if comp_deg > best_deg:
                    best_deg = comp_deg
                    best_v = v
                m ^= lb
            if best_v < 0:
                break
            # Color class: all uncolored vertices that don't conflict with best_v
            color_class = uncolored & ~conflict_mask[best_v]
            uncolored &= ~color_class
        return colors

    def branch_and_bound(initial_mask, time_limit):
        current = initial_mask
        current_size = popcount(current)
        best = initial_mask
        best_size = current_size

        # Vertices not in current set, that could potentially be added
        # We try to find a larger independent set by a bounded search.
        # Strategy: repeatedly try to add the "best" excluded vertex (fewest
        # conflicts with current) and if blocked, do a limited 1-for-2 swap.
        while time.time() < time_limit:
            # Compute an upper bound on the total achievable size
            remaining = 0
            for v in vertices:
                if not (current >> v) & 1:
                    remaining |= (1 << v)
            ub = current_size + color_bound(remaining)
            if ub <= best_size:
                break  # cannot beat best

            # Try to add a vertex with fewest conflicts in current set
            excluded = [v for v in vertices if not (current >> v) & 1]
            if not excluded:
                break
            # Sort excluded by number of conflicts with current (ascending)
            excluded.sort(key=lambda v: popcount(conflict_mask[v] & current))
            added = False
            for v in excluded[:15]:
                if can_add(v, current):
                    current |= (1 << v)
                    current_size += 1
                    if current_size > best_size:
                        best = current
                        best_size = current_size
                    added = True
                    break
            if not added:
                # Try a 1-for-2 swap to escape local optimum
                s_list = [v for v in vertices if (current >> v) & 1]
                random.shuffle(s_list)
                swap_done = False
                for u in s_list[:10]:
                    temp_mask = current & ~(1 << u)
                    addable = []
                    for v in vertices:
                        if v != u and not (current >> v) & 1 and can_add(v, temp_mask):
                            addable.append(v)
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        limit = min(len(addable), 15)
                        for i in range(limit):
                            for j in range(i + 1, limit):
                                v1, v2 = addable[i], addable[j]
                                if not ((conflict_mask[v1] >> v2) & 1):
                                    current = temp_mask | (1 << v1) | (1 << v2)
                                    current_size += 1
                                    if current_size > best_size:
                                        best = current
                                        best_size = current_size
                                    swap_done = True
                                    break
                            if swap_done:
                                break
                    if swap_done:
                        break
                if not swap_done:
                    break
        return best

    if time.time() < deadline - 3.0:
        best_mask = branch_and_bound(best_mask, deadline - 2.0)

    # ---------- Final validation ----------
    result_list = sorted(mask_to_set(best_mask))
    if not result_list:
        return list(range(1, 257, 2))
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                return list(range(1, 257, 2))

    return result_list
# EVOLVE-BLOCK-END
