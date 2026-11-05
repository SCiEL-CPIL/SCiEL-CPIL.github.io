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

    # ---------- Exact maximum independent set via bitset branch-and-bound ----------
    # Uses Tomita-style coloring upper bound on the induced subgraph.
    # The conflict graph is sparse enough (256 vertices) that this can find
    # or near-find the true MIS within the time budget.

    def exact_mis(time_limit):
        """Find a (near-)maximum independent set using exact B&B with coloring bound."""
        # We want an independent set in the conflict graph, i.e., a clique in the
        # complement. Standard Tomita algorithm finds max clique; we adapt it
        # to find max independent set by working on the conflict graph directly.
        #
        # Upper bound: greedy coloring of the induced subgraph on candidate set.
        # In the conflict graph, a color class is an independent set (no conflicts).
        # The number of colors upper-bounds the max independent set size... wait,
        # no. Let me think carefully.
        #
        # Actually, for maximum independent set in graph G:
        # - The complement graph G_bar has cliques = independent sets of G
        # - Tomita's coloring bound applies to cliques: color G_bar, #colors >= clique size
        # - Equivalently, color G, and each color class is an independent set of G
        #   but that doesn't directly bound the max independent set.
        #
        # Better approach: work directly. For a candidate set C, the upper bound
        # on the max independent set within C is the number of colors in a greedy
        # coloring of the *complement* subgraph on C (where edges in complement =
        # non-edges in conflict graph = compatible pairs).
        #
        # Actually the cleanest: use the standard max-clique algorithm on the
        # complement graph. Two vertices are adjacent in complement iff they
        # DON'T conflict. Max clique in complement = max independent set in conflict.
        #
        # But building the complement explicitly is expensive. Instead, let's use
        # the following: in the conflict graph, for a candidate set C, we can
        # greedily color C such that each color class is an independent set in
        # the conflict graph (i.e., all vertices in a color class are mutually
        # non-conflicting). The number of colors is an UPPER BOUND on the size
        # of the maximum clique in the complement = maximum independent set in conflict.
        # Wait no - that's wrong. The number of colors in a proper coloring of G
        # is an upper bound on the clique number of G_bar, which equals the
        # independence number of G. Yes! That's correct.
        #
        # So: greedy-color the conflict graph induced by C. #colors >= alpha(G[C]).
        # This gives a valid upper bound.

        best_set = [0] * (N + 1)  # 1-indexed, best_set[v] = 1 if in best IS
        best_size = [0]

        def color_bound(cand_mask):
            """Greedy coloring of conflict graph induced by cand_mask.
            Returns (num_colors, color_order) where color_order is a list of
            vertices ordered by color (last color first) for branch ordering."""
            if cand_mask == 0:
                return 0, []
            colors = 0
            uncolored = cand_mask
            color_classes = []
            while uncolored:
                colors += 1
                # Greedy: pick vertices that are independent (no conflicts among themselves)
                # Process in order of decreasing degree in remaining uncolored subgraph
                # to get tighter bounds
                class_mask = 0
                m = uncolored
                # Order by degree (descending) for tighter coloring
                verts_in_uncolored = []
                tmp = uncolored
                while tmp:
                    lb = tmp & (-tmp)
                    v = lb.bit_length() - 1
                    deg = popcount(conflict_mask[v] & uncolored)
                    verts_in_uncolored.append((deg, v))
                    tmp ^= lb
                verts_in_uncolored.sort(reverse=True)

                for deg, v in verts_in_uncolored:
                    if not (uncolored >> v) & 1:
                        continue
                    # Check if v can join current color class (no conflict with class members)
                    if (conflict_mask[v] & class_mask) == 0:
                        class_mask |= (1 << v)
                        uncolored &= ~(1 << v)
                color_classes.append(class_mask)
            # Return color classes in reverse (last color = highest, try those first)
            return colors, list(reversed(color_classes))

        def branch(cand_mask, current_mask, current_size):
            if time.time() >= time_limit:
                return
            if cand_mask == 0:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                    for v in vertices:
                        if (current_mask >> v) & 1:
                            best_set[v] = 1
                        else:
                            best_set[v] = 0
                return

            # Upper bound via coloring
            ub, color_classes = color_bound(cand_mask)
            if current_size + ub <= best_size[0]:
                return

            # Branch: try vertices from the last color class first (highest color)
            # These are the most "constrained" and likely part of a large IS
            for cc in color_classes:
                if time.time() >= time_limit:
                    return
                m = cc
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    m ^= lb
                    if not (cand_mask >> v) & 1:
                        continue
                    # Include v in the independent set
                    new_current = current_mask | (1 << v)
                    new_cand = cand_mask & ~(1 << v) & ~conflict_mask[v]
                    # Remaining candidates must not conflict with v
                    branch(new_cand, new_current, current_size + 1)
                    # Exclude v
                    cand_mask &= ~(1 << v)

        # Start with all vertices as candidates
        all_mask = 0
        for v in vertices:
            all_mask |= (1 << v)

        branch(all_mask, 0, 0)

        if best_size[0] > 0:
            result = 0
            for v in vertices:
                if best_set[v]:
                    result |= (1 << v)
            return result
        return 0

    if time.time() < deadline - 4.0:
        exact_result = exact_mis(deadline - 3.0)
        if popcount(exact_result) > popcount(best_mask):
            best_mask = exact_result

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
