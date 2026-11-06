import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # --- GF(16) base for known n=32 solution ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def base_32():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    # --- Bitset helpers ---
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def count_mono_triangles(adj, n):
        count = 0
        for c in range(4):
            for i in range(n):
                neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
                j_mask = neigh
                while j_mask:
                    j_bit = j_mask & (-j_mask)
                    j = j_bit.bit_length() - 1
                    j_mask ^= j_bit
                    common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                    count += common.bit_count()
        return count

    def find_one_mono_triangle(adj, n):
        for c in range(4):
            for i in range(n):
                neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
                j_mask = neigh
                while j_mask:
                    j_bit = j_mask & (-j_mask)
                    j = j_bit.bit_length() - 1
                    j_mask ^= j_bit
                    common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                    if common:
                        k_bit = common & (-common)
                        k = k_bit.bit_length() - 1
                        return (i, j, k, c)
        return None

    def color_index(n, i, j):
        """Index in flat colors list for pair (i,j), i<j, n vertices."""
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # --- Slack-based greedy construction ---
    def greedy_slack_build(n, rng, time_budget):
        """Build a coloring using slack-maximizing greedy vertex ordering."""
        start = time.time()
        m = n * (n - 1) // 2
        colors = [-1] * m
        adj = [[0] * n for _ in range(4)]

        # Random vertex ordering
        order = list(range(n))
        rng.shuffle(order)

        for v in order:
            if time.time() - start > time_budget:
                return None

            # For each color, compute slack: minimum over neighbors u of
            # (number of colors that would create a mono triangle if used for (v,u))
            # Actually: for color c, slack = min over u in already-placed neighbors
            # of (how many "forbidden" constraints)
            # Simpler: for each color c, count how many edges (u,v) with u placed
            # would create a mono triangle if colored c. Pick color with min count.
            # But we want to MAXIMIZE slack = distance to mono triangle.
            # Slack for color c at vertex v = min over u placed of
            #   (number of common neighbors w of u and v in color c, excluding v)
            #   but that's expensive. Use: for color c, the number of u placed
            #   such that edge (u,v) colored c would create a mono triangle.
            #   i.e., count of u where exists w < u with color(u,w)=c and color(w,v)=c... 
            #   Actually simpler: for each candidate color c, the "risk" is
            #   the number of placed vertices u where coloring (u,v) with c
            #   would immediately create a mono triangle (i.e., there exists w
            #   already placed with color(u,w)=c and color(w,v) already =c).
            #   But color(w,v) isn't set yet if w is placed but v is current.
            #   
            # Let me use a different slack: for color c, risk = number of placed u
            # such that adj[c][u] has at least one neighbor w that is also a neighbor
            # of v in adj[c] (excluding v itself, which isn't set yet).
            # This is: for each u placed, (adj[c][u] & adj[c][v]).bit_count() but
            # adj[c][v] only has edges to previously placed vertices.
            # Actually adj[c][v] is being built incrementally.
            # 
            # Simplified risk: for color c, sum over placed u of
            #   (adj[c][u] & adj[c][v]).bit_count()
            # where adj[c][v] contains only edges to already-placed vertices.
            # Lower risk = more slack.

            best_c = -1
            best_risk = float('inf')
            for c in range(4):
                risk = 0
                for u in order:
                    if colors[color_index(n, min(u,v), max(u,v))] != -1 and u != v:
                        # u is placed and edge (u,v) not yet colored (we're coloring it now)
                        # Wait, we're deciding the color of (u,v) for all u.
                        # Actually we assign one color per edge (u,v).
                        pass
                # For each u already placed (in order before v), the edge (u,v) gets color c.
                # Risk contribution from u: number of w (placed, w != u, w != v) 
                # such that color(u,w)=c and color(w,v)=c. But color(w,v) is also c (we're assigning).
                # So risk from u = (adj[c][u] & adj[c][v]).bit_count() where adj[c][v] 
                # only includes edges to vertices placed before v in the order.
                # But we haven't set adj[c][v] yet for the current assignment.
                # Let me compute: for each u placed before v, 
                #   common = adj[c][u] & (mask of vertices placed before v in order, excluding u)
                #   risk += common.bit_count()
                # But this is O(n^2) per color per vertex = O(n^3) total, too slow for n=50.
                
                # Faster: precompute for each color c and vertex v, as we go,
                # the set of placed neighbors. Actually let's just use the simple
                # greedy: pick color that minimizes immediate mono-triangle creation.
                # For color c, immediate risk = number of u placed where
                #   exists w placed with w < u such that color(u,w)=c and color(w,v)=c.
                # Since color(w,v) = c (we're assigning), this is:
                #   number of u placed where (adj[c][u] & adj[c][v]).bit_count() > 0
                # but adj[c][v] isn't set yet. 
                # 
                # Let me just use: for each color c, risk = sum over u placed
                #   of (adj[c][u] & placed_mask_excluding_u_and_v).bit_count()
                # where placed_mask includes all vertices placed before v.
                # This is a proxy for "how many mono-triangle opportunities exist."
                
                risk = 0
                for u in order:
                    if u == v:
                        continue
                    # Check if u was placed before v in the order
                    # Actually, we process vertices in order, so when we're at v,
                    # all u in order before v are "placed." But we need to know
                    # which u are before v in the order.
                    pass
                
                # This is getting complicated. Let me use a simpler heuristic:
                # For each color c, compute the number of placed vertices u
                # such that u has at least one neighbor w (placed, w != u) 
                # with color(u,w) = c. This is just the degree of u in color c
                # among placed vertices. Sum this over all u. Lower = safer.
                risk = 0
                for u in range(n):
                    if u == v:
                        continue
                    # Is u placed before v? In our order, yes if u appears before v.
                    # We track placed vertices.
                    # For now, assume all u < v in the order are placed.
                    # Let me restructure: process in order, track placed set.
                    pass
                
                # I'm overcomplicating. Let me use the standard approach:
                # For color c, the risk is the number of pairs (u,w) both placed
                # before v, with color(u,w)=c, such that both u and w are "connected"
                # to v in a way that would create a triangle. Specifically, if we
                # color (u,v) and (w,v) with c, and color(u,w)=c, we get a mono triangle.
                # But we color all edges (u,v) with the SAME color c (one color per vertex v).
                # So risk = number of pairs (u,w) placed before v with color(u,w)=c,
                # where u and w are both "visible" to v. Since all placed vertices are
                # connected to v, risk = number of mono edges in color c among placed vertices.
                # That's the same for all c? No, it's the count of color-c edges among placed.
                # 
                # Actually the correct risk: if we assign color c to ALL edges (u,v) for u placed,
                # then a mono triangle (u,w,v) is created iff color(u,w)=c.
                # So risk = number of pairs (u,w) both placed before v with color(u,w)=c.
                # This is just the number of color-c edges in the subgraph of placed vertices.
                # We want to minimize this. But it doesn't depend on v!
                # 
                # I see the issue: the risk DOES depend on v because not all placed
                # vertices are necessarily "close" to v in terms of creating triangles.
                # But in a complete graph, all pairs are connected. So the risk is
                # indeed just the number of color-c edges among placed vertices,
                # which is independent of v. That can't be right for a useful heuristic.
                #
                # The real constraint: for each u placed, edge (u,v) gets color c.
                # A mono triangle (u,w,v) exists iff color(u,w)=c AND color(w,v)=c.
                # Since color(w,v)=c for all w placed (same color c), 
                # mono triangles = number of u placed with color(u,w)=c for some w placed.
                # More precisely: for each u placed, if there exists w placed (w != u)
                # with color(u,w)=c, then (u,w,v) is a mono triangle.
                # So risk = number of u placed that have at least one color-c neighbor
                # among other placed vertices.
                # This DOES depend on the structure, not just the count.
                
                risk = 0
                for u in range(n):
                    if u == v:
                        continue
                    # Check if u is placed before v in the order
                    # We need to know the position of u and v in the order.
                    # Let me precompute positions.
                    pass

            # This is getting too complex for inline. Let me use a simpler but effective
            # greedy: for each vertex v, pick the color c that minimizes the number of
            # mono triangles that would be created. We can compute this incrementally.
            #
            # New approach: maintain adj as we go. For vertex v, for each color c:
            #   new_triangles = 0
            #   for each u already placed:
            #     if we color (u,v) with c, new triangles = number of w placed
            #     with color(u,w)=c and color(w,v)=c. But color(w,v) is also c.
            #     So new_triangles += (adj[c][u] & adj[c][v]).bit_count()
            #     where adj[c][v] only includes edges to previously placed vertices.
            #   But we color ALL (u,v) with the same c, so:
            #   total_new = sum over u placed of (adj[c][u] & adj[c][v]).bit_count()
            #   But adj[c][v] is the same for all u, so:
            #   total_new = adj[c][v] & (OR of adj[c][u] for all u placed)
            #   Actually no, it's sum of bit_counts, not OR.
            #   Hmm, but (adj[c][u] & adj[c][v]).bit_count() counts w where
            #   w is neighbor of both u and v in color c. Summing over u counts
            #   each w multiple times (once per u that connects to w).
            #   That's actually what we want: each pair (u,w) with color(u,w)=c
            #   and w neighbor of v in color c contributes 1.
            #
            # Let me just compute it directly.

            # Restructure: process vertices in order, maintain adj incrementally.
            # For each new vertex v, for each color c, compute:
            #   risk_c = sum over u (placed before v) of (adj[c][u] & adj[c][v]).bit_count()
            #   where adj[c][v] = adj[c][v] & mask_of_placed_before_v
            # Pick c with min risk_c.

            # Let me redo the whole function more carefully.

        return None

    # --- Better slack-based greedy ---
    def greedy_slack_v2(n, rng, time_budget):
        """Greedy construction with slack-based color selection."""
        start = time.time()
        m = n * (n - 1) // 2
        colors = [-1] * m
        adj = [[0] * n for _ in range(4)]
        placed = [False] * n

        order = list(range(n))
        rng.shuffle(order)

        for vi, v in enumerate(order):
            if time.time() - start > time_budget:
                return None
            placed[v] = True

            # For each color c, compute risk:
            # If we color all edges (u,v) for u placed with color c,
            # new mono triangles = number of pairs (u,w) both placed before v
            # such that color(u,w)=c and w is connected to v in color c.
            # But since we color (w,v) with c too, every color-c edge (u,w) among
            # placed vertices creates a triangle (u,w,v).
            # So risk_c = number of color-c edges among vertices placed before v.
            # Wait, that's independent of v. Let me think again.
            #
            # Actually: for each u placed before v, edge (u,v) gets color c.
            # For each w placed before v, edge (w,v) gets color c.
            # Triangle (u,w,v) is mono iff color(u,w)=c.
            # So risk_c = number of pairs (u,w) with u,w placed before v, u<w,
            # and color(u,w)=c.
            # This is indeed independent of v (only depends on the set of placed vertices).
            # So the greedy choice is the same for every v? That's degenerate.
            #
            # The issue is that we're assigning ONE color per vertex to ALL its edges.
            # That's a very restrictive model. Let me instead assign colors per edge.
            #
            # Revised: for vertex v, assign a color to each edge (u,v) independently.
            # For edge (u,v), color c is forbidden if there exists w placed
            # with color(u,w)=c and color(w,v)=c. But color(w,v) is also being assigned.
            # This is a CSP. Let me use a simpler per-edge greedy.

        return None

    # --- Per-edge greedy with slack ---
    def greedy_per_edge(n, rng, time_budget):
        """Assign colors edge-by-edge in a smart order, using slack heuristic."""
        start = time.time()
        m = n * (n - 1) // 2
        colors = [-1] * m
        adj = [[0] * n for _ in range(4)]

        # Build edge list in an order that prioritizes "constrained" edges first.
        # Start with edges in the known 32-vertex structure, then add others.
        # For simplicity, use a random order with some structure.

        # Actually, let's build vertices one at a time, but assign each edge (u,v)
        # individually with a greedy choice that minimizes immediate mono-triangle risk.

        for v in range(1, n):
            if time.time() - start > time_budget:
                return None

            # For each u < v, assign color to edge (u,v)
            # Process u in random order
            us = list(range(v))
            rng.shuffle(us)

            for u in us:
                if time.time() - start > time_budget:
                    return None

                # For each color c, compute risk:
                # Number of w < u such that color(u,w)=c and color(w,v)=c.
                # But color(w,v) may not be assigned yet if w > u in our processing order.
                # Since we process u in random order, some (w,v) may be unassigned.
                # Only consider w where color(w,v) is already assigned.
                best_c = -1
                best_risk = float('inf')
                for c in range(4):
                    risk = 0
                    for w in range(min(u, v)):
                        # Check if color(w,v) is assigned
                        idx_wv = color_index(n, min(w, v), max(w, v))
                        if colors[idx_wv] == -1:
                            continue
                        if colors[idx_wv] != c:
                            continue
                        # color(w,v) = c, check if color(u,w) = c
                        idx_uw = color_index(n, min(u, w), max(u, w))
                        if colors[idx_uw] == c:
                            risk += 1
                    if risk < best_risk:
                        best_risk = risk
                        best_c = c
                    if best_risk == 0:
                        break

                if best_c == -1:
                    best_c = rng.randint(0, 3)

                idx_uv = color_index(n, u, v)
                colors[idx_uv] = best_c
                adj[best_c][u] |= (1 << v)
                adj[best_c][v] |= (1 << u)

        return colors

    # --- Local search to fix mono triangles ---
    def local_search_fix(colors, n, time_budget):
        """Fix mono triangles via local search."""
        start = time.time()
        m = n * (n - 1) // 2
        adj = build_adj(colors, n)
        tri_count = count_mono_triangles(adj, n)

        if tri_count == 0:
            return colors

        no_improve = 0
        max_no_improve = 100

        while no_improve < max_no_improve:
            if time.time() - start > time_budget:
                break

            tri = find_one_mono_triangle(adj, n)
            if tri is None:
                return colors

            i, j, k, c = tri
            edges = [(i, j), (i, k), (j, k)]
            best_delta = float('inf')
            best_edge = None
            best_new_c = -1

            for u, v in edges:
                for new_c in range(4):
                    if new_c == c:
                        continue
                    removed = (adj[c][u] & adj[c][v]).bit_count()
                    added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    delta = added - removed
                    if delta < best_delta:
                        best_delta = delta
                        best_edge = (u, v)
                        best_new_c = new_c

            if best_edge is not None and best_delta < 0:
                u, v = best_edge
                old_c = c
                new_c = best_new_c
                idx = color_index(n, u, v)
                colors[idx] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += best_delta
                no_improve = 0
                if tri_count == 0:
                    return colors
            else:
                no_improve += 1

        return colors if tri_count == 0 else None

    # --- Simulated annealing ---
    def simulated_annealing(colors, n, time_budget, rng):
        """SA to eliminate mono triangles."""
        start = time.time()
        m = n * (n - 1) // 2
        adj = build_adj(colors, n)
        tri_count = count_mono_triangles(adj, n)

        if tri_count == 0:
            return colors

        T = 3.0
        T_min = 0.05
        cooling = 0.9998

        while T > T_min:
            if time.time() - start > time_budget:
                break

            ri = rng.randint(0, m - 1)
            old_c = colors[ri]
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue

            # Find edge (u,v) for index ri
            # ri = i*(n-1) - i*(i+1)//2 + (j-i-1)
            # Solve for i: i*(n-1) - i*(i+1)//2 <= ri < (i+1)*(n-1) - (i+1)*(i+2)//2
            i = 0
            while i < n - 1:
                base = i * (n - 1) - i * (i + 1) // 2
                base_next = (i + 1) * (n - 1) - (i + 1) * (i + 2) // 2
                if base <= ri < base_next:
                    break
                i += 1
            j = i + 1 + (ri - (i * (n - 1) - i * (i + 1) // 2))

            removed = (adj[old_c][i] & adj[old_c][j]).bit_count()
            added = (adj[new_c][i] & adj[new_c][j]).bit_count()
            delta = added - removed

            if delta <= 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                colors[ri] = new_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                tri_count += delta
                if tri_count == 0:
                    return colors

            T *= cooling

        return colors if tri_count == 0 else None

    # --- Main strategy ---
    rng = random.Random(42)

    # Start with known n=32 solution
    best_colors = base_32()
    best_n = 32

    # Phase 1: Try to extend from 32 using per-edge greedy + local search
    for target_n in range(33, 51):
        if time_left() < 20:
            break

        # Try multiple random restarts of greedy construction for target_n
        found = None
        for restart in range(10):
            if time_left() < 20:
                break
            time_budget = min(8.0, max(3.0, time_left() - 20))
            colors = greedy_per_edge(target_n, rng, time_budget)
            if colors is None:
                continue

            # Check if valid
            adj = build_adj(colors, target_n)
            tri_count = count_mono_triangles(adj, target_n)
            if tri_count == 0:
                found = colors
                break

            # Try to fix with local search
            time_budget2 = min(5.0, max(2.0, time_left() - 20))
            fixed = local_search_fix(colors, target_n, time_budget2)
            if fixed is not None:
                found = fixed
                break

            # Try SA
            time_budget3 = min(5.0, max(2.0, time_left() - 20))
            sa_result = simulated_annealing(colors, target_n, time_budget3, rng)
            if sa_result is not None:
                found = sa_result
                break

        if found is not None:
            best_n = target_n
            best_colors = found
        else:
            # Can't extend further, try a different approach
            break

    # Phase 2: If stuck below 50, try direct SA from random initializations
    if best_n < 50 and time_left() > 30:
        for target in [40, 45, 50]:
            if time_left() < 30:
                break
            if target <= best_n:
                continue

            found = None
            for restart in range(20):
                if time_left() < 30:
                    break
                time_budget = min(10.0, max(5.0, time_left() - 30))
                # Random initial coloring
                m = target * (target - 1) // 2
                colors = [rng.randint(0, 3) for _ in range(m)]
                result = simulated_annealing(colors, target, time_budget, rng)
                if result is not None:
                    found = result
                    break

            if found is not None:
                best_n = target
                best_colors = found
                break

    # Final verification
    adj_final = build_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        # Fallback to known valid 32
        return {'n': 32, 'colors': base_32()}

    return {'n': best_n, 'colors': best_colors}
