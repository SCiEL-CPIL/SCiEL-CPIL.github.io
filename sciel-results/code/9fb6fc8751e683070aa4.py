"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmask(edges, adj):
        """Precompute conflict bitmasks: conflict[i] is an int bitmask of conflicting edges."""
        n_edges = len(edges)
        conflict = [0] * n_edges
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                elif (ui in adj.get(uj, set()) or ui in adj.get(vj, set()) or
                      vi in adj.get(uj, set()) or vi in adj.get(vj, set())):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def greedy_color_fast(edges, order, conflict_bm):
        """Greedy coloring with bitmask conflict checks."""
        n_edges = len(edges)
        colour = [0] * n_edges
        for idx in order:
            taken_mask = 0
            cm = conflict_bm[idx]
            # Collect taken colours by scanning all edges that conflict with idx
            # We need to know which colours are used by conflicting edges
            # Efficient approach: iterate through set bits in cm
            m = cm
            while m:
                low_bit = m & (-m)
                j = low_bit.bit_length() - 1
                taken_mask |= (1 << colour[j])
                m ^= low_bit
            c = 0
            while (taken_mask >> c) & 1:
                c += 1
            colour[idx] = c
        return colour

    def local_improve_fast(edges, colour, conflict_bm, max_iter=80000):
        """
        Fast local improvement using bitmask conflicts.
        Focus on most crowded colour class, try single moves and 2-edge swaps.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return colour, 0

        current_max = max(colour)

        # Maintain per-colour member lists for fast access
        colour_members = defaultdict(list)
        for i in range(n_edges):
            colour_members[colour[i]].append(i)

        def get_top_colour():
            """Return the colour with the most members; tie-break by highest colour."""
            best_c, best_n = -1, -1
            for c, members in colour_members.items():
                if len(members) > best_n or (len(members) == best_n and c > best_c):
                    best_n = len(members)
                    best_c = c
            return best_c

        def try_move_edge(e_idx, target_c, current_c):
            """Try to move edge e_idx from current_c to target_c. Returns True if successful."""
            # Check if target_c conflicts with any neighbour
            cm = conflict_bm[e_idx]
            m = cm
            while m:
                low_bit = m & (-m)
                j = low_bit.bit_length() - 1
                if colour[j] == target_c:
                    return False
                m ^= low_bit
            # Perform the move
            colour[e_idx] = target_c
            colour_members[current_c].remove(e_idx)
            if not colour_members[current_c]:
                del colour_members[current_c]
            colour_members[target_c].append(e_idx)
            return True

        improved = True
        iterations = 0

        while improved and iterations < max_iter and time.time() < deadline - 3:
            improved = False
            iterations += 1

            if current_max <= 0:
                break

            focus = get_top_colour()
            if focus < 0:
                break

            members = colour_members[focus]
            if not members:
                continue

            random.shuffle(members)

            # --- Single-edge moves ---
            for e_idx in members[:20]:  # Limit to 20 candidates for speed
                if colour[e_idx] != focus:
                    continue
                # Find lowest available colour below focus
                for new_c in range(focus):
                    if try_move_edge(e_idx, new_c, focus):
                        improved = True
                        if focus not in colour_members:
                            while current_max >= 0 and current_max not in colour_members:
                                current_max -= 1
                        break
                if improved:
                    break

            # --- Two-edge swaps ---
            if not improved and current_max > 0 and time.time() < deadline - 3:
                focus = get_top_colour()
                if focus >= 0 and focus in colour_members:
                    members = colour_members[focus]
                    if len(members) >= 2:
                        random.shuffle(members)
                        sub = members[:15]
                        for a in sub:
                            if improved:
                                break
                            if colour[a] != focus:
                                continue
                            # Find colours a can move to (below focus)
                            avail_a = []
                            for c in range(focus):
                                cm = conflict_bm[a]
                                m = cm
                                conflict_with_c = False
                                while m:
                                    low_bit = m & (-m)
                                    j = low_bit.bit_length() - 1
                                    if colour[j] == c:
                                        conflict_with_c = True
                                        break
                                    m ^= low_bit
                                if not conflict_with_c:
                                    avail_a.append(c)
                            if not avail_a:
                                continue

                            for b in sub:
                                if a == b or colour[b] != focus:
                                    continue
                                # Find colours b can move to (below focus, not used by a's new colour)
                                for ca in avail_a[:5]:
                                    # b cannot use ca (a will occupy it)
                                    # b needs cb != ca, cb < focus, cb not conflicting with b
                                    cm_b = conflict_bm[b]
                                    for cb in range(focus):
                                        if cb == ca:
                                            continue
                                        # Check if cb conflicts with b (excluding a)
                                        m = cm_b
                                        conflict_with_cb = False
                                        while m:
                                            low_bit = m & (-m)
                                            j = low_bit.bit_length() - 1
                                            if j == a:
                                                m ^= low_bit
                                                continue
                                            if colour[j] == cb:
                                                conflict_with_cb = True
                                                break
                                            m ^= low_bit
                                        if not conflict_with_cb:
                                            # Perform swap
                                            colour[a] = ca
                                            colour[b] = cb
                                            colour_members[focus].remove(a)
                                            colour_members[focus].remove(b)
                                            if not colour_members[focus]:
                                                del colour_members[focus]
                                            colour_members[ca].append(a)
                                            colour_members[cb].append(b)
                                            if focus not in colour_members:
                                                while current_max >= 0 and current_max not in colour_members:
                                                    current_max -= 1
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break

        if colour:
            current_max = max(colour)
        return colour, current_max + 1

    def try_multiple_orders_fast(edges, conflict_bm, n_attempts=15):
        """Try multiple edge orderings and keep the best coloring."""
        n_edges = len(edges)
        # Compute constraint degrees from bitmask
        constraint_deg = [bin(conflict_bm[i]).count('1') for i in range(n_edges)]

        best_colour = None
        best_max = float('inf')

        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break

            if attempt == 0:
                order = sorted(range(n_edges), key=lambda i: -constraint_deg[i])
            elif attempt == 1:
                order = sorted(range(n_edges), key=lambda i: constraint_deg[i])
            elif attempt == 2:
                order = list(range(n_edges))
                random.shuffle(order)
            elif attempt == 3:
                # Randomized priority: high constraint degree + random noise
                order = sorted(range(n_edges), key=lambda i: -constraint_deg[i] + random.random() * 5)
            elif attempt == 4:
                # Weighted random selection
                weights = [constraint_deg[i] + 1 for i in range(n_edges)]
                order = []
                remaining = list(range(n_edges))
                rem_weights = weights[:]
                for _ in range(n_edges):
                    if not remaining:
                        break
                    total_w = sum(rem_weights)
                    r = random.random() * total_w
                    cum = 0
                    chosen = None
                    for idx, w in enumerate(rem_weights):
                        cum += w
                        if r <= cum:
                            chosen = idx
                            break
                    if chosen is None:
                        chosen = len(rem_weights) - 1
                    order.append(remaining[chosen])
                    rem_weights.pop(chosen)
                    remaining.pop(chosen)
            else:
                # More random shuffles with slight bias
                order = list(range(n_edges))
                random.shuffle(order)
                # Sort by a mix of constraint degree and random
                order.sort(key=lambda i: -constraint_deg[i] * 0.5 + random.random() * 3)

            colour = greedy_color_fast(edges, order, conflict_bm)
            max_c = max(colour) + 1

            if max_c < best_max:
                best_max = max_c
                best_colour = colour[:]

        return best_colour, best_max

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    # Allocate time per instance: first gets more, later ones less if time is tight
    time_per_instance = [40.0, 40.0, 40.0]

    for spec_idx, spec in enumerate(specs):
        instance_start = time.time()
        instance_deadline = instance_start + time_per_instance[spec_idx]

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict_bm = build_conflict_bitmask(edges, adj)

        # Try multiple orderings.
        colour, num_colors = try_multiple_orders_fast(edges, conflict_bm, n_attempts=15)

        # Local improvement (main pass).
        if time.time() < instance_deadline - 5:
            colour, num_colors = local_improve_fast(edges, colour, conflict_bm, max_iter=80000)

        # High-precision polish pass.
        if time.time() < instance_deadline - 2:
            colour, num_colors = local_improve_fast(edges, colour, conflict_bm, max_iter=40000)

        result = [colour[i] for i in range(n_edges)]
        out.append(result)

        # Adjust remaining time allocation
        elapsed = time.time() - instance_start
        remaining_budget = deadline - time.time() - 5
        if spec_idx < 2:
            time_per_instance[spec_idx + 1] = max(20.0, remaining_budget / (2 - spec_idx))

    # Fill remaining with simple greedy if time ran out.
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict_bm = build_conflict_bitmask(edges, adj)

        colour = greedy_color_fast(edges, list(range(n_edges)), conflict_bm)
        out.append(colour)

    return out
# EVOLVE-BLOCK-END
