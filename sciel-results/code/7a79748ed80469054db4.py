"""Strong edge colourings via greedy + local merging."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def build_adj(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def precompute_conflicts(edges, adj):
        """Precompute the conflict set (near vertices) for each edge."""
        conflicts = []
        for u, v in edges:
            near = {u, v}
            near.update(adj[u])
            near.update(adj[v])
            conflicts.append(frozenset(near))
        return conflicts

    def greedy_color(edges, adj, conflicts, order):
        """Greedy colouring with given edge order. Returns dict edge->colour."""
        m = len(edges)
        colour = [-1] * m
        # Track for each colour which vertices are "near" (conflict set union)
        colour_near = []  # colour_near[c] = set of vertices that conflict with colour c
        for idx in order:
            e_near = conflicts[idx]
            best_c = 0
            while True:
                if best_c < len(colour_near):
                    # Check if any vertex in e_near is in colour_near[best_c]
                    if e_near & colour_near[best_c]:
                        best_c += 1
                        continue
                # Assign best_c
                break
            colour[idx] = best_c
            if best_c == len(colour_near):
                colour_near.append(set())
            colour_near[best_c] |= e_near
        return colour, colour_near

    def try_merge(colour, colour_near, conflicts, c1, c2, edge_to_idx):
        """Try to merge colour classes c1 and c2. Returns (new_colour, new_colour_near) or (None, None)."""
        # Collect edges in c1 and c2
        cls1 = [i for i, c in enumerate(colour) if c == c1]
        cls2 = [i for i, c in enumerate(colour) if c == c2]
        if not cls1 or not cls2:
            return colour, colour_near
        # Check cross-conflicts: for each edge in cls1, its near set must not intersect any edge in cls2's near set
        # Efficient: union of near sets for cls2
        near2 = set()
        for i in cls2:
            near2 |= conflicts[i]
        for i in cls1:
            if conflicts[i] & near2:
                return None, None
        # Merge: reassign cls2 to c1
        new_colour = list(colour)
        for i in cls2:
            new_colour[i] = c1
        # Update colour_near: merge near2 into colour_near[c1], remove c2
        new_colour_near = [set(s) for s in colour_near]
        new_colour_near[c1] |= near2
        # Mark c2 as empty
        if c2 < len(new_colour_near):
            new_colour_near[c2] = set()
        return new_colour, new_colour_near

    def local_improve(colour, colour_near, conflicts, max_iters=200):
        """Try to merge colour classes to reduce total colour count."""
        current_colour = list(colour)
        current_near = [set(s) for s in colour_near]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Get all colour values in use (non-empty classes)
            colours_used = sorted(set(c for c in current_colour if c >= 0 and current_near[c]))
            ncolours = len(colours_used)
            if ncolours <= 1:
                break
            # Try merging pairs
            for i in range(ncolours):
                for j in range(i + 1, ncolours):
                    c1, c2 = colours_used[i], colours_used[j]
                    new_colour, new_near = try_merge(current_colour, current_near, conflicts, c1, c2)
                    if new_colour is not None:
                        current_colour = new_colour
                        current_near = new_near
                        improved = True
                        break
                if improved:
                    break
        return current_colour, current_near

    def total_colours(colour, colour_near):
        return len(set(c for c in colour if c >= 0 and colour_near[c]))

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        conflicts = precompute_conflicts(edges, adj)
        m = len(edges)

        best_colour = None
        best_count = float('inf')

        # Precompute conflict degree for each edge (number of other edges that conflict)
        # Edge i conflicts with edge j if their near sets intersect
        # Approximate: use |near_i| as a proxy for conflict degree
        conflict_deg = [len(conf) for conf in conflicts]

        # Strategy 1: Conflict-degree-sorted order (high conflict first)
        order_cd = sorted(range(m), key=lambda i: -conflict_deg[i])
        if time.time() < deadline:
            colour, colour_near = greedy_color(edges, adj, conflicts, order_cd)
            colour, colour_near = local_improve(colour, colour_near, conflicts, max_iters=300)
            count = total_colours(colour, colour_near)
            if count < best_count:
                best_count = count
                best_colour = colour

        # Strategy 2: Multiple random orderings
        num_trials = 40
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 1000 + trial)
            order = list(range(m))
            rng.shuffle(order)
            colour, colour_near = greedy_color(edges, adj, conflicts, order)
            colour, colour_near = local_improve(colour, colour_near, conflicts, max_iters=200)
            count = total_colours(colour, colour_near)
            if count < best_count:
                best_count = count
                best_colour = colour

        # Strategy 3: Deterministic order (by edge index)
        if time.time() < deadline:
            colour, colour_near = greedy_color(edges, adj, conflicts, list(range(m)))
            colour, colour_near = local_improve(colour, colour_near, conflicts, max_iters=300)
            count = total_colours(colour, colour_near)
            if count < best_count:
                best_count = count
                best_colour = colour

        # Strategy 4: Weighted order (conflict degree + random tiebreak)
        if time.time() < deadline:
            rng = random.Random(seed * 2000)
            weights = [(conflict_deg[i], rng.random()) for i in range(m)]
            order_w = sorted(range(m), key=lambda i: -weights[i][0])
            colour, colour_near = greedy_color(edges, adj, conflicts, order_w)
            colour, colour_near = local_improve(colour, colour_near, conflicts, max_iters=300)
            count = total_colours(colour, colour_near)
            if count < best_count:
                best_count = count
                best_colour = colour

        if best_colour is None:
            # Fallback: simple greedy
            colour, colour_near = greedy_color(edges, adj, conflicts, list(range(m)))
            best_colour = colour

        results.append([best_colour[i] for i in range(m)])

    return results
# EVOLVE-BLOCK-END
