import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # 1. Precompute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)

    # 2. Build adjacency bitmasks: adj[u] has bit v set iff u and v conflict
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask

    # 3. Greedy independent set construction
    def greedy_from_mask(P_mask):
        """Build an independent set from available vertices P_mask using min-degree heuristic."""
        current = 0
        available = P_mask
        while available:
            # Find candidate with minimum degree within available
            best_v = -1
            best_deg = m + 1
            for v in range(m):
                if (available >> v) & 1:
                    # Count neighbors of v that are also in available
                    deg = bin(available & adj[v]).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                        if deg == 0:
                            break
            if best_v == -1:
                break
            current |= (1 << best_v)
            # Remove best_v and its neighbors from available
            available &= ~(1 << best_v)
            available &= ~adj[best_v]
        return current

    # 4. Greedy coloring upper bound for branch-and-bound
    def color_bound(available_mask):
        """
        Compute a greedy coloring upper bound on the size of an independent set
        within available_mask. The number of colors used is an upper bound on
        the maximum independent set size (since an independent set can use at
        most one vertex per color).
        """
        colors = 0
        remaining = available_mask
        while remaining:
            colors += 1
            # Greedily pick a vertex and remove it and its neighbors (independent set for this color class)
            # Actually for coloring: we want to assign a color to a maximal set of non-adjacent vertices
            # But standard greedy coloring: pick a vertex, color it, remove it from uncolored
            # Upper bound = number of colors needed to color the subgraph
            # Greedy: repeatedly pick a vertex, remove it and all its neighbors from remaining
            # This gives an independent set per "round", and the number of rounds is an upper bound
            # Wait: that's actually a lower bound on chromatic number, not upper bound on independent set
            # 
            # Let me reconsider. For branch and bound on MIS:
            # Upper bound on |MIS| in a subgraph: use greedy coloring where we color the subgraph.
            # The chromatic number χ ≥ α (independence number), so χ is an upper bound on α.
            # Greedy coloring gives an upper bound on χ? No, greedy coloring gives a valid coloring,
            # so the number of colors used is an upper bound on χ.
            # So: upper_bound_on_MIS ≤ colors_used_in_greedy_coloring
            #
            # But actually we want: if current_size + colors_used ≤ best_size, prune.
            #
            # Let me implement greedy coloring properly:
            # Color the subgraph induced by available_mask.
            pass

        return colors

    # Better: implement a proper greedy coloring that returns the number of colors
    def greedy_color_count(available_mask):
        """
        Greedy coloring of the subgraph induced by available_mask.
        Returns the number of colors used, which is an upper bound on the
        maximum independent set size in that subgraph.
        """
        uncolored = available_mask
        colors = 0
        while uncolored:
            colors += 1
            # In this color class, greedily pick vertices that don't conflict with each other
            # Pick a vertex, add to this color class, remove it and its neighbors from uncolored
            # Wait, that's not right either. In coloring, adjacent vertices must have different colors.
            # So in one color class, we can only have non-adjacent vertices.
            # Greedy: pick a vertex v, assign it this color, remove v from uncolored.
            # But we can also assign the same color to other non-adjacent vertices.
            # Simple greedy: just pick one vertex per color? That would give |V| colors.
            # 
            # Standard greedy coloring: order vertices, assign smallest available color.
            # Let me do that.
            pass

        return colors

    # Let me use a different approach for the upper bound.
    # A simple upper bound: for each vertex v in available, 1 + max MIS in (available \ {v} \ N(v))
    # But that's recursive. For a quick bound, use:
    # Upper bound = sum over vertices of 1/(1+deg(v)) ... no that's not right.
    #
    # Actually, a well-known simple bound: α(G) ≤ n - |E|/2 ... not tight.
    #
    # Let me use a greedy approach: repeatedly pick a vertex of minimum degree,
    # and note that in any independent set, at most one vertex from {v} ∪ N(v) can be chosen.
    # So we can bound by: while there are vertices, pick min-degree vertex v,
    # add 1 to the bound, remove v and N(v) from consideration.
    # This gives a valid upper bound on the independent set size.
    
    def greedy_upper_bound(available_mask):
        """
        Upper bound on the size of a maximum independent set in the subgraph
        induced by available_mask. Uses the packing argument:
        repeatedly pick a vertex v, count 1, remove v and its neighbors.
        Each "round" accounts for at most one vertex in any independent set.
        """
        remaining = available_mask
        bound = 0
        while remaining:
            bound += 1
            # Pick vertex with minimum degree in remaining
            best_v = -1
            best_deg = m + 1
            for v in range(m):
                if (remaining >> v) & 1:
                    deg = bin(remaining & adj[v]).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            # Remove best_v and its neighbors from remaining
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return bound

    # 5. Branch and bound search
    best_size = 0
    best_mask = 0

    # Initialize with several greedy runs
    for _ in range(200):
        if time.time() > deadline:
            break
        # Randomize by permuting vertex order implicitly through random subset
        P = (1 << m) - 1
        sol = greedy_from_mask(P)
        size = sol.bit_count()
        if size > best_size:
            best_size = size
            best_mask = sol

    # Branch and bound with randomization
    def branch_and_bound(available_mask, current_mask, current_size, depth=0):
        nonlocal best_size, best_mask
        if time.time() > deadline:
            return

        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # Upper bound check
        ub = greedy_upper_bound(available_mask)
        if current_size + ub <= best_size:
            return

        # Pick a branching vertex: min degree in available
        best_v = -1
        best_deg = m + 1
        for v in range(m):
            if (available_mask >> v) & 1:
                deg = bin(available_mask & adj[v]).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v

        if best_v == -1:
            return

        # Branch 1: include best_v
        new_current = current_mask | (1 << best_v)
        new_available = available_mask & ~(1 << best_v) & ~adj[best_v]
        branch_and_bound(new_available, new_current, current_size + 1, depth + 1)

        # Branch 2: exclude best_v
        branch_and_bound(available_mask & ~(1 << best_v), current_mask, current_size, depth + 1)

    # Run B&B from the full graph
    full_mask = (1 << m) - 1
    branch_and_bound(full_mask, 0, 0)

    # 6. Local search: kick-based improvement
    # If B&B didn't improve much, do local search from best solution
    if time.time() < deadline - 30:
        iterations = 0
        while time.time() < deadline - 5:
            iterations += 1
            # Remove 1-4 random vertices from best_mask
            verts_in_set = [i for i in range(m) if (best_mask >> i) & 1]
            if not verts_in_set:
                break
            num_removals = min(random.randint(1, 4), len(verts_in_set))
            current_mask = best_mask
            to_remove = random.sample(verts_in_set, num_removals)
            for v in to_remove:
                current_mask &= ~(1 << v)

            # Greedily refill
            new_mask = greedy_from_mask(((1 << m) - 1) & ~current_mask) | current_mask
            new_size = new_mask.bit_count()
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask

    # 7. Extract result
    result = []
    for i in range(m):
        if (best_mask >> i) & 1:
            result.append(i)

    if not result:
        result = [0]

    return np.array(result, dtype=np.int64)
