import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n = 256

    # Build conflict adjacency as Python int bitmasks (bits 0..n, bit i means node i+1)
    # adj[i] is a bitmask of nodes that conflict with node i (i in 1..n)
    adj = [0] * (n + 1)

    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Greedy independent set (minimum degree heuristic) for initial lower bound
    def greedy_independent_set(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            # Find node with minimum degree in remaining
            min_deg = float('inf')
            chosen = -1
            t = remaining
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    chosen = node
                t &= t - 1
            if chosen == -1:
                break
            selected |= (1 << chosen)
            remaining &= ~(adj[chosen] | (1 << chosen))
        return selected

    all_mask = (1 << (n + 1)) - 2  # bits 1..n set

    # Initial lower bound with multiple random-order greedy tries
    best_set = greedy_independent_set(all_mask)
    best_size = best_set.bit_count()
    
    # Try a few different greedy strategies to improve initial bound
    for _ in range(3):
        # Greedy with max degree removal
        selected = 0
        remaining = all_mask
        while remaining:
            max_deg = -1
            chosen = -1
            t = remaining
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = node
                t &= t - 1
            if chosen == -1:
                break
            selected |= (1 << chosen)
            remaining &= ~(adj[chosen] | (1 << chosen))
        if selected.bit_count() > best_size:
            best_size = selected.bit_count()
            best_set = selected

    # Greedy coloring upper bound on the candidate subgraph
    # Returns an upper bound on the independence number of the subgraph induced by candidates
    def color_bound(candidates):
        """Greedy coloring of the conflict graph restricted to candidates.
        Returns the number of colors used, which upper-bounds the independence number."""
        if candidates == 0:
            return 0
        uncolored = candidates
        color_count = 0
        while uncolored:
            color_count += 1
            # Greedily pick an independent set for this color class
            color_class = 0
            t = uncolored
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                # Node can join this color if it doesn't conflict with any node already in color_class
                if (adj[node] & color_class) == 0:
                    color_class |= lsb
                t &= t - 1
            uncolored &= ~color_class
        return color_count

    # Branch and bound with coloring upper bound
    def branch_and_bound(candidates, current_size, selected_mask):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = selected_mask
            return

        # Quick bound: can't beat best
        if current_size + candidates.bit_count() <= best_size:
            return

        # Upper bound via greedy coloring
        ub = current_size + color_bound(candidates)
        if ub <= best_size:
            return

        # Pick branching vertex: highest degree in candidate subgraph
        max_deg = -1
        chosen_node = -1
        t = candidates
        while t:
            lsb = t & (-t)
            node = lsb.bit_length() - 1
            deg = (adj[node] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                chosen_node = node
            t &= t - 1

        if chosen_node == -1:
            return

        chosen_bit = 1 << chosen_node

        # Branch 1: Include chosen_node
        new_candidates = candidates & ~(adj[chosen_node] | chosen_bit)
        new_selected = selected_mask | chosen_bit
        branch_and_bound(new_candidates, current_size + 1, new_selected)

        # Branch 2: Exclude chosen_node
        new_candidates = candidates & ~chosen_bit
        branch_and_bound(new_candidates, current_size, selected_mask)

    branch_and_bound(all_mask, 0, 0)

    # Convert best_set bitmask to list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << i):
            result.append(i)

    return result if result else [1]
