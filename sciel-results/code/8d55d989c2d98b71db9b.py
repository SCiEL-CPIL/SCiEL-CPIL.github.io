import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    random.seed(42)
    
    n = 256

    # Build conflict adjacency as Python int bitmasks
    # adj[i] is a bitmask of nodes that conflict with node i (i in 1..n)
    adj = [0] * (n + 1)

    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Precompute degrees
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = adj[i].bit_count()

    all_mask = (1 << (n + 1)) - 2  # bits 1..n set

    # Greedy independent set with minimum degree heuristic
    def greedy_independent_set(candidates, strategy='min'):
        selected = 0
        remaining = candidates
        while remaining:
            chosen = -1
            best_val = float('inf') if strategy == 'min' else -1
            t = remaining
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if strategy == 'min':
                    if deg < best_val:
                        best_val = deg
                        chosen = node
                else:
                    if deg > best_val:
                        best_val = deg
                        chosen = node
                t &= t - 1
            if chosen == -1:
                break
            selected |= (1 << chosen)
            remaining &= ~(adj[chosen] | (1 << chosen))
        return selected

    # Get initial lower bound
    best_set = greedy_independent_set(all_mask, 'min')
    best_size = best_set.bit_count()
    
    # Try max-degree removal
    alt = greedy_independent_set(all_mask, 'max')
    if alt.bit_count() > best_size:
        best_size = alt.bit_count()
        best_set = alt

    # Greedy coloring upper bound
    def color_bound(candidates):
        if candidates == 0:
            return 0
        uncolored = candidates
        color_count = 0
        while uncolored:
            color_count += 1
            color_class = 0
            t = uncolored
            nodes = []
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                nodes.append((degrees[node], node))
                t &= t - 1
            nodes.sort(reverse=True)
            for _, node in nodes:
                if (adj[node] & color_class) == 0:
                    color_class |= (1 << node)
            uncolored &= ~color_class
        return color_count

    # Branch and bound
    def branch_and_bound(candidates, current_size, selected_mask, depth=0):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = selected_mask
            return

        if current_size + candidates.bit_count() <= best_size:
            return

        ub = current_size + color_bound(candidates)
        if ub <= best_size:
            return

        # Pick branching vertex: highest degree
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
        branch_and_bound(new_candidates, current_size + 1, new_selected, depth + 1)

        # Branch 2: Exclude chosen_node
        new_candidates = candidates & ~chosen_bit
        branch_and_bound(new_candidates, current_size, selected_mask, depth + 1)

    branch_and_bound(all_mask, 0, 0)

    # Convert best_set bitmask to list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << i):
            result.append(i)

    return result if result else [1]
