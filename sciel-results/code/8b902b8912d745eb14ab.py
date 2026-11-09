import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256

    # Precompute conflict masks: conflict_mask[i] has bit j set iff (i*j)%(i+j)==0
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Full universe mask (bits 1..N set)
    all_mask = 0
    for i in range(1, N + 1):
        all_mask |= (1 << i)

    # Helper: iterate set bits
    def iter_bits(mask):
        while mask:
            lb = mask & (-mask)
            yield lb.bit_length() - 1
            mask ^= lb

    # Helper: popcount
    def popcount(m):
        return bin(m).count('1')

    # Greedy coloring upper bound on the independence number of the subgraph induced by `mask`
    # Uses a greedy sequential coloring: repeatedly pick a vertex of max degree, assign it a color,
    # remove it and its neighbors. The number of colors used is an upper bound on alpha.
    def color_bound(mask):
        if not mask:
            return 0
        remaining = mask
        colors = 0
        while remaining:
            # Find vertex with max degree in remaining
            max_deg = -1
            best_v = -1
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = popcount(conflict_mask[v] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lb
            if best_v == -1:
                break
            colors += 1
            # Remove best_v and its neighbors from remaining
            to_remove = (1 << best_v) | (conflict_mask[best_v] & remaining)
            remaining &= ~to_remove
        return colors

    # Greedy lower bound: min-degree greedy independent set
    def greedy_min_degree(initial_mask):
        remaining = initial_mask
        selected = 0
        while remaining:
            min_deg = float('inf')
            best_v = -1
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = popcount(conflict_mask[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lb
            if best_v == -1:
                break
            selected |= (1 << best_v)
            to_remove = (1 << best_v) | conflict_mask[best_v]
            remaining &= ~to_remove
        return selected

    # Branch and bound for maximum independent set
    # Uses recursion with coloring bound for pruning
    # Returns the best independent set mask found
    best_found = [0]
    best_size = [0]

    def branch_and_bound(remaining, selected, current_size):
        if time.time() > deadline - 1:
            return

        if not remaining:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_found[0] = selected
            return

        # Upper bound via coloring
        ub = color_bound(remaining)
        if current_size + ub <= best_size[0]:
            return

        # Pick a vertex to branch on: use max degree in remaining for better pruning
        max_deg = -1
        pivot = -1
        temp = remaining
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = popcount(conflict_mask[v] & remaining)
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= lb

        if pivot == -1:
            return

        # Branch 1: include pivot
        new_selected = selected | (1 << pivot)
        # Remove pivot and its neighbors from remaining
        new_remaining = remaining & ~( (1 << pivot) | conflict_mask[pivot] )
        branch_and_bound(new_remaining, new_selected, current_size + 1)

        # Branch 2: exclude pivot
        new_remaining2 = remaining & ~(1 << pivot)
        branch_and_bound(new_remaining2, selected, current_size)

    # Initialize with a good lower bound
    initial_lb = greedy_min_degree(all_mask)
    best_found[0] = initial_lb
    best_size[0] = popcount(initial_lb)

    # Run branch and bound
    branch_and_bound(all_mask, 0, 0)

    # If B&B didn't improve, try a few more greedy strategies
    if best_size[0] == popcount(initial_lb):
        # Try starting from odds
        odds_mask = 0
        for x in range(1, N + 1):
            if x % 2 == 1:
                odds_mask |= (1 << x)
        # Add compatible evens
        compatible_evens = 0
        temp = all_mask & ~odds_mask
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            if not (conflict_mask[v] & odds_mask):
                compatible_evens |= lb
            temp ^= lb

        # Greedy add compatible evens
        remaining = compatible_evens
        selected = odds_mask
        while remaining:
            min_deg = float('inf')
            best_v = -1
            temp = remaining
            while temp:
                lb2 = temp & (-temp)
                v = lb2.bit_length() - 1
                deg = popcount(conflict_mask[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lb2
            if best_v == -1:
                break
            selected |= (1 << best_v)
            to_remove = (1 << best_v) | conflict_mask[best_v]
            remaining &= ~to_remove

        sz = popcount(selected)
        if sz > best_size[0]:
            best_size[0] = sz
            best_found[0] = selected

    # Convert best mask to list
    result = []
    temp = best_found[0]
    while temp:
        lb = temp & (-temp)
        v = lb.bit_length() - 1
        result.append(v)
        temp ^= lb

    # Fallback: all odds (always valid since for odd a,b: a+b is even, ab is odd, so (a+b)|ab impossible)
    if not result:
        result = [x for x in range(1, N + 1) if x % 2 == 1]

    return result
