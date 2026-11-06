"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin

    N = 256
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)

    # Numbers where 2a is a perfect square are forbidden (diagonal constraint)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers

    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:
                    adj[i] |= (1 << j)

    # Greedy coloring for upper bound (Tomita-style)
    # Returns list of color classes (each is a bitmask)
    def color_sort(P):
        """Color vertices in P and return (order, color_classes)."""
        order = []
        color_classes = []
        colored = 0
        remaining = P
        while remaining:
            # Greedily pick a maximal independent set from remaining
            color_class = 0
            new_remaining = remaining
            # Try to add vertices to current color class
            # Use a simple greedy: iterate through remaining, add if no conflict with color_class
            temp_remaining = remaining
            while temp_remaining:
                # Pick lowest set bit
                lb = temp_remaining & (-temp_remaining)
                v = lb.bit_length() - 1
                if (adj[v] & color_class) == 0:
                    color_class |= lb
                    new_remaining &= ~lb
                temp_remaining &= ~lb
            if color_class == 0:
                break
            color_classes.append(color_class)
            remaining = new_remaining
            # Build order: vertices in color_class in order of addition
            # Actually for Tomita we need the order of coloring
            # Let's use a simpler approach: just track the count
        # For the upper bound, we just need the number of colors
        return len(color_classes)

    # Better: implement Tomita's branch and bound properly
    # We'll use a recursive approach with bitsets

    # Precompute: for each vertex, its neighbors in the graph
    # adj[i] is already a bitmask of neighbors

    # Tomita's algorithm:
    # State: P (candidates to add), X (excluded vertices), current solution
    # Upper bound: |current| + coloring(P)
    # Branch: pick a vertex from P, add to solution, recurse with P' = P & ~adj[v] & ~v

    # For efficiency, use a greedy coloring that gives a good upper bound
    def greedy_color_upper_bound(P):
        """Return upper bound on independent set size in P using greedy coloring."""
        if P == 0:
            return 0
        count = 0
        remaining = P
        while remaining:
            # Find a maximal independent set in remaining
            color_class = 0
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                if (adj[v] & color_class) == 0:
                    color_class |= lb
                temp &= ~lb
            count += 1
            remaining &= ~color_class
        return count

    # Branch and bound
    best_size = [0]
    best_mask = [0]

    def branch_and_bound(P, X, current_mask, current_size):
        if time.time() > deadline:
            return
        if P == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_mask[0] = current_mask
            return

        # Upper bound
        ub = current_size + greedy_color_upper_bound(P)
        if ub <= best_size[0]:
            return

        # Pick a vertex from P to branch on
        # Heuristic: pick vertex with highest degree in P
        # For simplicity, pick the first one, or use a better heuristic
        # Let's pick the vertex that appears in the last color class (Tomita's strategy)
        # Actually, let's just iterate through P and try each vertex

        # For better performance, pick vertex with max degree in P
        # But for simplicity and speed, let's use a fixed ordering
        # Actually, Tomita picks from the last color class. Let me implement that.

        # Simpler: just iterate through all vertices in P
        temp = P
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            # Add v to solution
            new_mask = current_mask | lb
            new_size = current_size + 1
            # New P: vertices in P that are not neighbors of v and not v
            new_P = P & ~adj[v] & ~lb
            # New X: X | {v} | (neighbors of v in P)
            new_X = X | lb | (adj[v] & P)
            # Actually X doesn't matter much for the search, P is what matters
            branch_and_bound(new_P, new_X, new_mask, new_size)
            temp &= ~lb
            if time.time() > deadline:
                return

    # Initial P: all valid vertices
    all_mask = (1 << M) - 1
    branch_and_bound(all_mask, 0, 0, 0)

    if best_size[0] > 0:
        result = [idx_to_num[i] for i in range(M) if (best_mask[0] >> i) & 1]
    else:
        # Fallback: greedy
        result = []
        for x in valid_numbers:
            ok = True
            for y in result:
                if x + y in squares:
                    ok = False
                    break
            if ok:
                result.append(x)

    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]

    return sorted(result)
# EVOLVE-BLOCK-END
