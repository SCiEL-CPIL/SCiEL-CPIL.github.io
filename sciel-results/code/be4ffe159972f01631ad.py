"""MIS via Branch-and-Bound with greedy coloring bound and degree-based branching."""
import time
import sys

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    N = 256

    # Build conflict graph: edge (a,b) if a*b divisible by a+b
    # vertices 0..255 represent numbers 1..256
    # Use bitmasks (Python ints) for adjacency
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Degree of each vertex
    deg = [adj[i].bit_count() for i in range(N)]

    # Greedy coloring of a set (bitmask) to get upper bound on independent set size
    def greedy_color_bound(cand):
        """Returns (upper_bound, color_classes) using greedy coloring.
        cand is a bitmask of candidate vertices."""
        if cand == 0:
            return 0, []
        colors = []  # list of bitmasks, one per color class
        remaining = cand
        while remaining:
            color_class = 0
            rem = remaining
            while rem:
                # pick lowest set bit
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                # can we put v in this color? v must not be adjacent to any in color_class
                if (adj[v] & color_class) == 0:
                    color_class |= lsb
                    rem &= rem - 1
                else:
                    rem &= rem - 1
            colors.append(color_class)
            remaining &= ~color_class
        return len(colors), colors

    # Greedy independent set heuristic for lower bound
    def greedy_independent_set(cand):
        """Greedy: repeatedly pick vertex with min degree in cand, add it, remove neighbors."""
        result = 0
        remaining = cand
        while remaining:
            # pick vertex with minimum degree within remaining
            best_v = -1
            best_d = N + 1
            rem = remaining
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                d = (adj[v] & remaining).bit_count()
                if d < best_d:
                    best_d = d
                    best_v = v
                rem &= rem - 1
            result |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        return result

    full_mask = (1 << N) - 1
    # Initial lower bound via greedy
    best_set = greedy_independent_set(full_mask)
    best_size = best_set.bit_count()

    # Also try greedy picking max-degree first
    def greedy_independent_set_maxdeg(cand):
        result = 0
        remaining = cand
        while remaining:
            best_v = -1
            best_d = -1
            rem = remaining
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                d = (adj[v] & remaining).bit_count()
                if d > best_d:
                    best_d = d
                    best_v = v
                rem &= rem - 1
            result |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        return result

    alt = greedy_independent_set_maxdeg(full_mask)
    if alt.bit_count() > best_size:
        best_size = alt.bit_count()
        best_set = alt

    # Branch and Bound
    nodes_explored = 0

    def bb(cand, current_set, current_size):
        nonlocal best_set, best_size, nodes_explored

        if time.time() > DEADLINE:
            return

        if cand == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return

        # Upper bound via greedy coloring
        ub, _ = greedy_color_bound(cand)
        if current_size + ub <= best_size:
            return

        # Reduction: if a vertex has degree 0 in cand, include it
        # (it can't conflict with anything in cand)
        rem = cand
        forced = 0
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            if (adj[v] & cand) == 0:
                forced |= lsb
            rem &= rem - 1
        if forced:
            bb(cand & ~forced, current_set | forced, current_size + forced.bit_count())
            return

        # Branch on vertex with max degree in cand
        max_d = -1
        pivot = -1
        rem = cand
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            d = (adj[v] & cand).bit_count()
            if d > max_d:
                max_d = d
                pivot = v
            rem &= rem - 1

        pivot_bit = 1 << pivot

        # Branch 1: include pivot
        nodes_explored += 1
        new_cand = cand & ~(adj[pivot] | pivot_bit)
        bb(new_cand, current_set | pivot_bit, current_size + 1)

        # Branch 2: exclude pivot
        nodes_explored += 1
        bb(cand & ~pivot_bit, current_set, current_size)

    # Start B&B
    bb(full_mask, 0, 0)

    # Convert bitmask to list
    result = []
    m = best_set
    while m:
        lsb = m & (-m)
        result.append(lsb.bit_length())  # 1-indexed
        m &= m - 1

    if not result:
        result = [1]

    return result

# Verify and return
if __name__ == "__main__":
    s = construct()
    print(len(s))
