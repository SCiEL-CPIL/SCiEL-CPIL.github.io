# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/205Z of maximum
    size that does NOT contain any two elements differing by a nonzero perfect square
    modulo 205.

    This implementation models the problem as a maximum independent set in the conflict
    graph (where edges connect pairs with forbidden differences) and solves it exactly
    using a branch-and-bound algorithm with bitset-based conflict pruning and
    degree-based vertex ordering.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205

    # Compute forbidden differences: nonzero quadratic residues modulo 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build conflict graph: two vertices are connected if their difference is in forbidden
    # We want maximum independent set, i.e., no two selected vertices are adjacent
    # adjacency[i] is a bitmask of vertices j such that i and j have a forbidden difference
    adjacency = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                adjacency[i] |= (1 << j)
                adjacency[j] |= (1 << i)

    # Precompute degree for each vertex
    degrees = [bin(adjacency[i]).count('1') for i in range(m)]

    # Branch and bound for maximum independent set
    # State: current set of available vertices (as a bitmask), current solution
    best_set = 0
    best_size = 0

    def popcount(x):
        return bin(x).count('1')

    def bb(avail, current_set, current_size):
        nonlocal best_set, best_size

        # Upper bound: current_size + popcount(avail)
        if current_size + popcount(avail) <= best_size:
            return

        if avail == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return

        # Greedy coloring bound for tighter pruning (optional, can be expensive)
        # For now, use simple bound

        # Select vertex with maximum degree in the induced subgraph on avail
        # Find vertex with max degree among available vertices
        max_deg = -1
        chosen = -1
        v = avail
        while v:
            # Extract lowest set bit
            lsb = v & (-v)
            idx = lsb.bit_length() - 1
            # Count neighbors of idx that are in avail
            deg = popcount(adjacency[idx] & avail)
            if deg > max_deg:
                max_deg = deg
                chosen = idx
            v ^= lsb

        if chosen == -1:
            return

        # Branch 1: exclude chosen
        new_avail_exclude = avail & ~(1 << chosen)
        bb(new_avail_exclude, current_set, current_size)

        # Branch 2: include chosen
        # If we include chosen, we must exclude all its neighbors
        new_avail_include = avail & ~(1 << chosen) & ~adjacency[chosen]
        new_current_set = current_set | (1 << chosen)
        bb(new_avail_include, new_current_set, current_size + 1)

    # Start with all vertices available
    all_avail = (1 << m) - 1
    bb(all_avail, 0, 0)

    # Extract the solution
    result = []
    for i in range(m):
        if best_set & (1 << i):
            result.append(i)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
