import math
import time
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] that share a factor with 255
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    N = len(candidates)

    # Find index of 255
    idx_255 = candidates.index(255)

    # Build adjacency using bitsets (Python ints as bitsets)
    # adj[i] is a bitset of neighbors of i
    adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # We need max clique containing idx_255
    # Equivalent to: max clique in subgraph induced by neighbors of idx_255, then +1
    # Neighbors of 255 (as bitset)
    neighbors_255 = adj[idx_255]

    # Get list of neighbor indices
    neighbor_indices = []
    for i in range(N):
        if i != idx_255 and (neighbors_255 >> i) & 1:
            neighbor_indices.append(i)

    M = len(neighbor_indices)

    # Remap: create a local adjacency for the neighbor subgraph
    # Map local index -> global index
    local_to_global = neighbor_indices
    # Build local adj bitsets
    local_adj = [0] * M
    for i in range(M):
        gi = local_to_global[i]
        for j in range(M):
            if i != j and (adj[gi] >> local_to_global[j]) & 1:
                local_adj[i] |= (1 << j)

    # Tomita's algorithm for maximum clique using coloring bound
    # We'll use a recursive branch-and-bound with greedy coloring

    best_size = 0
    best_clique = []

    def color_sort(R, P_bitset):
        """Greedy coloring to get an upper bound and ordering.
        Returns (color_bound, order) where order is list of vertices in P
        sorted by color (ascending), and color_bound is the number of colors used."""
        # Get vertices in P
        P_list = []
        for i in range(M):
            if (P_bitset >> i) & 1:
                P_list.append(i)

        if not P_list:
            return 0, []

        # Greedy sequential coloring
        colors = {}
        color_of = {}
        order = []
        available = set(P_list)

        c = 1
        while available:
            # Greedily assign color c to as many non-adjacent vertices as possible
            color_class = []
            for v in sorted(available, key=lambda x: -bin(local_adj[x] & P_bitset).count('1')):
                # Check if v is non-adjacent to all in color_class
                can_add = True
                for u in color_class:
                    if (local_adj[v] >> u) & 1:
                        can_add = False
                        break
                if can_add:
                    color_class.append(v)
                    available.discard(v)
                    color_of[v] = c

            if not color_class:
                # Shouldn't happen, but safety
                for v in list(available):
                    color_of[v] = c
                    color_class.append(v)
                    available.discard(v)

            order.extend(color_class)
            c += 1

        return c, order

    def expand(R_size, R_bitset, P_bitset, depth):
        nonlocal best_size, best_clique

        if time.time() > deadline:
            return

        if P_bitset == 0:
            if R_size > best_size:
                best_size = R_size
                # Reconstruct clique from R_bitset
                best_clique = [local_to_global[i] for i in range(M) if (R_bitset >> i) & 1]
            return

        # Color bound
        color_bound, order = color_sort(R_size, P_bitset)
        if R_size + color_bound <= best_size:
            return

        # Process vertices in reverse color order (highest color first)
        for v in reversed(order):
            if (P_bitset >> v) & 1:
                # Branch: include v
                new_R = R_bitset | (1 << v)
                new_P = P_bitset & local_adj[v]
                expand(R_size + 1, new_R, new_P, depth + 1)
                P_bitset &= ~(1 << v)

                # Update bound check
                # Recompute remaining bound
                remaining = P_bitset
                if remaining == 0:
                    break
                # Simple bound: count remaining
                rem_count = bin(remaining).count('1')
                if R_size + 1 + rem_count <= best_size:
                    break

    # Initialize
    initial_P = 0
    for i in range(M):
        initial_P |= (1 << i)

    expand(0, 0, initial_P, 0)

    # Build result
    if best_size > 0:
        result = [candidates[i] for i in best_clique]
        result.append(255)
    else:
        # Fallback: multiples of 3
        result = [x for x in range(1, 256) if x % 3 == 0]

    # Ensure 255 is included and no duplicates
    result = list(set(result))
    if 255 not in result:
        result.append(255)

    # Validate
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) <= 1:
                # Invalid, fall back to multiples of 3
                result = [x for x in range(1, 256) if x % 3 == 0]
                break

    return result
