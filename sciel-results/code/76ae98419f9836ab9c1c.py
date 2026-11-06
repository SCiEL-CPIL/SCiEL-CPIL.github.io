"""Optimized solver for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from fractions import Fraction
    import sys

    N = 256
    # Nodes are numbers 2..256, indexed 0..254
    nums = list(range(2, 257))
    n = len(nums)  # 255

    # Build adjacency: edge (i,j) if lcm(nums[i], nums[j]) > 256
    # We want a clique in this graph (all pairs have lcm > 256)
    # Use bitsets for speed
    adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            if lcm(ai, bj) > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Weights: reciprocal of each number
    weights = [Fraction(1, nums[i]) for i in range(n)]

    # Maximum weight clique in a graph with 255 nodes
    # Branch and bound with bitset candidates and greedy coloring bound

    best_weight = Fraction(0, 1)
    best_clique = []

    # Greedy coloring for upper bound on weight
    # We use a simple greedy coloring: order nodes by weight descending,
    # assign smallest available color. The number of colors gives an upper bound
    # on the maximum weight clique (since each color class is an independent set,
    # and a clique can have at most one node per color).
    # But we need a weight-based bound: sum of top k weights where k = num_colors?
    # Actually, for maximum weight clique, we can use:
    # Order candidates by weight descending. Color greedily.
    # The upper bound for a subset of candidates is: for each color c, take the
    # max weight node of that color among the candidates. Sum those.
    # This is tighter than just counting colors.

    def greedy_color_bound(candidates_bitset, weights_list):
        """
        candidates_bitset: int bitmask of candidate nodes
        weights_list: list of Fractions indexed by node
        Returns upper bound on max weight clique within candidates.
        """
        # Extract candidate indices
        cands = []
        tmp = candidates_bitset
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            cands.append(idx)
            tmp ^= lsb

        if not cands:
            return Fraction(0, 1)

        # Sort by weight descending
        cands.sort(key=lambda x: weights_list[x], reverse=True)

        # Greedy coloring: assign each node the smallest color not used by its neighbors in cands
        # But for a bound, we just need a valid coloring. Let's do simple greedy.
        color_of = {}
        color_weights = {}  # color -> max weight assigned to that color
        for node in cands:
            # Find colors used by neighbors of node in cands
            neighbor_colors = set()
            for other in cands:
                if other == node:
                    continue
                # Check if edge exists
                if adj[node] & (1 << other):
                    if other in color_of:
                        neighbor_colors.add(color_of[other])
            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[node] = c
            if c not in color_weights or weights_list[node] > color_weights[c]:
                color_weights[c] = weights_list[node]

        bound = sum(color_weights.values())
        return bound

    # Branch and bound
    # We process nodes in order of decreasing weight for better pruning
    node_order = list(range(n))
    node_order.sort(key=lambda x: weights[x], reverse=True)

    # Map original index to position in order
    pos_in_order = [0] * n
    for pos, orig in enumerate(node_order):
        pos_in_order[orig] = pos

    # For branch and bound, we maintain a set of candidates as a bitset
    # We branch on including or excluding nodes in order

    # To make it efficient, we use a recursive approach with a stack
    # But 255 nodes is large. Let's use a more clever approach.

    # Actually, let's use a different strategy:
    # Since the graph is dense (most pairs have lcm > 256), the complement
    # graph (edges where lcm <= 256) is sparse. A clique in our graph is an
    # independent set in the complement.
    # So we want a maximum weight independent set in the complement graph.

    # Build complement adjacency
    comp_adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                comp_adj[i] |= (1 << j)
                comp_adj[j] |= (1 << i)

    # Maximum weight independent set in a sparse graph
    # Branch and bound: pick a node, branch on include or exclude
    # If we include node v, we must exclude all its neighbors in comp_adj
    # If we exclude node v, we just remove v from candidates

    # Use a priority-based B&B with fractional programming or simple weight bound

    # Let's implement a simple but effective B&B for MWIS
    # State: (candidates_bitset, current_weight, current_set)
    # Upper bound: sum of weights of all candidates (loose but simple)
    # Better: use the fact that in an independent set, no two adjacent nodes can be chosen.
    # A better bound: for each connected component, solve separately? Too complex.
    # Let's use: bound = sum of weights of candidates that are not adjacent to any other candidate
    # plus... this is getting complex.

    # Simpler approach: since the complement graph is sparse, the maximum independent
    # set is large. Let's just do a greedy construction and then local search.

    # Greedy: pick highest weight node, remove it and its neighbors, repeat
    def greedy_mwis():
        cands = (1 << n) - 1
        result = []
        while cands:
            # Find highest weight node in cands
            best_node = -1
            best_w = Fraction(0, 1)
            tmp = cands
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_node = idx
                tmp ^= lsb
            if best_node < 0:
                break
            result.append(best_node)
            # Remove best_node and its neighbors
            cands &= ~(1 << best_node)
            cands &= ~comp_adj[best_node]
        return result

    # Get initial solution
    init_clique = greedy_mwis()
    init_weight = sum(weights[i] for i in init_clique)
    best_weight = init_weight
    best_clique = init_clique[:]

    # Local search: try to swap nodes to improve
    # For each node in clique, try removing it and adding a non-neighbor from outside
    # This is a 1-swap local search

    in_clique = [False] * n
    for i in best_clique:
        in_clique[i] = True

    improved = True
    while improved:
        improved = False
        for i in list(best_clique):
            # Try removing i and adding some j not in clique that is compatible
            # j must not be adjacent to any other node in clique (in comp graph)
            # i.e., j must be independent with all other clique nodes
            for j in range(n):
                if in_clique[j]:
                    continue
                # Check if j is compatible with all other clique nodes
                compatible = True
                for k in best_clique:
                    if k == i:
                        continue
                    if comp_adj[j] & (1 << k):
                        compatible = False
                        break
                if not compatible:
                    continue
                # Try swap: remove i, add j
                new_weight = best_weight - weights[i] + weights[j]
                if new_weight > best_weight:
                    best_clique.remove(i)
                    best_clique.append(j)
                    in_clique[i] = False
                    in_clique[j] = True
                    best_weight = new_weight
                    improved = True
                    break

    # Convert back to original numbers
    result = [nums[i] for i in best_clique]
    return result
# EVOLVE-BLOCK-END
