"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # We want a subset S of {2,...,256} such that for all distinct a,b in S, lcm(a,b) > 256.
    # Maximize sum(1/x for x in S).

    # Note: lcm(a,b) <= 256 iff a and b are "incompatible".
    # We can model this as a maximum weight independent set in a graph where
    # vertices are 2..256, and an edge exists between a and b if lcm(a,b) <= 256.
    # We want a maximum weight independent set.
    # Alternatively, in the complement graph (edges where lcm > 256), we want a max weight clique.

    # Let's use a branch and bound approach on the graph where edges mean "compatible" (lcm > 256).
    # Actually, it's easier to think in terms of the conflict graph: edge if lcm <= 256.
    # Then we want max weight independent set.

    # Precompute the conflict graph
    n = 255  # numbers 2 to 256
    nums = list(range(2, 257))
    idx = {x: i for i, x in enumerate(nums)}

    # conflict[i] is a bitset of vertices j such that lcm(nums[i], nums[j]) <= 256
    conflict = [0] * n
    for i in range(n):
        a = nums[i]
        for j in range(i+1, n):
            b = nums[j]
            if math_lcm(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # weights: 1/x
    weights = [1.0 / nums[i] for i in range(n)]

    # For branch and bound max weight independent set:
    # State: current set S (list of vertices), candidate set P (vertices that can be added to S)
    # Upper bound: sum of weights of vertices in P (since all weights are positive, the max additional
    #   weight we can get is at most the sum of all weights in P, but we can do better with a greedy bound)

    # A better upper bound: sort P by weight descending, and for each vertex, check if it conflicts with
    #   already selected. But a simple bound is: for each vertex in P, the max weight we can add from
    #   P given that vertex is included is weight[v] + sum of weights of vertices in P that are
    #   compatible with v and with each other... that's recursive.

    # Simpler: use the bound that the max weight independent set in P is at most the sum of weights
    #   of a maximum weight clique in the conflict subgraph induced by P... no, that's the same problem.

    # Let's use a greedy coloring bound on the conflict graph induced by P. In the conflict graph,
    #   an independent set can have at most one vertex from each color class in a proper coloring.
    #   So if we color the conflict subgraph induced by P, the max weight independent set weight is
    #   at most the sum over colors of the max weight vertex in that color class.

    # But coloring is expensive. Let's try a simpler approach first: since n=255, and the graph might
    #   be sparse or dense, let's check.

    # Actually, let's think about the structure. lcm(a,b) <= 256.
    # If a is large, say a > 128, then for any b >= 2, lcm(a,b) >= a > 128. But lcm(a,b) <= 256 requires
    #   that b divides some multiple of a that is <= 256. If a > 128, the only multiple of a that is <= 256
    #   is a itself. So lcm(a,b) = a * b / gcd(a,b) <= 256. If a > 128, then a*b/gcd(a,b) >= a > 128,
    #   and for it to be <= 256, we need b/gcd(a,b) <= 256/a < 2. So b/gcd(a,b) = 1, meaning b divides a.
    #   So for a > 128, b must divide a.

    # Similarly, for a in (64, 128], lcm(a,b) <= 256 requires b/gcd(a,b) <= 256/a <= 4.

    # This suggests the conflict graph might be relatively sparse for large numbers.

    # Let's try a greedy approach first to get a good initial solution, then use branch and bound.

    # Greedy: sort by weight descending (i.e., by value ascending), and add if compatible.
    # But this is what the seed does. Let's try sorting by value ascending (highest weight first).
    order = sorted(range(n), key=lambda i: nums[i])  # ascending value = descending weight
    greedy_set = []
    greedy_conflict_mask = 0
    for i in order:
        if (conflict[i] & greedy_conflict_mask) == 0:
            greedy_set.append(i)
            greedy_conflict_mask |= (1 << i)

    greedy_weight = sum(weights[i] for i in greedy_set)

    # Now branch and bound
    best_weight = greedy_weight
    best_set = set(greedy_set)

    # We'll do DFS. State: (candidates, current_weight, current_set)
    # candidates is a list of vertices that are compatible with all in current_set
    # and with each other? No, candidates just need to be compatible with current_set.
    # The independent set property among candidates is enforced by the search.

    # Actually, let's define:
    # candidates: list of vertices that are not in current_set and are compatible with all in current_set.
    # We want to find the max weight independent set within candidates (in the conflict graph).

    # Upper bound for a state: current_weight + sum of weights of candidates
    # This is a valid but loose bound.

    # Better: use a greedy coloring bound. Let's implement a simple one.

    def compute_bound(candidates, conflict_list, weights_list):
        # candidates: list of vertex indices
        # Returns an upper bound on the max weight independent set in the subgraph induced by candidates
        # Use greedy coloring: repeatedly pick the vertex with max weight, assign it a color,
        #   remove it and its neighbors. The bound is the sum of max weights per color.
        if not candidates:
            return 0.0

        # Build local index
        local_conflict = {}
        for i, v in enumerate(candidates):
            local_conflict[i] = 0
        for i, v in enumerate(candidates):
            for j, w in enumerate(candidates):
                if i != j and (conflict_list[v] >> w) & 1:
                    local_conflict[i] |= (1 << j)

        # Greedy coloring
        remaining = set(range(len(candidates)))
        color_max_weights = []
        while remaining:
            # Pick the vertex with max weight
            v = max(remaining, key=lambda i: weights_list[candidates[i]])
            color_max_weights.append(weights_list[candidates[v]])
            # Remove v and its neighbors
            neighbors = local_conflict[v]
            to_remove = {v}
            for i in remaining:
                if (neighbors >> i) & 1:
                    to_remove.add(i)
            remaining -= to_remove

        return sum(color_max_weights)

    # This bound might be expensive. Let's try a simpler one first: just sum of weights.
    # And see if we can prune.

    # Let's try a different approach: since the problem size is 255, and the conflict graph
    #   might have a special structure, let's try to exploit it.

    # Observation: if a and b are both > 128, then lcm(a,b) <= 256 implies one divides the other.
    #   So among numbers > 128, the conflict graph is a comparability graph (edges if one divides the other).
    #   The independent sets in a comparability graph are... well, it's still NP-hard in general.

    # Let's just do branch and bound with the simple sum bound and see how it goes.
    # We'll sort candidates by weight descending to find good solutions early.

    def branch_and_bound(candidates, current_weight, current_set):
        nonlocal best_weight, best_set
        if time.time() > deadline:
            return

        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return

        # Upper bound
        bound = current_weight + sum(weights[i] for i in candidates)
        if bound <= best_weight:
            return

        # Pick a vertex to branch on. Choose the one with highest weight.
        # Actually, let's choose the vertex that maximizes the bound reduction.
        # Simple heuristic: pick the vertex with highest weight.
        v = max(candidates, key=lambda i: weights[i])

        # Branch 1: include v
        new_set = current_set + [v]
        new_candidates = [i for i in candidates if i != v and not ((conflict[i] >> v) & 1)]
        branch_and_bound(new_candidates, current_weight + weights[v], new_set)

        # Branch 2: exclude v
        new_candidates = [i for i in candidates if i != v]
        branch_and_bound(new_candidates, current_weight, current_set)

    # Initial candidates: all vertices
    initial_candidates = list(range(n))
    # But we can reduce: the greedy set already gives a good solution.
    # Let's start the search with the greedy set as the current set, and candidates as the rest
    #   that are compatible with the greedy set.
    greedy_mask = 0
    for i in greedy_set:
        greedy_mask |= (1 << i)

    initial_candidates = [i for i in range(n) if not (greedy_mask >> i) & 1 and (conflict[i] & greedy_mask) == 0]
    branch_and_bound(initial_candidates, greedy_weight, list(greedy_set))

    # Convert best_set to the actual numbers
    result = sorted(nums[i] for i in best_set)
    return result
# EVOLVE-BLOCK-END
