"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    # Precompute the conflict graph: edge if lcm(a,b) <= 256
    n = 255  # numbers 2 to 256
    nums = list(range(2, 257))

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

    # Sort vertices by weight descending (i.e., by value ascending)
    order = sorted(range(n), key=lambda i: nums[i])
    
    # Greedy initial solution
    greedy_set = []
    greedy_mask = 0
    for i in order:
        if (conflict[i] & greedy_mask) == 0:
            greedy_set.append(i)
            greedy_mask |= (1 << i)

    greedy_weight = sum(weights[i] for i in greedy_set)

    best_weight = greedy_weight
    best_set = list(greedy_set)

    # Tight upper bound using greedy coloring on the conflict subgraph
    def color_bound(candidates):
        """Compute upper bound on max weight independent set in conflict subgraph induced by candidates.
        Uses greedy coloring: repeatedly pick highest-weight vertex, assign color, remove it and neighbors.
        Bound = sum of max weight per color class."""
        if not candidates:
            return 0.0
        
        # Build local adjacency among candidates
        local_adj = {}
        cand_list = list(candidates)
        for i, v in enumerate(cand_list):
            local_adj[i] = 0
        for i in range(len(cand_list)):
            vi = cand_list[i]
            for j in range(i+1, len(cand_list)):
                vj = cand_list[j]
                if (conflict[vi] >> vj) & 1:
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)
        
        # Greedy coloring by weight
        remaining = set(range(len(cand_list)))
        color_weights = []
        while remaining:
            # Pick vertex with max weight
            v = max(remaining, key=lambda i: weights[cand_list[i]])
            color_weights.append(weights[cand_list[v]])
            # Remove v and its neighbors
            to_remove = {v}
            adj = local_adj[v]
            for i in remaining:
                if (adj >> i) & 1:
                    to_remove.add(i)
            remaining -= to_remove
        
        return sum(color_weights)

    # Branch and bound with tight coloring bound
    def branch_and_bound(candidates, current_weight, current_set):
        nonlocal best_weight, best_set
        if time.time() > deadline:
            return

        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(current_set)
            return

        # Tight upper bound
        bound = current_weight + color_bound(candidates)
        if bound <= best_weight:
            return

        # Pick branching vertex: highest weight
        v = max(candidates, key=lambda i: weights[i])

        # Branch 1: include v
        new_set = current_set + [v]
        new_candidates = [i for i in candidates if i != v and not ((conflict[i] >> v) & 1)]
        branch_and_bound(new_candidates, current_weight + weights[v], new_set)

        # Branch 2: exclude v
        new_candidates = [i for i in candidates if i != v]
        branch_and_bound(new_candidates, current_weight, current_set)

    # Start with greedy set, candidates are remaining compatible vertices
    initial_candidates = [i for i in range(n) if not (greedy_mask >> i) & 1 and (conflict[i] & greedy_mask) == 0]
    branch_and_bound(initial_candidates, greedy_weight, list(greedy_set))

    # Convert best_set to the actual numbers
    result = sorted(nums[i] for i in best_set)
    return result
# EVOLVE-BLOCK-END
