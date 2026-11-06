"""Exact Branch and Bound with bitsets for weighted independent set."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)

    # Build conflict bitsets
    conflict = [0] * n
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            if lcm(a, b) <= N:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    weights = [1.0 / numbers[i] for i in range(n)]

    # Greedy initial solution
    def greedy_solve():
        order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        selected = []
        sel_bits = 0
        for i in order:
            if conflict[i] & sel_bits:
                continue
            selected.append(i)
            sel_bits |= (1 << i)
        return selected

    best_solution = greedy_solve()
    best_score = sum(weights[i] for i in best_solution)

    # B&B with bitsets
    # available: bitmask of nodes that can still be added
    # We process in a fixed order to make it deterministic and prunable

    # Precompute: for each node, its conflict mask (already have as conflict[])
    # For B&B, we need to track available nodes as a bitmask.

    # To compute upper bound quickly, we need sum of weights of available nodes.
    # We can precompute prefix sums if we process in a fixed order, but available
    # changes arbitrarily. Instead, compute on the fly - with 255 nodes, summing
    # weights of set bits is O(n) which is acceptable if B&B prunes well.

    # Better: precompute weight array and use bit iteration.

    def sum_available_weights(available_mask):
        total = 0.0
        m = available_mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            total += weights[i]
            m ^= lsb
        return total

    # For faster bound computation, let's precompute a lookup for small masks? No, too large.
    # Just compute directly.

    # B&B: we'll use a stack to avoid recursion depth issues.
    # Each state: (available_mask, current_score, current_selection_mask)
    # We process states in a DFS manner.

    # To improve pruning, process nodes in order of decreasing weight.
    # At each step, pick the highest-weight available node to branch on.

    # For efficiency, precompute for each node its position in weight order.
    weight_order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    weight_rank = [0] * n
    for rank, node in enumerate(weight_order):
        weight_rank[node] = rank

    # In B&B, when picking which node to branch on, pick the one with smallest weight_rank (highest weight)
    # from the available set.

    def pick_best_node(available_mask):
        """Pick the available node with highest weight (smallest weight_rank)."""
        best_node = -1
        best_rank = n + 1
        m = available_mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            if weight_rank[i] < best_rank:
                best_rank = weight_rank[i]
                best_node = i
            m ^= lsb
        return best_node

    # B&B with explicit stack
    # State: (available_mask, current_score, selection_mask)
    # We'll use a list as stack, processing LIFO (DFS).

    # To avoid memory blowup, we limit stack size or use time-based pruning.

    all_mask = (1 << n) - 1

    # Start with all nodes available
    stack = [(all_mask, 0.0, 0)]

    while stack:
        if not (time.time() - start_time) < DEADLINE:
            break

        available_mask, current_score, sel_mask = stack.pop()

        if not available_mask:
            if current_score > best_score:
                best_score = current_score
                best_solution = []
                m = sel_mask
                while m:
                    lsb = m & (-m)
                    i = lsb.bit_length() - 1
                    best_solution.append(i)
                    m ^= lsb
            continue

        # Upper bound
        ub = current_score + sum_available_weights(available_mask)
        if ub <= best_score + 1e-15:
            continue

        # Pick node to branch on
        v = pick_best_node(available_mask)
        if v < 0:
            continue

        # Branch 1: include v
        # New available: remove v and all its neighbors
        new_avail_include = available_mask & ~(1 << v) & ~conflict[v]
        stack.append((new_avail_include, current_score + weights[v], sel_mask | (1 << v)))

        # Branch 2: exclude v
        new_avail_exclude = available_mask & ~(1 << v)
        stack.append((new_avail_exclude, current_score, sel_mask))

    return [numbers[i] for i in best_solution]

# EVOLVE-BLOCK-END
