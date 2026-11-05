"""Rewritten program: Branch-and-bound with bitset conflicts for maximum weight independent set."""
import time
from math import lcm, gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    n_max = 256
    nodes = list(range(2, 257))
    N = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    weights = [1.0 / v for v in nodes]

    # Build conflict bitsets
    # conflict_mask[i] is a Python int with bit j set if nodes[i] and nodes[j] conflict
    conflict_mask = [0] * N
    for i in range(N):
        a = nodes[i]
        for j in range(i + 1, N):
            b = nodes[j]
            if lcm(a, b) <= n_max:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Sort nodes by weight descending for B&B ordering
    order = sorted(range(N), key=lambda i: -weights[i])
    # Re-index: new_order[k] = old index of k-th most valuable node
    # We'll work in the reordered space
    # Actually, let's just use the original indices but process in 'order'

    # Precompute for each node, the set of nodes that come after it in 'order' and don't conflict
    # For B&B: at step k, we consider order[k], and available nodes are those in order[k+1:] not conflicting with chosen

    # Branch and bound
    best_val = 0.0
    best_set = set()

    # Use iterative deepening or just B&B
    # State: (pos, available_mask, current_val, current_set)
    # available_mask: bitmask of nodes still available (in original index space)

    # To speed up, precompute suffix sums of weights in 'order'
    suffix_weight = [0.0] * (N + 1)
    for k in range(N - 1, -1, -1):
        suffix_weight[k] = suffix_weight[k + 1] + weights[order[k]]

    def branch_and_bound(pos, available_mask, current_val, current_set):
        nonlocal best_val, best_set
        if time.time() > DEADLINE:
            return
        if pos == N:
            if current_val > best_val:
                best_val = current_val
                best_set = set(current_set)
            return

        # Upper bound: current_val + sum of weights of available nodes from pos onward
        # Quick bound: current_val + suffix_weight[pos] (ignoring availability)
        if current_val + suffix_weight[pos] <= best_val + 1e-15:
            return

        # Try including order[pos] if available
        node = order[pos]
        if (available_mask >> node) & 1:
            # Include this node
            new_mask = available_mask & ~(1 << node) & ~conflict_mask[node]
            new_val = current_val + weights[node]
            new_set = current_set | {node}
            if new_val > best_val:
                best_val = new_val
                best_set = set(new_set)
            branch_and_bound(pos + 1, new_mask, new_val, new_set)

        # Try excluding this node
        branch_and_bound(pos + 1, available_mask, current_val, current_set)

    # Initial available mask: all nodes
    full_mask = (1 << N) - 1
    branch_and_bound(0, full_mask, 0.0, set())

    # Convert back to values
    result = sorted(nodes[i] for i in best_set)
    if not result:
        # Fallback: return a valid singleton
        result = [256]

    return result
