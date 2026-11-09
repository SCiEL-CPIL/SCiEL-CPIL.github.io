"""Maximum weight independent set on the non-coprime conflict graph.

Nodes: integers 1..511, weight w(a) = 1/(512 - a).
Edges: pairs (a, b) with gcd(a, b) > 1.
Goal: maximize sum of weights of a pairwise-coprime subset.

Solved by branch-and-bound with a fractional-knapsack upper bound:
at a node with available set S, the bound is sum_{v in S} w(v)
(a valid upper bound for any independent subset of S).
"""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 512
    # vertices indexed 0..510 representing a = 1..511
    n = N - 1
    a_of = list(range(1, N))  # a_of[i] = i+1
    w = [1.0 / (N - (i + 1)) for i in range(n)]  # w[i] = 1/(512 - (i+1))

    # Precompute conflict bitsets (neighbors that share a factor > 1)
    # neighbors[i] is a Python int bitmask of indices j != i with gcd > 1
    neighbors = [0] * n
    for i in range(n):
        ai = i + 1
        mask = 0
        for j in range(n):
            if j == i:
                continue
            if math.gcd(ai, j + 1) > 1:
                mask |= (1 << j)
        neighbors[i] = mask

    all_mask = (1 << n) - 1

    # Greedy seed: pick in decreasing weight order (i.e. a from 511 down to 1),
    # keep if coprime to all chosen. This is the seed program's approach.
    chosen = []
    for i in range(n - 1, -1, -1):
        if all(math.gcd(i + 1, c) == 1 for c in chosen):
            chosen.append(i + 1)
    best_val = 0.0
    for c in chosen:
        best_val += 1.0 / (N - c)
    best_set = chosen[:]

    start = time.time()
    TIME_LIMIT = 110.0

    # Iterative DFS with explicit stack to avoid recursion-depth issues.
    # State: (available_mask, current_value, current_set_list)
    # We process include-branch first (higher weight) to raise incumbent fast.
    #
    # Bound at a node: current_value + sum(w[i] for i in available_mask)
    # We precompute nothing fancy; compute the available-weight sum incrementally.

    # To make the bound cheap, we maintain avail_weight alongside the mask.
    # Stack entries: (mask, value, avail_weight, set_list)
    stack = [(all_mask, 0.0, sum(w), [])]

    # Order of processing: we want to dive into include-branches of high-weight
    # vertices. We'll pop a node, then push exclude-branch and include-branch
    # (include pushed last so it's popped first).
    #
    # To pick which vertex to branch on, choose the highest-weight available.
    # Precompute for each mask? Too expensive. Instead, find max-weight available
    # by scanning; n=511 so scanning is fine, and we do it only at branch nodes.

    # Cache: for speed, precompute index->weight and a list sorted by weight desc.
    order_desc = sorted(range(n), key=lambda i: w[i], reverse=True)

    def pop_available_max(mask):
        # return index of highest-weight vertex in mask, or -1
        for i in order_desc:
            if mask & (1 << i):
                return i
        return -1

    count = 0
    while stack:
        mask, value, avail_weight, slist = stack.pop()
        # Upper bound check
        ub = value + avail_weight
        if ub <= best_val + 1e-15:
            continue
        if mask == 0:
            if value > best_val:
                best_val = value
                best_set = slist[:]
            continue
        # Time check (periodic)
        if count & 0x1FF == 0:
            if time.time() - start > TIME_LIMIT:
                break
        count += 1

        # Choose branch vertex: highest weight available
        v = pop_available_max(mask)
        if v < 0:
            continue

        wv = w[v]
        # Include branch: remove v and its neighbors
        new_mask = mask & ~(1 << v) & ~neighbors[v]
        new_avail = avail_weight - wv
        # subtract weights of neighbors that were in mask
        nb_in_mask = neighbors[v] & mask
        # sum weights of nb_in_mask
        # compute by scanning
        s = 0.0
        m = nb_in_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            s += w[idx]
            m ^= lsb
        new_avail -= s
        # Build new set list
        new_slist = slist + [v + 1]
        # Push include branch (process first)
        stack.append((new_mask, value + wv, new_avail, new_slist))

        # Exclude branch: remove v only
        excl_mask = mask & ~(1 << v)
        excl_avail = avail_weight - wv
        stack.append((excl_mask, value, excl_avail, slist))

    return best_set
# EVOLVE-BLOCK-END
