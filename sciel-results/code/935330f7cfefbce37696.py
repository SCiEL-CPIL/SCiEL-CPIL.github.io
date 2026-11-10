"""Independent-set optimization for large-reciprocal-sum under lcm separation."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin under 90s wall-clock

    # Universe: [2, 256]
    N = 255
    vals = list(range(2, 257))  # vals[i] corresponds to element i (i in 0..254)

    # Incompatibility: pair (a,b) incompatible if lcm(a,b) <= 256
    # We need independent set in this graph (no edge among selected).
    # Build adjacency as bitsets (int) for speed.
    adj = [0] * N
    for i in range(N):
        a = vals[i]
        for j in range(i + 1, N):
            b = vals[j]
            if math.lcm(a, b) <= 256:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Reciprocal weights
    w = [1.0 / vals[i] for i in range(N)]

    def weight_of(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            s += w[i]
            m ^= lsb
        return s

    # Greedy builders: several heuristics to get diverse strong starts.
    def greedy_order(order):
        mask = 0
        for i in order:
            if (adj[i] & mask) == 0:
                mask |= (1 << i)
        return mask

    # Heuristic orders
    orders = []
    # smallest value first (max reciprocal)
    orders.append(sorted(range(N), key=lambda i: vals[i]))
    # order by increasing number of neighbors (less constrained first)
    orders.append(sorted(range(N), key=lambda i: bin(adj[i]).count('1')))
    # order by decreasing weight (smallest values first)
    orders.append(sorted(range(N), key=lambda i: -w[i]))
    # random shuffles
    rnd = random.Random(12345)
    for _ in range(10):
        o = list(range(N))
        rnd.shuffle(o)
        orders.append(o)

    starts = []
    for o in orders:
        m = greedy_order(o)
        if m and m not in starts:
            starts.append(m)

    # Local search: add, remove, swap moves.
    def neighbors_of(mask):
        # candidate additions: elements not in mask and compatible with all in mask
        cands = []
        for i in range(N):
            if (mask >> i) & 1:
                continue
            if (adj[i] & mask) == 0:
                cands.append(i)
        return cands

    def local_search(mask, t_limit):
        cur = mask
        curw = weight_of(cur)
        improved = True
        while improved and (time.time() < t_limit):
            improved = False
            # ADD moves
            for i in neighbors_of(cur):
                new = cur | (1 << i)
                nw = curw + w[i]
                if nw > curw + 1e-15:
                    cur = new
                    curw = nw
                    improved = True
                    break
            if improved:
                continue
            # SWAP moves: remove one element, add a compatible one
            for r in range(N):
                if (cur >> r) & 1:
                    base = cur ^ (1 << r)
                    for a in range(N):
                        if (base >> a) & 1:
                            continue
                        if (adj[a] & base) == 0:
                            new = base | (1 << a)
                            nw = curw - w[r] + w[a]
                            if nw > curw + 1e-15:
                                cur = new
                                curw = nw
                                improved = True
                                break
                    if improved:
                        break
        return cur, curw

    best_mask = max(starts, key=weight_of) if starts else 0
    best_w = weight_of(best_mask)

    # Run local search from each start, keeping best.
    for m in starts:
        if time.time() > DEADLINE:
            break
        lm, lw = local_search(m, DEADLINE)
        if lw > best_w:
            best_w = lw
            best_mask = lm

    # Branch-and-bound for maximum weight independent set
    # Sort vertices by weight descending for better pruning
    order = sorted(range(N), key=lambda i: -w[i])
    # suffix_w[k] = sum of weights of vertices from position k to end in order
    suffix_w = [0.0] * (N + 1)
    for k in range(N - 1, -1, -1):
        suffix_w[k] = suffix_w[k + 1] + w[order[k]]

    # Precompute for each position, the mask of vertices from that position onward
    # that are compatible with each vertex
    # Actually, let's just do a simpler BB with the ordering

    def branch_and_bound(t_limit):
        nonlocal best_mask, best_w
        # Stack-based DFS
        # State: (pos, mask, weight)
        # pos = current position in order
        # mask = selected vertices so far
        # weight = total weight so far
        stack = [(0, 0, 0.0)]
        nodes_visited = 0

        while stack:
            if time.time() > t_limit:
                break
            pos, mask, weight = stack.pop()
            nodes_visited += 1

            # Upper bound: current weight + sum of all remaining weights
            ub = weight + suffix_w[pos]
            if ub <= best_w + 1e-15:
                continue

            if pos == N:
                if weight > best_w:
                    best_w = weight
                    best_mask = mask
                continue

            v = order[pos]

            # Option 1: exclude v
            if pos + 1 < N:
                ub_excl = weight + suffix_w[pos + 1]
            else:
                ub_excl = weight
            if ub_excl > best_w + 1e-15:
                stack.append((pos + 1, mask, weight))

            # Option 2: include v (if compatible with all in mask)
            if (adj[v] & mask) == 0:
                new_mask = mask | (1 << v)
                new_weight = weight + w[v]
                if new_weight > best_w:
                    best_w = new_weight
                    best_mask = new_mask

                if pos + 1 < N:
                    ub_incl = new_weight + suffix_w[pos + 1]
                else:
                    ub_incl = new_weight
                if ub_incl > best_w + 1e-15:
                    stack.append((pos + 1, new_mask, new_weight))

        return nodes_visited

    # Run branch and bound
    if time.time() < DEADLINE:
        remaining = DEADLINE - time.time()
        if remaining > 5.0:
            nodes = branch_and_bound(time.time() + remaining - 2.0)

    # Final local search from the refined best
    if time.time() < DEADLINE:
        lm, lw = local_search(best_mask, DEADLINE)
        if lw > best_w:
            best_mask, best_w = lm, lw

    # Convert mask to value list
    result = [vals[i] for i in range(N) if (best_mask >> i) & 1]
    # Safety: ensure nonempty
    if not result:
        result = [2]
    return result
# EVOLVE-BLOCK-END
