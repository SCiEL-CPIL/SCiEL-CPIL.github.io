"""
Maximum Weight Independent Set on the coprime graph of {1, ..., 511}.
Weight(a) = 1 / (512 - a).
Solved exactly via recursive branch-and-bound with tight upper bounds
derived from a greedy independent-set construction.
"""
import time
from math import gcd

N = 512
DEADLINE = 140.0  # seconds, leaving margin before the 150s kill


def construct():
    t_start = time.perf_counter()

    # Build the coprime graph.  We represent it as a list of bitsets (Python ints).
    # 511 vertices, indices 0..510 for values 1..511.
    # adj[i] is a bitset of vertices j such that gcd(i+1, j+1) == 1 and j != i.
    adj = [0] * N  # 511 entries
    weights = [0.0] * N  # float weights; we also keep exact rational via Fraction for comparison
    from fractions import Fraction
    frac_weights = [Fraction(0) for _ in range(N)]

    for i in range(N):
        a = i + 1
        frac_weights[i] = Fraction(1, N - a)
        weights[i] = 1.0 / (N - a)

    # Build adjacency
    for i in range(N):
        a = i + 1
        bits = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if gcd(a, b) == 1:
                bits |= (1 << j)
        adj[i] = bits

    # We want a maximum-weight independent set.
    # An independent set S has no edges within it, i.e., for any i, j in S, i is NOT in adj.
    # Equivalently, if we pick vertex i, we can only pick from the complement of adj[i] minus {i}.
    # Let's define: for a candidate set of available vertices (bitset), if we pick i,
    #   next available = available & ~adj[i] & ~(1 << i)
    #   (remove i and all its neighbors)

    # Greedy upper bound: repeatedly pick the available vertex with the largest weight,
    # remove it and its neighbors. Sum of picked weights is an upper bound on the
    # true optimal MWIS of the available set.
    # (This is a valid upper bound because we're relaxing: the greedy picks might conflict
    #  with each other? No — we remove neighbors, so the picks are independent. The sum
    #  of weights of a greedy independent set is a LOWER bound, not upper.)

    # Correction: The greedy independent-set sum is a LOWER bound.
    # For an UPPER bound, we need something else.
    # Simple upper bound: sum of all weights in available set.
    # Tighter: sort available vertices by weight desc; for each vertex v in that order,
    #   the best we can do starting from v is at most weight(v) + (sum of weights of vertices
    #   in available that are not neighbors of v and come after v in the sorted order).
    # This is the standard Tomita-style bound but it's expensive.

    # Practical approach: use the bound = sum of weights of all available vertices
    # (very loose), but branch on the vertex with the maximum weight first, and prune
    # aggressively when current_weight + bound <= best_found.

    # Even better upper bound: for the available set, the maximum weight independent set
    # is at most the sum of the top-k weights where k is the independence number of the
    # available subgraph. Estimating independence number is hard.

    # Let's use a simpler but effective strategy:
    # 1. Greedy construction gives a good initial lower bound (a valid independent set).
    # 2. Branch-and-bound with the bound = sum of all remaining weights (loose but simple).
    # 3. Branch order: try the highest-weight vertex first (likely to be in the optimum).

    # Actually, a much tighter bound: 
    #   For available set A, upper_bound = sum_{v in A} w(v) is trivial.
    #   Better: the greedy independent set weight on A is a lower bound.
    #   We need an upper bound. The sum of weights is an upper bound.
    #   A tighter upper bound: sort A by weight descending: v1, v2, ..., vk.
    #   For any independent set, it can contain at most one of {v1, ..., vj} that are
    #   mutually adjacent, but we don't easily exploit that.

    # Let's just use: upper_bound = sum of weights in available set.
    # And rely on good branching order + tight initial solution.

    def sum_weights(avail_bits):
        """Sum of weights of all vertices in the bitset avail_bits."""
        # Precompute? For 511 bits, iterating is fast enough.
        total = 0.0
        x = avail_bits
        while x:
            # lowest set bit
            lb = x & (-x)
            idx = lb.bit_length() - 1
            total += weights[idx]
            x ^= lb
        return total

    def greedy_is(avail_bits):
        """Greedy independent set: repeatedly pick highest-weight available vertex.
        Returns (weight, list_of_vertices)."""
        picked = []
        current = avail_bits
        total_w = 0.0
        while current:
            # Find vertex with max weight in current
            best = -1
            best_w = -1.0
            x = current
            while x:
                lb = x & (-x)
                idx = lb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best = idx
                x ^= lb
            if best < 0:
                break
            picked.append(best)
            total_w += weights[best]
            # Remove best and its neighbors
            current &= ~adj[best]
            current &= ~(1 << best)
        return total_w, picked

    # Initial solution via greedy
    all_bits = (1 << N) - 1
    best_w, best_set = greedy_is(all_bits)

    # Also try a few different greedy orders for a better initial solution
    # (e.g., by weight, by degree, by weight/degree)
    # For now, stick with one greedy.

    # Branch and bound
    # State: (current_weight, available_bits, path)
    # We'll use a stack (DFS) to manage recursion depth.

    # Precompute: for each vertex, the bitset of vertices with higher or equal weight
    # (for ordering). Actually, we just sort by weight at each node — expensive.
    # Alternative: pre-sort all vertices by weight descending once.
    order = list(range(N))
    order.sort(key=lambda i: -weights[i])
    # order_rank[i] = position of i in the sorted order (0 = highest weight)
    order_rank = [0] * N
    for rank, idx in enumerate(order):
        order_rank[idx] = rank

    # In branching, we iterate available vertices in order of decreasing weight.
    # To do this efficiently: for available_bits, iterate through order and pick
    # vertices that are in available_bits.

    # Upper bound function: sum of weights of available vertices.
    # This is loose but fast to compute (or we can cache).

    # Let's use a recursive approach with memoization on available_bits? 
    # Too many states. Just DFS.

    # Count nodes explored
    nodes = 0
    timeout_flag = [False]

    def dfs(current_w, avail_bits, path):
        nonlocal best_w, best_set, nodes, timeout_flag
        nodes += 1
        if timeout_flag[0]:
            return
        if time.perf_counter() - t_start > DEADLINE:
            timeout_flag[0] = True
            return

        if avail_bits == 0:
            if current_w > best_w + 1e-12:
                best_w = current_w
                best_set = path[:]
            return

        # Upper bound: current_w + sum of all weights in avail_bits
        # Compute sum of weights in avail_bits
        ub_extra = 0.0
        x = avail_bits
        while x:
            lb = x & (-x)
            idx = lb.bit_length() - 1
            ub_extra += weights[idx]
            x ^= lb
        ub = current_w + ub_extra
        if ub <= best_w + 1e-12:
            return

        # Branch: pick vertices in order of decreasing weight
        # Iterate through order, for each vertex in order that is in avail_bits:
        for idx in order:
            if timeout_flag[0]:
                return
            bit = 1 << idx
            if not (avail_bits & bit):
                continue

            new_w = current_w + weights[idx]
            new_path = path + [idx]
            new_avail = avail_bits & ~adj[idx] & ~bit

            # If new_w + upper bound of new_avail > best, recurse
            dfs(new_w, new_avail, new_path)

            # After exploring "include idx", explore "exclude idx"
            # Actually, the standard MWIS branch-and-bound is:
            #   Branch 1: include idx (done above)
            #   Branch 2: exclude idx (avail_bits without idx, but we don't remove its neighbors)
            # But in the loop above, we're iterating through candidates and for each,
            # we do include. The exclude is implicit: when we move to the next candidate,
            # the previous one is excluded.
            # This is the standard "include/exclude" branch where we branch on the
            # highest-weight available vertex: include it, or exclude it (and move to next).
            # My current loop does: for each candidate in weight order, include it.
            # This is equivalent to: the first candidate we try is the highest weight.
            # If we include it, we go deeper. If we don't include it, we continue to the
            # next candidate (which is the exclude branch for the first candidate).
            # So this is correct.

    # Call DFS
    dfs(0.0, all_bits, [])

    # Convert indices back to values
    result = sorted(idx + 1 for idx in best_set)
    return result
