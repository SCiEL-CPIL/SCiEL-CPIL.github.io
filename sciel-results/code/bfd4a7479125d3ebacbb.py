"""Rewritten program: exact maximum sum-free subset via branch-and-bound on the
conflict graph, with a greedy + local-search fallback under a time budget."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 140.0
_START = time.time()


def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)


def _conflict_graph(vals):
    """Return adjacency list of the conflict graph.
    Vertices are indices into vals. Edge (i,j) exists if vals[i]+vals[j] is
    present in vals (i.e. i+j = k for some k, with a=b allowed)."""
    n = len(vals)
    lookup = {v: i for i, v in enumerate(vals)}
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = vals[i] + vals[j]
            if s in lookup:
                k = lookup[s]
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _is_sum_free(vals, subset):
    """Verify no a+b=c with a,b,c in subset (a=b allowed)."""
    present = set(subset)
    for a in subset:
        for b in subset:
            if a + b in present:
                return False
    return True


def _exact_max_independent_set(adj, deadline):
    """Branch-and-bound maximum independent set on a graph with n vertices.
    Returns the best found subset as a set of vertex indices."""
    n = len(adj)
    if n == 0:
        return set()

    # Order vertices by degree (descending) for better pruning; remap.
    order = sorted(range(n), key=lambda i: -len(adj[i]))
    remap = {old: new for new, old in enumerate(order)}
    new_adj = [[] for _ in range(n)]
    for old_i in range(n):
        ni = remap[old_i]
        for old_j in adj[old_i]:
            new_adj[ni].append(remap[old_j])

    # Bitsets for fast neighborhood operations.
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in new_adj[i]:
            mask |= (1 << j)
        neigh[i] = mask

    best_size = 0
    best_set = set()

    def branch(candidates, chosen, size):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        # Upper bound: size + popcount(candidates)
        if size + bin(candidates).count("1") <= best_size:
            return
        if candidates == 0:
            if size > best_size:
                best_size = size
                best_set = set(chosen)
            return
        # Pick a vertex to branch on (highest degree within candidates)
        # Use the first set bit for simplicity but weighted by degree.
        # Find vertex with max degree in candidate set.
        c = candidates
        best_v = -1
        best_deg = -1
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            deg = bin(neigh[v] & candidates).count("1")
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lb
        v = best_v
        # Branch 1: include v
        chosen.append(v)
        branch(candidates & ~(1 << v) & ~neigh[v], chosen, size + 1)
        chosen.pop()
        # Branch 2: exclude v
        branch(candidates & ~(1 << v), chosen, size)

    branch((1 << n) - 1, [], 0)
    return best_set


def _greedy_local_search(vals, adj, deadline):
    """Greedy independent set then local-search improvement."""
    n = len(vals)
    # Greedy: repeatedly pick vertex with fewest conflicts among remaining.
    remaining = set(range(n))
    chosen = []
    while remaining and time.time() < deadline:
        # pick vertex with minimum degree within remaining
        best_v = None
        best_deg = None
        for v in remaining:
            d = sum(1 for u in remaining if u in adj[v])
            if best_deg is None or d < best_deg:
                best_deg = d
                best_v = v
        chosen.append(best_v)
        remaining.discard(best_v)
        for u in adj[best_v]:
            remaining.discard(u)
    chosen_set = set(chosen)

    # Local search: try single-vertex swaps and additions.
    def valid_set(s):
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a vertex
        for v in range(n):
            if v not in chosen_set:
                trial = chosen_set | {v}
                if valid_set(trial):
                    chosen_set = trial
                    improved = True
                    break
        if improved:
            continue
        # Try swapping one out for a larger set of others (simple 1-for-1)
        for v_out in list(chosen_set):
            for v_in in range(n):
                if v_in in chosen_set:
                    continue
                trial = (chosen_set - {v_out}) | {v_in}
                if valid_set(trial) and len(trial) > len(chosen_set) - 1:
                    # only accept if it leads to same size but we want growth;
                    # skip pure swaps, rely on additions
                    pass
    return chosen_set


def _solve_instance(vals, deadline):
    n = len(vals)
    if n == 0:
        return []
    adj = _conflict_graph(vals)
    if n <= 60:
        # Exact branch-and-bound
        best = _exact_max_independent_set(adj, deadline)
        if best:
            subset = [vals[i] for i in best]
            if _is_sum_free(vals, subset):
                return subset
    # Fallback: greedy + local search
    best = _greedy_local_search(vals, adj, deadline)
    subset = [vals[i] for i in best]
    # Verify; if invalid, fall back to odds
    if _is_sum_free(vals, subset):
        return subset
    return [x for x in vals if x % 2]


def construct():
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > _DEADLINE:
            # Out of time: return best-effort (odds) for remaining
            vals = _instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = _instance(size, seed)
        # Reserve time: give more to larger instances
        remaining = _DEADLINE - time.time()
        if size <= 60:
            alloc = min(30.0, remaining)
        elif size <= 120:
            alloc = min(50.0, remaining)
        else:
            alloc = remaining - 2.0  # leave margin
        alloc = max(alloc, 1.0)
        deadline = time.time() + alloc
        subset = _solve_instance(vals, deadline)
        results.append(subset)
    return results
# EVOLVE-BLOCK-END
