"""Maximum-weight 3-AP-free subset of [1,256] via branch-and-bound."""
import time
from fractions import Fraction
import numpy as np

_N = 256

def _build_graph():
    """Build conflict bitsets: conflict[i] is a bitset of j such that
    {i,j} is a 'dangerous pair' (i.e. there exists k in [1,N] with
    2j - i = k, i.e. i and j are two members of some 3-AP)."""
    conflict = [0] * (_N + 1)  # 1-indexed, index 0 unused
    for i in range(1, _N + 1):
        # j > i such that 2j - i <= N  =>  j <= (N+i)//2, and 2j-i >= i => j>=i
        # Also need 2j-i != i (nonconstant) => j != i
        # Also 2j-i >= 1 always
        # For each j > i with 2j - i <= N: conflict between i and j
        #   because k = 2j - i is in [1,N] and i < j < k (since j>i => 2j-i > i)
        # Also j < i with 2j - i >= 1: i.e. 2j >= i+1 => j >= ceil((i+1)/2)
        #   then k = 2j - i < j, and j is the middle: k < j < i
        #   so i and j conflict.
        # Simpler: i and j conflict iff |2j - i| in [1,N] and 2j-i != i and 2j-i != j.
        # 2j-i != i iff j != i. 2j-i != j iff j != i. So for i != j:
        #   conflict iff 1 <= 2j - i <= N.
        for j in range(1, _N + 1):
            if j == i:
                continue
            k = 2 * j - i
            if 1 <= k <= _N:
                conflict[i] |= (1 << j)
    return conflict

def _greedy_initial(conflict, weights):
    """Greedy: repeatedly pick the vertex with max weight/(1+degree in remaining)
    or just max weight that is safe. Try a few orderings."""
    # Weights are 1/i for i in 1..N. Smaller i have larger weight.
    # Greedy by descending weight (i.e., ascending i) with conflict check.
    best_set = set()
    best_w = 0.0
    # Try greedy ascending i (smallest first, largest weight)
    chosen = set()
    total = 0.0
    for i in range(1, _N + 1):
        # check if i conflicts with any chosen
        ok = True
        for c in chosen:
            if (conflict[i] >> c) & 1:
                ok = False
                break
        if ok:
            chosen.add(i)
            total += 1.0 / i
    if total > best_w:
        best_w = total
        best_set = set(chosen)

    # Try greedy by weight/degree ratio
    remaining = list(range(1, _N + 1))
    chosen = set()
    total = 0.0
    # Precompute degrees within remaining dynamically is expensive; approximate.
    # Use a simple ratio: weight / (1 + popcount(conflict[i] & mask of remaining))
    remaining_mask = 0
    for i in range(1, _N + 1):
        remaining_mask |= (1 << i)
    while remaining:
        best_i = -1
        best_ratio = -1.0
        for i in remaining:
            deg = bin(conflict[i] & remaining_mask).count('1')
            ratio = (1.0 / i) / (1 + deg)
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i
        if best_i < 0:
            break
        chosen.add(best_i)
        total += 1.0 / best_i
        remaining_mask &= ~(1 << best_i)
        remaining.remove(best_i)
        # remove conflicts from remaining
        to_remove = []
        for j in remaining:
            if (conflict[best_i] >> j) & 1:
                to_remove.append(j)
        for j in to_remove:
            remaining.remove(j)
            remaining_mask &= ~(1 << j)
    if total > best_w:
        best_w = total
        best_set = set(chosen)

    # Try greedy descending i (largest first)
    chosen = set()
    total = 0.0
    for i in range(_N, 0, -1):
        ok = True
        for c in chosen:
            if (conflict[i] >> c) & 1:
                ok = False
                break
        if ok:
            chosen.add(i)
            total += 1.0 / i
    if total > best_w:
        best_w = total
        best_set = set(chosen)

    return best_set, best_w

def _local_search(conflict, start_set, weights, deadline):
    """1-swap local search: try removing one and adding another, or adding one."""
    chosen = set(start_set)
    total = sum(1.0 / i for i in chosen)
    chosen_mask = 0
    for i in chosen:
        chosen_mask |= (1 << i)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unchosen vertex
        best_gain = 0.0
        best_add = -1
        for i in range(1, _N + 1):
            if i in chosen:
                continue
            if (conflict[i] & chosen_mask) == 0:
                gain = 1.0 / i
                if gain > best_gain:
                    best_gain = gain
                    best_add = i
        if best_add > 0:
            chosen.add(best_add)
            chosen_mask |= (1 << best_add)
            total += best_gain
            improved = True
            continue
        # Try swap: remove r, add a
        for r in list(chosen):
            if time.time() >= deadline:
                break
            new_mask = chosen_mask & ~(1 << r)
            new_total = total - 1.0 / r
            for a in range(1, _N + 1):
                if a in chosen or a == r:
                    continue
                if (conflict[a] & new_mask) == 0:
                    gain = 1.0 / a
                    if new_total + gain > total + 1e-15:
                        total = new_total + gain
                        chosen_mask = new_mask | (1 << a)
                        chosen.remove(r)
                        chosen.add(a)
                        improved = True
                        break
            if improved:
                break
    return chosen, total

def _branch_and_bound(conflict, weights, deadline):
    """Branch and bound on the conflict graph.
    vertices 1..N, weight[i] = 1/i.
    Use a simple recursive B&B with a greedy coloring upper bound on the
    remaining candidate set."""
    # We'll work with a bitmask of available vertices.
    # Upper bound: sum of weights of all available vertices is a trivial bound.
    # Better: greedy coloring bound (like maximum clique on complement).
    # For maximum weight independent set, a simple bound is the sum of weights
    # of available vertices. We can improve with a greedy packing bound.

    full_mask = 0
    for i in range(1, _N + 1):
        full_mask |= (1 << i)

    best_weight = 0.0
    best_set = set()

    # Order vertices by descending weight for branching
    order = list(range(1, _N + 1))
    order.sort(key=lambda i: -1.0 / i)

    # Use a stack-based DFS
    # State: (available_mask, current_weight, current_set)
    # But storing set is expensive. Instead, we track decisions.

    # Simpler: recursive with a list of chosen.
    # Given time constraints, do a limited B&B.

    def _upper_bound(avail_mask):
        # Sum of weights of available vertices
        s = 0.0
        m = avail_mask
        while m:
            low = m & (-m)
            i = low.bit_length() - 1
            s += 1.0 / i
            m ^= low
        return s

    # Greedy coloring bound for the subgraph induced by avail_mask:
    # Color the conflict graph; the max weight independent set <= sum over
    # colors of max weight in that color class... actually for MWIS on the
    # conflict graph, we want to color the *complement*? No.
    # The conflict graph: edges mean cannot both be chosen.
    # An independent set in the conflict graph = no two chosen are adjacent.
    # Upper bound via coloring: color the conflict graph so adjacent vertices
    # have different colors. Then an independent set can contain at most one
    # vertex per... no, that's for clique. For independent set, the bound is
    # the sum over color classes of the max weight in each? No.
    # Actually: if we color the conflict graph with c colors, then the
    # complement graph has cliques of size... this is getting complicated.
    #
    # Simpler bound: greedy fractional packing. Just use sum of weights.

    def _dfs(avail_mask, current_weight, chosen_list):
        nonlocal best_weight, best_set
        if time.time() > deadline:
            return
        if avail_mask == 0:
            if current_weight > best_weight + 1e-15:
                best_weight = current_weight
                best_set = set(chosen_list)
            return
        # Upper bound
        ub = current_weight + _upper_bound(avail_mask)
        if ub <= best_weight + 1e-15:
            return
        # Pick a vertex to branch on: highest weight in avail
        # Find highest weight vertex in avail
        best_v = -1
        best_w = -1.0
        m = avail_mask
        while m:
            low = m & (-m)
            i = low.bit_length() - 1
            w = 1.0 / i
            if w > best_w:
                best_w = w
                best_v = i
            m ^= low
        if best_v < 0:
            if current_weight > best_weight + 1e-15:
                best_weight = current_weight
                best_set = set(chosen_list)
            return
        # Branch 1: include best_v
        new_avail = avail_mask & ~(conflict[best_v] | (1 << best_v))
        _dfs(new_avail, current_weight + best_w, chosen_list + [best_v])
        # Branch 2: exclude best_v
        _dfs(avail_mask & ~(1 << best_v), current_weight, chosen_list)

    _dfs(full_mask, 0.0, [])
    return best_set, best_weight

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # safety margin before 60s limit

    conflict = _build_graph()
    weights = [0.0] + [1.0 / i for i in range(1, _N + 1)]

    # Greedy initial solution
    init_set, init_w = _greedy_initial(conflict, weights)

    # Local search
    if time.time() < deadline - 5.0:
        ls_set, ls_w = _local_search(conflict, init_set, weights, deadline - 5.0)
        if ls_w > init_w:
            init_set, init_w = ls_set, ls_w

    # Try B&B if time permits
    best_set, best_w = init_set, init_w
    if time.time() < deadline - 2.0:
        bb_set, bb_w = _branch_and_bound(conflict, weights, deadline - 2.0)
        if bb_w > best_w:
            best_set, best_w = bb_set, bb_w

    # Final local search polish
    if time.time() < deadline - 0.5:
        ls_set, ls_w = _local_search(conflict, best_set, weights, deadline - 0.5)
        if ls_w > best_w:
            best_set, best_w = ls_set, ls_w

    # Ensure nonempty
    if not best_set:
        best_set = {1}

    return sorted(best_set)
