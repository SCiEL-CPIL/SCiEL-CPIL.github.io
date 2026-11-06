"""Maximum independent set on the square-sum conflict graph via branch-and-bound with bitset pruning."""

# EVOLVE-BLOCK-START
import math

def _is_square(n):
    r = math.isqrt(n)
    return r * r == n

def _build_conflict_bitsets():
    """Return list of ints where bit j set in masks[i] means i and j conflict (i+j is a perfect square)."""
    N = 256
    masks = [0] * N
    for i in range(N):
        val = i + 1
        m = 0
        # diagonal: 2*val square?
        if _is_square(2 * val):
            m |= (1 << i)
        for j in range(i + 1, N):
            if _is_square(val + (j + 1)):
                m |= (1 << j)
        masks[i] = m
    return masks

def _greedy_independent_set(cands, masks):
    """Greedy: repeatedly pick a vertex with smallest degree among cands, remove it and its neighbors.
    Returns the bitset of the selected vertices (an independent set)."""
    selected = 0
    remaining = cands
    while remaining:
        # pick vertex with min (degree within remaining)
        # iterate bits
        best_v = -1
        best_deg = 1 << 30
        temp = remaining
        while temp:
            low = temp & (-temp)
            v = low.bit_length() - 1
            deg = (masks[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
            temp -= low
        selected |= (1 << best_v)
        remaining &= ~(masks[best_v] | (1 << best_v))
    return selected

def _upper_bound_clique_cover(C, masks):
    """Compute a valid upper bound on the size of the maximum independent set in C.
    This is the minimum number of cliques in G[C] that cover C (Clique Cover Number).
    We approximate this with a greedy clique cover."""
    V = C
    k = 0
    while V:
        # Find a vertex with max degree in G[V] to start a clique
        best_v = -1
        max_deg = -1
        temp = V
        while temp:
            low = temp & (-temp)
            v = low.bit_length() - 1
            deg = (masks[v] & V).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp -= low
        
        if best_v == -1:
            break
            
        # Greedily build a maximal clique starting from best_v
        clique = 1 << best_v
        # Candidates must be neighbors of best_v in V
        pool = masks[best_v] & V
        
        # We want to add vertices from pool to clique.
        # A vertex u can be added if it is connected to all vertices currently in clique.
        # Initially, clique={best_v}. pool contains neighbors of best_v.
        # If we add u, new pool = pool & masks[u].
        
        # To maximize clique size (and minimize k), we should pick u carefully.
        # A simple heuristic: pick u with max degree in pool.
        while pool:
            # Pick u with max degree in pool
            best_u = -1
            max_deg_u = -1
            temp_pool = pool
            while temp_pool:
                low = temp_pool & (-temp_pool)
                u = low.bit_length() - 1
                deg_u = (masks[u] & pool).bit_count()
                if deg_u > max_deg_u:
                    max_deg_u = deg_u
                    best_u = u
                temp_pool -= low
            
            if best_u == -1:
                break
                
            # Add best_u to clique
            clique |= (1 << best_u)
            # Update pool: must be connected to best_u
            pool &= masks[best_u]
            # Remove best_u from pool (it's in clique now)
            pool &= ~(1 << best_u)
            
        # Remove the entire clique from V
        V &= ~clique
        k += 1
        
    return k

def _solve_max_independent_set(masks, N, time_budget=None):
    """Branch and bound for maximum independent set. Returns a list of selected vertex indices."""
    import time
    start = time.time()
    full = (1 << N) - 1

    # Precompute degree ordering: sort vertices by degree descending for branching
    order = sorted(range(N), key=lambda v: masks[v].bit_count(), reverse=True)
    # We'll branch on vertices in this order; but for B&B we need dynamic candidate set.
    # Use a recursive search over candidate set C (bitset).
    # best_solution stored as bitset for speed, converted at end.
    best = [0]  # bitset
    best_size = [0]

    # Upper bound function: greedy independent set size of C
    def upper_bound(C):
        # greedy: repeatedly pick vertex of min degree in C
        count = 0
        rem = C
        while rem:
            best_v = -1
            best_deg = 1 << 30
            temp = rem
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                deg = (masks[v] & rem).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                temp -= low
            count += 1
            rem &= ~(masks[best_v] | (1 << best_v))
        return count

    def search(C, current):
        import time as _t
        if _t.time() - start > time_budget:
            return
        
        # Update incumbent with current + greedy(C)
        greedy_set = _greedy_independent_set(C, masks)
        greedy_size = greedy_set.bit_count()
        if len(current) + greedy_size > best_size[0]:
            best_size[0] = len(current) + greedy_size
            # Construct the bitset for the incumbent
            best[0] = 0
            for v in current:
                best[0] |= (1 << v)
            best[0] |= greedy_set
        # Upper bound: if |current| + ub(C) <= best_size, prune
        ub = _upper_bound_clique_cover(C, masks)
        if len(current) + ub <= best_size[0]:
            return
        
        if C == 0:
            if len(current) > best_size[0]:
                best_size[0] = len(current)
                best[0] = 0
                for v in current:
                    best[0] |= (1 << v)
            return

        # Pick a vertex to branch on: choose one with high degree in C for better pruning
        # But we want to include/exclude. Standard: pick v, branch include v, then exclude v (remove v and neighbors from C).
        # Choose v as vertex with max degree in C to maximize pruning in the exclude branch.
        temp = C
        v = -1
        maxdeg = -1
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            deg = (masks[idx] & C).bit_count()
            if deg > maxdeg:
                maxdeg = deg
                v = idx
            temp -= low

        # Branch 1: include v
        newC = C & ~(masks[v] | (1 << v))
        search(newC, current + [v])

        # Branch 2: exclude v
        C2 = C & ~(1 << v)
        search(C2, current)

    # Initial upper bound on full set
    search(full, [])

    # Convert best bitset to list of values (1-indexed)
    result = []
    b = best[0]
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        result.append(idx + 1)
        b -= low
    return result

def construct():
    import time
    start = time.time()
    N = 256
    masks = _build_conflict_bitsets()
    # Use branch and bound with time budget
    time_budget = 55.0
    result = _solve_max_independent_set(masks, N, time_budget)
    # Fallback if something went wrong
    if not result:
        return list(range(1, 257, 3))
    # Validate: ensure it's a valid independent set
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a <= b:
                if _is_square(a + b):
                    # invalid; fall back
                    return list(range(1, 257, 3))
    return sorted(res_set)
# EVOLVE-BLOCK-END
