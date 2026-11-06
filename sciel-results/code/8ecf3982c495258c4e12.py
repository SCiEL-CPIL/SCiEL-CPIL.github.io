import time
import random

DEADLINE = 145.0  # safety margin before 150s kill


def _has_triangle_fast(adj, n):
    """Fast triangle check using bitsets. Returns True if any triangle exists."""
    for i in range(n):
        ai = adj[i]
        j = ai & ~((1 << (i + 1)) - 1)  # only j > i
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            common = ai & adj[jj] & ~((1 << (jj + 1)) - 1)
            if common:
                return True
            j ^= lb
    return False


def _alpha_exact_bnb(adj, n, limit=10):
    """Exact max independent set size via branch and bound.
    Returns the max independent set size (capped at limit for early exit)."""
    best = [0]

    def bnb(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Pick vertex with max degree in candidate subgraph (heuristic ordering)
        # For speed with n<=40, pick lowest bit but use degree-aware ordering
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        # Branch: exclude v first (often prunes more)
        bnb(chosen_count, rest)
        # Branch: include v
        new_cands = rest & ~adj[v]
        bnb(chosen_count + 1, new_cands)

    bnb(0, (1 << n) - 1)
    return best[0]


def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_fast(adj, n):
        return False
    if _alpha_exact_bnb(adj, n, limit=10) >= 10:
        return False
    return True


def _construct_andrasfai(k):
    """Construct the Andrasfai graph A_k on k^2 vertices.
    Vertices are (x,y) in Z_k x Z_k. Edge between (x1,y1) and (x2,y2) if
    x1 != x2 and (y2-y1) mod k == (x1*x2) mod k."""
    n = k * k
    adj = [0] * n
    for i in range(n):
        x1 = i // k
        y1 = i % k
        for j in range(i + 1, n):
            x2 = j // k
            y2 = j % k
            if x1 != x2:
                if (y2 - y1) % k == (x1 * x2) % k:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    return adj, n


def _local_search_minimize_alpha(adj, n, deadline, target_alpha=9):
    """Local search: flip edges to reduce alpha while maintaining triangle-free.
    Returns (best_adj, best_alpha)."""
    adj = list(adj)
    best_adj = list(adj)
    best_alpha = _alpha_exact_bnb(adj, n, limit=10)

    if best_alpha <= target_alpha:
        return best_adj, best_alpha

    # Precompute all non-edges and edges for efficiency
    while time.time() < deadline:
        improved = False

        # Try adding edges (reduces alpha, must maintain triangle-free)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best_adj, best_alpha
                if adj[i] & (1 << j):
                    continue
                # Check if adding (i,j) creates a triangle
                common = adj[i] & adj[j]
                if common:
                    continue
                # Try adding
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                alpha = _alpha_exact_bnb(adj, n, limit=10)
                if alpha < best_alpha:
                    best_adj = list(adj)
                    best_alpha = alpha
                    improved = True
                    break
                # Revert
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
            if improved:
                break

        if improved:
            # Sync adj with best if we broke out
            continue

        # Try removing edges (may increase alpha, but sometimes helps overall)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best_adj, best_alpha
                if not (adj[i] & (1 << j)):
                    continue
                # Try removing
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                alpha = _alpha_exact_bnb(adj, n, limit=10)
                if alpha < best_alpha:
                    best_adj = list(adj)
                    best_alpha = alpha
                    improved = True
                    break
                # Revert
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            if improved:
                break

        if not improved:
            # No single-edge flip improves; try 2-edge flips (small neighborhood)
            found_two = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return best_adj, best_alpha
                    if adj[i] & (1 << j):
                        continue
                    common1 = adj[i] & adj[j]
                    if common1:
                        continue
                    # Try adding (i,j)
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

                    for k2 in range(n):
                        for l2 in range(k2 + 1, n):
                            if time.time() > deadline:
                                break
                            if (k2, l2) == (i, j):
                                continue
                            if adj[k2] & (1 << l2):
                                continue
                            common2 = adj[k2] & adj[l2]
                            if common2:
                                continue
                            # Try adding (k2,l2)
                            adj[k2] |= (1 << l2)
                            adj[l2] |= (1 << k2)

                            alpha = _alpha_exact_bnb(adj, n, limit=10)
                            if alpha < best_alpha:
                                best_adj = list(adj)
                                best_alpha = alpha
                                found_two = True
                                improved = True
                                # Revert second
                                adj[k2] ^= (1 << l2)
                                adj[l2] ^= (1 << k2)
                                # Revert first
                                adj[i] ^= (1 << j)
                                adj[j] ^= (1 << i)
                                break
                            # Revert second
                            adj[k2] ^= (1 << l2)
                            adj[l2] ^= (1 << k2)
                        if found_two:
                            break
                    if found_two:
                        break
                    # Revert first
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                if found_two:
                    break

            if not improved:
                # Stuck in local optimum; do a random perturbation
                # Add a random set of valid edges to escape
                perturbed = list(adj)
                random.seed(int(time.time() * 1000) % 100000)
                for _ in range(5):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    a, b = min(i, j), max(i, j)
                    if not (perturbed[a] & (1 << b)):
                        if not (perturbed[a] & perturbed[b]):
                            perturbed[a] |= (1 << b)
                            perturbed[b] |= (1 << a)
                adj = perturbed
                alpha = _alpha_exact_bnb(adj, n, limit=10)
                if alpha < best_alpha:
                    best_adj = list(adj)
                    best_alpha = alpha

    return best_adj, best_alpha


def _random_triangle_free(n, seed=0):
    """Generate a random triangle-free graph on n vertices."""
    random.seed(seed)
    adj = [0] * n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(pairs)
    for i, j in pairs:
        common = adj[i] & adj[j]
        if common == 0:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    return adj


def _edges_from_adj(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges


def construct():
    start_time = time.time()

    # Strategy 1: Andrasfai graph subgraph + local search refinement
    # A_9 has 81 vertices, triangle-free, alpha=9 (or similar small value)
    # Take a 40-vertex induced subgraph, then optimize edges to minimize alpha

    best_result = None
    best_n = 0

    # Try Andrasfai A_9 subgraph on 40 vertices
    k = 9
    adj_full, n_full = _construct_andrasfai(k)

    for n_try in [40, 39]:
        if time.time() > start_time + DEADLINE - 20:
            break
        if n_full >= n_try:
            adj = [0] * n_try
            for i in range(n_try):
                adj[i] = adj_full[i] & ((1 << n_try) - 1)

            # Check initial validity
            alpha_init = _alpha_exact_bnb(adj, n_try, limit=10)
            if alpha_init < 10:
                # Already valid!
                if n_try > best_n:
                    best_n = n_try
                    best_result = {'n': n_try, 'edges': _edges_from_adj(adj, n_try)}
                continue

            # Need to reduce alpha via local search
            search_deadline = start_time + DEADLINE - 15
            best_adj, best_alpha = _local_search_minimize_alpha(
                adj, n_try, search_deadline, target_alpha=9
            )
            if best_alpha < 10:
                if n_try > best_n:
                    best_n = n_try
                    best_result = {'n': n_try, 'edges': _edges_from_adj(best_adj, n_try)}

    if best_result and best_n >= 39:
        return best_result

    # Strategy 2: Try Andrasfai A_8 (64 vertices) subgraphs
    k = 8
    adj_full, n_full = _construct_andrasfai(k)

    for n_try in [40, 39, 38, 37]:
        if time.time() > start_time + DEADLINE - 15:
            break
        if n_full >= n_try:
            adj = [0] * n_try
            for i in range(n_try):
                adj[i] = adj_full[i] & ((1 << n_try) - 1)

            alpha_init = _alpha_exact_bnb(adj, n_try, limit=10)
            if alpha_init < 10:
                if n_try > best_n:
                    best_n = n_try
                    best_result = {'n': n_try, 'edges': _edges_from_adj(adj, n_try)}
                continue

            search_deadline = start_time + DEADLINE - 10
            best_adj, best_alpha = _local_search_minimize_alpha(
                adj, n_try, search_deadline, target_alpha=9
            )
            if best_alpha < 10:
                if n_try > best_n:
                    best_n = n_try
                    best_result = {'n': n_try, 'edges': _edges_from_adj(best_adj, n_try)}

    if best_result and best_n >= 35:
        return best_result

    # Strategy 3: Random triangle-free graphs + local search
    for n_try in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > start_time + DEADLINE - 10:
            break
        found = False
        for attempt in range(20):
            if time.time() > start_time + DEADLINE - 5:
                break
            adj = _random_triangle_free(n_try, seed=attempt * 1000 + n_try)
            alpha = _alpha_exact_bnb(adj, n_try, limit=10)
            if alpha < 10:
                if n_try > best_n:
                    best_n = n_try
                    best_result = {'n': n_try, 'edges': _edges_from_adj(adj, n_try)}
                found = True
                break

            # Try local search to reduce alpha
            search_deadline = min(start_time + DEADLINE - 5, time.time() + 3)
            best_adj, best_alpha = _local_search_minimize_alpha(
                adj, n_try, search_deadline, target_alpha=9
            )
            if best_alpha < 10:
                if n_try > best_n:
                    best_n = n_try
                    best_result = {'n': n_try, 'edges': _edges_from_adj(best_adj, n_try)}
                found = True
                break

    if best_result and best_n >= 30:
        return best_result

    # Fallback: C5 (n=5, triangle-free, alpha=2)
    return {'n': 5, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]}
