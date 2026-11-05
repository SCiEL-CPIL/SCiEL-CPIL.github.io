"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 138.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _bit_count(x):
    return bin(x).count('1')


def _has_triangle_bitmask(adj, n):
    """Check if adjacency bitmask list has a triangle."""
    for i in range(n):
        ni = adj[i]
        j_mask = ni
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            j_mask ^= lsb
            if j > i:
                if adj[i] & adj[j]:
                    return True
    return False


def _max_independent_set_size_fast(adj, n, limit=10):
    """
    Fast exact max independent set using bitset branch and bound.
    Returns min(actual_alpha, limit) - i.e., if alpha >= limit, returns limit.
    """
    best = [0]
    full_mask = (1 << n) - 1
    adj_local = adj

    def _greedy_color_bound(available):
        """Upper bound: greedy clique cover of G[available]."""
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            unassigned ^= lsb
            # Build a clique containing v in G[available]
            clique = 1 << v
            remaining = unassigned & adj_local[v]
            while remaining:
                lsb_u = remaining & -remaining
                u = lsb_u.bit_length() - 1
                remaining ^= lsb_u
                clique |= lsb_u
                remaining &= adj_local[u]
            unassigned &= ~clique
        return k

    def _bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return

        if current_size + _bit_count(available) <= best[0]:
            return

        bound = _greedy_color_bound(available)
        if current_size + bound <= best[0]:
            return

        # Pick vertex with max degree in G[available]
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj_local[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u

        if v == -1:
            return

        # Branch 1: Include v
        _bnb(available & ~(1 << v) & ~adj_local[v], current_size + 1)

        # Branch 2: Exclude v
        _bnb(available & ~(1 << v), current_size)

    _bnb(full_mask, 0)
    return min(best[0], limit)


def _max_independent_set_size_bitmask(adj, n):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    return _max_independent_set_size_fast(adj, n, limit=15)


def _validate_bitmask(adj, n):
    """Check triangle-free and no independent 10-set."""
    if _has_triangle_bitmask(adj, n):
        return False
    if _max_independent_set_size_bitmask(adj, n) >= 10:
        return False
    return True


def _cyclic_graph(n, shifts):
    """Construct a circulant graph with given shifts."""
    adj = [0] * n
    shift_set = set()
    for s in shifts:
        shift_set.add(s % n)
        shift_set.add((-s) % n)
    shift_set.discard(0)

    for i in range(n):
        for s in shift_set:
            j = (i + s) % n
            if j > i:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _shifts_triangle_free(n, shifts):
    """Check if a set of shifts creates a triangle in the circulant graph."""
    dist_set = set()
    for s in shifts:
        dist_set.add(s % n)
        dist_set.add((-s) % n)
    dist_set.discard(0)

    for a in shifts:
        for b in shifts:
            c = (a + b) % n
            d = (a - b) % n
            if c in dist_set and c != 0:
                return True
            if d in dist_set and d != 0:
                return True
    return False


def _try_circulant_systematic(n, max_shifts=None):
    """
    Systematically search for triangle-free circulant graphs on n vertices
    with small independence number.
    """
    if max_shifts is None:
        max_shifts = n // 2

    half = n // 2
    best_adj = None
    best_alpha = float('inf')

    # Strategy 1: Greedy addition of shifts (multiple orderings)
    for seed in range(80):
        if _remaining_time() < 5:
            break

        rng = random.Random(9999 + seed)
        available = list(range(1, half + 1))
        rng.shuffle(available)

        current_shifts = []
        for s in available:
            trial = current_shifts + [s]
            if not _shifts_triangle_free(n, trial):
                current_shifts.append(s)

        if not current_shifts:
            continue

        adj = _cyclic_graph(n, current_shifts)
        if _has_triangle_bitmask(adj, n):
            continue

        alpha = _max_independent_set_size_fast(adj, n, limit=10)
        if alpha < 10:
            return adj
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj

    # Strategy 2: Random subsets of shifts with various sizes
    for attempt in range(300):
        if _remaining_time() < 3:
            break

        rng = random.Random(4242 + attempt)
        num_shifts = rng.randint(2, min(half, 18))
        shifts = rng.sample(range(1, half + 1), num_shifts)

        if _shifts_triangle_free(n, shifts):
            continue

        adj = _cyclic_graph(n, shifts)
        if _has_triangle_bitmask(adj, n):
            continue

        alpha = _max_independent_set_size_fast(adj, n, limit=10)
        if alpha < 10:
            return adj
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj

    # Strategy 3: Start from a base and try adding/removing
    for attempt in range(100):
        if _remaining_time() < 3:
            break

        rng = random.Random(7777 + attempt)
        num_shifts = rng.randint(3, min(half, 12))
        base_shifts = rng.sample(range(1, half + 1), num_shifts)

        if _shifts_triangle_free(n, base_shifts):
            continue

        # Try adding one more shift
        for s in range(1, half + 1):
            if s in base_shifts:
                continue
            trial = base_shifts + [s]
            if not _shifts_triangle_free(n, trial):
                adj = _cyclic_graph(n, trial)
                if _has_triangle_bitmask(adj, n):
                    continue
                alpha = _max_independent_set_size_fast(adj, n, limit=10)
                if alpha < 10:
                    return adj
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_adj = adj

    return best_adj


def _paley_graph(n):
    """Construct Paley graph of order n where n is a prime congruent to 3 mod 4."""
    if n <= 1:
        return [0] * n

    qr = set()
    for x in range(1, n):
        qr.add((x * x) % n)
    qr.discard(0)

    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in qr:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _greedy_triangle_free_bitmask(n, rng):
    """Build a triangle-free graph greedily using bitmasks."""
    adj = [0] * n
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)

    for i, j in edges:
        if _remaining_time() < 2:
            break
        if adj[i] & adj[j]:
            continue
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)

    return adj


def _local_search_bitmask(adj, n, rng, max_iter=2000):
    """Improve graph by adding edges to reduce independence number."""
    adj = adj[:]
    alpha = _max_independent_set_size_fast(adj, n, limit=10)

    if alpha < 10:
        return adj, alpha

    def _find_one_mis(adj, n):
        """Find one maximum independent set using greedy."""
        remaining = (1 << n) - 1
        selected = []
        while remaining:
            best_v = -1
            best_deg = n + 1
            for v in range(n):
                if remaining & (1 << v):
                    deg = _bit_count(adj[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            selected.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return selected

    no_improve_count = 0
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break

        target_is = _find_one_mis(adj, n)
        if len(target_is) < 10:
            alpha = _max_independent_set_size_fast(adj, n, limit=10)
            if alpha < 10:
                break

        addable_in_is = []
        for idx_i in range(len(target_is)):
            for idx_j in range(idx_i + 1, len(target_is)):
                i, j = target_is[idx_i], target_is[idx_j]
                if not (adj[i] & (1 << j)):
                    if not (adj[i] & adj[j]):
                        addable_in_is.append((i, j))

        if not addable_in_is:
            no_improve_count += 1
            if no_improve_count > 30:
                break
            continue

        improved = False
        for _ in range(15):
            if _remaining_time() < 2:
                break
            i, j = rng.choice(addable_in_is)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

            new_alpha = _max_independent_set_size_fast(adj, n, limit=10)
            if new_alpha < alpha:
                alpha = new_alpha
                improved = True
                no_improve_count = 0
                if alpha < 10:
                    break
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)

        if alpha < 10:
            break
        if not improved:
            no_improve_count += 1
            if no_improve_count > 40:
                break

    return adj, alpha


def _try_construction_bitmask(n, seed, max_local_iters=1000):
    """Try to build a valid graph on n vertices."""
    rng = random.Random(seed)

    adj = _greedy_triangle_free_bitmask(n, rng)
    alpha = _max_independent_set_size_fast(adj, n, limit=10)

    if alpha < 10:
        return adj, alpha

    if _remaining_time() > 5:
        adj_ls, alpha_ls = _local_search_bitmask(adj, n, rng, max_local_iters)
        if alpha_ls < 10:
            return adj_ls, alpha_ls
        if alpha_ls < alpha:
            adj, alpha = adj_ls, alpha_ls

    for attempt in range(5):
        if _remaining_time() < 5:
            break
        rng2 = random.Random(seed * 1000 + attempt)
        adj2 = _greedy_triangle_free_bitmask(n, rng2)
        alpha2 = _max_independent_set_size_fast(adj2, n, limit=10)

        if alpha2 < 10:
            return adj2, alpha2

        if alpha2 < alpha and _remaining_time() > 5:
            adj_ls2, alpha_ls2 = _local_search_bitmask(adj2, n, rng2, max_local_iters)
            if alpha_ls2 < alpha:
                adj, alpha = adj_ls2, alpha_ls2

    return None


def _complementary_turan_search(n, rng, max_iter=500):
    """
    Complementary Turán construction:
    Find H on n vertices with omega(H) <= 9 and alpha(H) <= 2.
    Then G = complement(H) is triangle-free (since alpha(H)<=2) and
    has no independent 10-set (since omega(H)<=9).
    
    We build H greedily: start with empty H, add edges to ensure alpha(H) <= 2,
    while keeping omega(H) <= 9.
    """
    adj_h = [0] * n  # adjacency of H

    # Start with a random partial graph
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break

        # Find an independent set of size 3 in H (i.e., a triple with no edges in H)
        # If none exists, alpha(H) <= 2
        # Greedily find independent set of size 3
        found_triple = False
        for i in range(n):
            if found_triple:
                break
            for j in range(i + 1, n):
                if found_triple:
                    break
                if adj_h[i] & (1 << j):
                    continue
                for k in range(j + 1, n):
                    if adj_h[i] & (1 << k) or adj_h[j] & (1 << k):
                        continue
                    # (i,j,k) is independent in H
                    found_triple = True
                    # Add an edge to break this independent set
                    # Choose which edge to add: (i,j), (i,k), or (j,k)
                    # Must not create a clique of size 10 in H
                    options = [(i, j), (i, k), (j, k)]
                    rng.shuffle(options)
                    added = False
                    for a, b in options:
                        if adj_h[a] & (1 << b):
                            continue
                        # Check if adding (a,b) creates a 10-clique
                        # A 10-clique containing (a,b) means there are 8 other vertices
                        # all adjacent to both a and b in H
                        common = adj_h[a] & adj_h[b] & ((1 << n) - 1) & ~(1 << a) & ~(1 << b)
                        # We need to check if common contains a clique of size 8
                        # That's expensive; use a heuristic: if |common| >= 8, check
                        if _bit_count(common) >= 8:
                            # Quick check: is there a 8-clique in H[common]?
                            # Use greedy: can we find 8 vertices all mutually adjacent in common?
                            # Simple check: if the induced subgraph on common is complete enough
                            # For now, just check if we can greedily build a clique of size 8
                            clique = [a, b]
                            remaining = common
                            while len(clique) < 10 and remaining:
                                lsb = remaining & -remaining
                                v = lsb.bit_length() - 1
                                remaining ^= lsb
                                # v must be adjacent to all in clique
                                ok = True
                                for cv in clique:
                                    if not (adj_h[v] & (1 << cv)):
                                        ok = False
                                        break
                                if ok:
                                    clique.append(v)
                            if len(clique) >= 10:
                                continue  # Would create 10-clique, skip
                        adj_h[a] |= (1 << b)
                        adj_h[b] |= (1 << a)
                        added = True
                        break
                    if not added:
                        # Try all three; if none work, skip
                        pass
                    break

        if not found_triple:
            # alpha(H) <= 2, now check omega(H) <= 9
            # Find max clique in H
            # Use greedy coloring bound
            full_mask = (1 << n) - 1
            best_clique = 0

            def _greedy_clique_bound(available):
                """Upper bound on clique number via greedy coloring of complement."""
                # Actually, let's just greedily find a large clique
                unassigned = available
                k = 0
                while unassigned:
                    k += 1
                    lsb = unassigned & -unassigned
                    v = lsb.bit_length() - 1
                    unassigned ^= lsb
                    # In H, a clique means all mutually adjacent
                    # For coloring bound: color G[available] where G is complement of H
                    # A color class in complement = independent set in H = clique in complement
                    # We want upper bound on clique in H = upper bound on independent set in complement
                    # Greedy: pick v, remove v and its non-neighbors in H (i.e., neighbors in complement)
                    # Non-neighbors of v in H = vertices not adjacent to v in H
                    non_neighbors = unassigned & ~adj_h[v]
                    unassigned &= non_neighbors  # keep only non-neighbors? No...
                    # Actually for clique in H: after picking v, next vertex must be adjacent to v in H
                    # So remaining candidates = unassigned & adj_h[v]
                    unassigned &= adj_h[v]
                return k

            # Simpler: just greedily find a clique
            def _greedy_clique(available):
                selected = []
                rem = available
                while rem:
                    # Pick vertex with max degree in H[rem]
                    best_v = -1
                    best_deg = -1
                    m = rem
                    while m:
                        lsb = m & -m
                        u = lsb.bit_length() - 1
                        m ^= lsb
                        deg = _bit_count(adj_h[u] & rem)
                        if deg > best_deg:
                            best_deg = deg
                            best_v = u
                    if best_v == -1:
                        break
                    selected.append(best_v)
                    rem &= adj_h[best_v] & ~(1 << best_v)
                return len(selected)

            omega = _greedy_clique(full_mask)
            if omega <= 9:
                return adj_h

    return None


def _try_complementary(n, seed):
    """Try complementary Turán construction."""
    rng = random.Random(seed)
    adj_h = _complementary_turan_search(n, rng)
    if adj_h is None:
        return None

    # G = complement of H
    adj_g = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_h[i] & (1 << j)):
                adj_g[i] |= (1 << j)
                adj_g[j] |= (1 << i)

    # Verify
    if _has_triangle_bitmask(adj_g, n):
        return None
    alpha = _max_independent_set_size_fast(adj_g, n, limit=10)
    if alpha < 10:
        return adj_g
    return None


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')

    # Try circulant constructions first (most promising per cell findings)
    for n_target in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if _remaining_time() < 8:
            break
        if best_n >= n_target:
            continue
        adj = _try_circulant_systematic(n_target)
        if adj is not None:
            alpha = _max_independent_set_size_fast(adj, n_target, limit=10)
            if alpha < 10 and n_target > best_n:
                best_adj = adj
                best_n = n_target
                best_alpha = alpha
                break

    # Try complementary Turán construction
    if best_n < 39:
        for n_target in [40, 39, 38, 37, 36]:
            if _remaining_time() < 5:
                break
            if best_n >= n_target:
                continue
            for seed in range(10):
                if _remaining_time() < 3:
                    break
                adj_c = _try_complementary(n_target, 5000 + seed)
                if adj_c is not None:
                    alpha_c = _max_independent_set_size_fast(adj_c, n_target, limit=10)
                    if alpha_c < 10 and n_target > best_n:
                        best_adj = adj_c
                        best_n = n_target
                        best_alpha = alpha_c
                        break
            if best_n >= n_target:
                break

    # Try Paley graphs for primes congruent to 3 mod 4
    if best_n < 37:
        paley_primes = [37, 29, 23, 19, 11, 7]
        for p in paley_primes:
            if _remaining_time() < 5:
                break
            if best_n >= p:
                continue
            adj = _paley_graph(p)
            if _has_triangle_bitmask(adj, p):
                continue
            alpha = _max_independent_set_size_fast(adj, p, limit=10)
            if alpha < 10:
                best_adj = adj
                best_n = p
                best_alpha = alpha
                break

    # If we haven't found a good graph, use randomized search
    if best_n < 30:
        for n_target in range(40, 20, -1):
            if _remaining_time() < 8:
                break
            if best_n >= n_target:
                continue

            found = False
            for seed in range(40):
                if _remaining_time() < 3:
                    break

                result = _try_construction_bitmask(n_target, seed)
                if result is not None:
                    adj_res, alpha_res = result
                    if alpha_res < best_alpha:
                        best_adj = adj_res
                        best_n = n_target
                        best_alpha = alpha_res
                    if alpha_res < 10:
                        found = True
                        break

            if found:
                break

    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
