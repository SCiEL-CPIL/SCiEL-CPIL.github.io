"""Bootstrap search program: Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools

DEADLINE = 150
SAFETY_MARGIN = 12
START_TIME = None


def find_violation(adj_masks, n):
    """Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None.
    Uses bitmasks; checks size-6 subsets first (most numerous), then larger.
    """
    # Check size-6 subsets first - most common violation type
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            return combo

    # Check larger subsets
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            ok = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    ok = False
                    break
            if ok:
                return combo
    return None


def random_graph(n, p, rng):
    adj_masks = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append([u, v])
    return adj_masks, edges


def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e


def count_6violations(adj_masks, n, all_6subsets, subset_masks):
    """Count how many 6-subsets induce a regular subgraph."""
    count = 0
    for i, combo in enumerate(all_6subsets):
        mask = subset_masks[i]
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            count += 1
    return count


def local_search(adj_masks, n, rng, deadline):
    """Flip edges to eliminate regular induced subgraphs.
    Key insight: flipping an edge with both endpoints in a regular induced subgraph
    preserves regularity (both degrees change by same amount). Must flip boundary edges.
    Uses constraint propagation: for each violating set, try boundary edge flips and
    pick the one that reduces total violation count the most.
    """
    cur = adj_masks[:]

    # Precompute 6-subsets and their masks
    all_6subsets = [tuple(c) for c in itertools.combinations(range(n), 6)]
    subset_masks = []
    for combo in all_6subsets:
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subset_masks.append(mask)

    it = 0
    max_iters = 300
    while it < max_iters:
        if time.time() > deadline:
            break

        viol = find_violation(cur, n)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = list(viol)
        vset_set = set(vset)

        # Collect boundary edges (exactly one endpoint in vset)
        boundary_edges = []
        for u in vset:
            for v in range(n):
                if v not in vset_set:
                    key = (u, v) if u < v else (v, u)
                    boundary_edges.append(key)

        # Sample if too many
        if len(boundary_edges) > 25:
            boundary_edges = rng.sample(boundary_edges, 25)

        best_edge = None
        best_delta = -10**9
        base_count = None

        for (u, v) in boundary_edges:
            # Try flipping
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)

            # Count 6-violations
            vc = 0
            for i, combo in enumerate(all_6subsets):
                mask = subset_masks[i]
                d0 = (cur[combo[0]] & mask).bit_count()
                ok = True
                for vv in combo[1:]:
                    if (cur[vv] & mask).bit_count() != d0:
                        ok = False
                        break
                if ok:
                    vc += 1

            # Undo
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)

            if base_count is None:
                base_count = vc
                delta = 0
            else:
                delta = base_count - vc

            if delta > best_delta:
                best_delta = delta
                best_edge = (u, v)

        if best_edge is not None and best_delta > 0:
            u, v = best_edge
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)
        elif boundary_edges:
            # No improvement found; pick random boundary edge
            u, v = rng.choice(boundary_edges)
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)
        else:
            # Fallback: random edge
            u = rng.randrange(n)
            v = rng.randrange(n)
            while u == v:
                v = rng.randrange(n)
            if u > v:
                u, v = v, u
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)

        it += 1

    viol = find_violation(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        max_attempts = 12 if n >= 19 else 20

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            p = 0.35 + 0.3 * rng.random()

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break

            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            continue

    if not best_valid or best_n < 5:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
