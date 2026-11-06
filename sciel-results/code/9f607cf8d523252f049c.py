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
    rng = None  # not needed
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            # degrees within the induced subgraph
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


def local_search(adj_masks, n, rng, deadline):
    """Flip edges to eliminate regular induced subgraphs. Prefer flipping edges inside a violating set."""
    cur = adj_masks[:]

    viol = find_violation(cur, n)
    if viol is None:
        return n, get_edges_from_masks(cur, n), True

    it = 0
    while it < 500:
        if time.time() > deadline:
            break

        viol = find_violation(cur, n)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = set(viol)
        # candidate edges: prefer edges within the violating set, then all edges
        internal = [(u, v) for u in vset for v in vset if u < v and (cur[u] & (1 << v))]
        # also include non-edges within the set (adding an edge can break regularity)
        internal_nonedges = [(u, v) for u in vset for v in vset if u < v and not (cur[u] & (1 << v))]
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

        if internal:
            u, v = rng.choice(internal)
        elif internal_nonedges:
            u, v = rng.choice(internal_nonedges)
        else:
            u, v = rng.choice(all_edges)

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

        # More restarts for larger n; the random graph is unlikely to be valid,
        # so local search must do the heavy lifting.
        max_attempts = 8 if n >= 19 else 15

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Vary edge probability around 0.5
            p = 0.35 + 0.3 * rng.random()

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break  # found a valid graph for this n, move to next n

            # Local search to repair
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            # Already have a valid graph at this size or larger
            continue

    if not best_valid or best_n < 5:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
