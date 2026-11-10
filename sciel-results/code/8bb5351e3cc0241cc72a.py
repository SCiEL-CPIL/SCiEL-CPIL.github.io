import time
import random
from itertools import combinations

# Precompute all 560 triples on 0..15 as sorted tuples
ALL_TRIPLES = list(combinations(range(16), 3))

# Precompute bitmask for each triple
TRIPLE_MASKS = []
for t in ALL_TRIPLES:
    m = 0
    for p in t:
        m |= (1 << p)
    TRIPLE_MASKS.append(m)

# Precompute the 3 unordered pairs (as frozensets) for each triple
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    TRIPLE_PAIRS.append((
        (t[0], t[1]),
        (t[0], t[2]),
        (t[1], t[2]),
    ))


def calc_alpha(edges):
    """Compute the independence number of the hypergraph given by edges.
    Uses bitmask containment tests. Fast enough for ~37 edges."""
    if not edges:
        return 16
    masks = [TRIPLE_MASKS[ALL_TRIPLES.index(e)] if False else 0 for e in []]  # placeholder
    # Build masks directly
    edge_masks = []
    for e in edges:
        m = 0
        for p in e:
            m |= (1 << p)
        edge_masks.append(m)
    n = 16
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            indep = True
            for em in edge_masks:
                if (mask & em) == em:
                    indep = False
                    break
            if indep:
                return k
    return 1


def find_one_independent_set(edges, k):
    """Return one independent set of size k, or None if none exists."""
    edge_masks = []
    for e in edges:
        m = 0
        for p in e:
            m |= (1 << p)
        edge_masks.append(m)
    for subset in combinations(range(16), k):
        mask = 0
        for p in subset:
            mask |= (1 << p)
        indep = True
        for em in edge_masks:
            if (mask & em) == em:
                indep = False
                break
        if indep:
            return set(subset)
    return None


def is_valid_edge(edge, used_pairs):
    """Check if edge can be added given the used pairs (linearity)."""
    for p in TRIPLE_PAIRS[ALL_TRIPLES.index(edge)]:
        if p in used_pairs:
            return False
    return True


def build_initial():
    """Random greedy construction of a linear triple system."""
    used_pairs = set()
    edges = []
    candidates = ALL_TRIPLES[:]
    random.shuffle(candidates)
    for t in candidates:
        pairs = TRIPLE_PAIRS[ALL_TRIPLES.index(t)]
        ok = True
        for p in pairs:
            if p in used_pairs:
                ok = False
                break
        if ok:
            edges.append(t)
            used_pairs.update(pairs)
    return edges, used_pairs


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    # Initial construction
    best_edges, best_used_pairs = build_initial()
    best_alpha = calc_alpha(best_edges)

    # Local search: remove one edge, try to add a better one
    # Optionally bias towards breaking an independent set
    while time.time() < deadline:
        current_alpha = best_alpha
        if current_alpha <= 5:
            break

        # Pick an edge to remove
        idx = random.randint(0, len(best_edges) - 1)
        removed = best_edges[idx]
        best_edges.pop(idx)

        # Recompute used pairs after removal (rebuild is cheap for ~37 edges)
        used_pairs = set()
        for e in best_edges:
            used_pairs.update(TRIPLE_PAIRS[ALL_TRIPLES.index(e)])

        # Find candidate edges that are now valid
        candidates = []
        for t in ALL_TRIPLES:
            if t in best_edges:
                continue
            ok = True
            for p in TRIPLE_PAIRS[ALL_TRIPLES.index(t)]:
                if p in used_pairs:
                    ok = False
                    break
            if ok:
                candidates.append(t)

        if not candidates:
            # Restore
            best_edges.append(removed)
            continue

        # Try a few candidates, pick the one that gives best alpha
        # To save time, only try a random subset
        n_try = min(len(candidates), 5)
        sample = random.sample(candidates, n_try)

        best_new_alpha = current_alpha
        best_new_edge = None
        for cand in sample:
            trial = best_edges[:]
            trial.append(cand)
            a = calc_alpha(trial)
            if a < best_new_alpha:
                best_new_alpha = a
                best_new_edge = cand
            elif a == best_new_alpha and best_new_edge is None:
                best_new_edge = cand

        if best_new_edge is not None and best_new_alpha <= current_alpha:
            # Accept
            best_edges.append(best_new_edge)
            if best_new_alpha < current_alpha:
                best_alpha = best_new_alpha
        else:
            # Restore removed edge
            best_edges.append(removed)

    return [list(e) for e in best_edges]


if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
