"""Rewritten program: structural change to a local-search / colour-merging family."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

# Precompute all K4s (sets of 4 vertices) and the edges within each.
_N = 16
_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_IDX = {(i, j): k for k, (i, j) in enumerate(_EDGES)}
_K4S = list(itertools.combinations(range(_N), 4))
# For each K4, list of edge indices (6 edges)
_K4_EDGES = []
for quad in _K4S:
    eidx = []
    for a, b in itertools.combinations(quad, 2):
        eidx.append(_EDGE_IDX[(a, b)])
    _K4_EDGES.append(eidx)
_NUM_K4 = len(_K4S)


def _count_colours_k4(colour, edge_idx_list):
    """Return number of distinct colours among the given edges."""
    c0 = colour[edge_idx_list[0]]
    c1 = colour[edge_idx_list[1]]
    c2 = colour[edge_idx_list[2]]
    c3 = colour[edge_idx_list[3]]
    c4 = colour[edge_idx_list[4]]
    c5 = colour[edge_idx_list[5]]
    # distinct count via set is fine for 6 items
    return len({c0, c1, c2, c3, c4, c5})


def _is_valid(colour):
    """Check every K4 has >=5 distinct colours."""
    for eidx in _K4_EDGES:
        if _count_colours_k4(colour, eidx) < 5:
            return False
    return True


def _count_colours_used(colour):
    return len(set(colour))


def _initial_colouring():
    """Start from the seed (i+j) mod 17, which uses 17 colours and is valid."""
    return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]


def _try_merge(colour, c_from, c_to, rng):
    """
    Attempt to merge colour c_from into c_to.
    Returns (new_colour, True) if valid after merge, else (None, False).
    c_from must be a colour present in the current colouring.
    """
    # Find edges with colour c_from
    affected = [k for k, c in enumerate(colour) if c == c_from]
    if not affected:
        return None, False

    # Tentatively recolor
    new_colour = list(colour)
    for k in affected:
        new_colour[k] = c_to

    # Quick validity check: only K4s that contain at least one affected edge can be broken.
    # Build a map from edge index -> list of K4 indices containing that edge.
    # We precompute this globally below.
    for k4_idx in _AFFECTED_K4S_FOR_EDGE_UNION(affected):
        if _count_colours_k4(new_colour, _K4_EDGES[k4_idx]) < 5:
            return None, False
    return new_colour, True


# Precompute for each edge index, the list of K4 indices containing it.
_EDGE_TO_K4S = [[] for _ in range(len(_EDGES))]
for _k4i, _eidx in enumerate(_K4_EDGES):
    for _e in _eidx:
        _EDGE_TO_K4S[_e].append(_k4i)


def _AFFECTED_K4S_FOR_EDGE_UNION(edge_indices):
    """Union of K4 indices affected by the given edge indices."""
    seen = set()
    for e in edge_indices:
        for k4i in _EDGE_TO_K4S[e]:
            seen.add(k4i)
    return seen


def _local_search(colour, rng, time_limit):
    """
    Local search: repeatedly try to merge a colour into another.
    Prefer merging the least-used colour into a random other colour.
    """
    start_time = time.time()
    best = list(colour)
    best_count = _count_colours_used(best)

    while time.time() - start_time < time_limit:
        # Count usage of each colour
        counts = {}
        for c in best:
            counts[c] = counts.get(c, 0) + 1
        colours_present = sorted(counts.keys())
        if len(colours_present) <= 1:
            break

        # Pick the least-used colour to eliminate
        # Randomize among ties for diversity
        min_count = min(counts.values())
        candidates = [c for c in colours_present if counts[c] == min_count]
        c_from = rng.choice(candidates)

        # Try merging into each other colour, random order
        targets = [c for c in colours_present if c != c_from]
        rng.shuffle(targets)
        merged = False
        for c_to in targets:
            new_colour, ok = _try_merge(best, c_from, c_to, rng)
            if ok:
                best = new_colour
                best_count = _count_colours_used(best)
                merged = True
                break
        if not merged:
            # No single merge works; try a random small perturbation to escape local min
            # Recolour a random edge to a random present colour, then retry
            if rng.random() < 0.3:
                # Perturb: change one edge's colour to a random present colour
                e = rng.randint(0, len(best) - 1)
                old_c = best[e]
                new_c = rng.choice(colours_present)
                if new_c != old_c:
                    best[e] = new_c
                    # If this perturbation broke validity, revert
                    if not _is_valid(best):
                        best[e] = old_c
                    else:
                        # Continue; best_count may have changed
                        best_count = _count_colours_used(best)

    return best, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    rng = random.Random(12345)

    # Start from seed
    colour = _initial_colouring()
    best = list(colour)
    best_count = _count_colours_used(best)

    # Run local search with time budget
    # Use multiple restarts with different seeds if time permits
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Each restart: fresh perturbation of the initial or best solution
        if restarts == 1:
            cur = _initial_colouring()
        else:
            # Perturb the best found so far
            cur = list(best)
            # Randomly swap some colours
            for _ in range(rng.randint(5, 30)):
                e = rng.randint(0, len(cur) - 1)
                old = cur[e]
                # Pick a random colour from 0..best_count-1 or a new one
                new = rng.randint(0, best_count)
                if new != old:
                    cur[e] = new
            if not _is_valid(cur):
                cur = list(best)

        # Local search on cur
        t_left = deadline - time.time() - 5.0
        if t_left < 1.0:
            break
        cur, cnt = _local_search(cur, rng, t_left)
        if cnt < best_count:
            best = cur
            best_count = cnt
        elif cnt == best_count and rng.random() < 0.1:
            # Keep diversity: sometimes accept equal solutions
            best = cur

    # Final validation
    if not _is_valid(best):
        # Fallback to seed
        best = _initial_colouring()
        best_count = _count_colours_used(best)

    # Remap colours to 0..best_count-1 for compactness
    colour_map = {}
    remapped = []
    next_id = 0
    for c in best:
        if c not in colour_map:
            colour_map[c] = next_id
            next_id += 1
        remapped.append(colour_map[c])

    return remapped
# EVOLVE-BLOCK-END
