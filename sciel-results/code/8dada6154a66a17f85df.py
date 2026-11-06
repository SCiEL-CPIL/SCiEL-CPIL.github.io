"""Traveling salesman constructive heuristic — beam-search variant."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150 s wall-clock kill

# Beam width: how many partial tours we keep alive.
# A beam of 4 is a large jump in search breadth over the parent's greedy path,
# but still fast enough to finish well under the deadline for 50-city instances.
_BEAM = 4


def _tour_closed_length(tour, dm, destination_node):
    """Total length of the partial tour plus the closing edge to destination."""
    if tour.size == 0:
        return 0.0
    s = float(np.sum(dm[tour[:-1], tour[1:]]))
    s += float(dm[tour[-1], destination_node])
    return s


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])

    # Deadline guard: fall back to nearest-neighbour if time is running out
    if time.time() > _DEADLINE:
        return int(unv[np.argmin(distance_matrix[current_node, unv])])

    dm = distance_matrix

    # Build a beam of partial tours.  The first beam entry is the degenerate
    # path [current_node]; subsequent entries extend the best beam entry with
    # the best next node under the parent's 2-step heuristic (a cheap way to
    # diversify the beam without a full exhaustive search).
    beam = [np.array([current_node], dtype=np.intp)]
    remaining = set(unv.tolist())
    for _ in range(_BEAM - 1):
        if not remaining:
            break
        last_tour = beam[-1]
        last = int(last_tour[-1])
        avail = np.fromiter([x for x in unv if x in remaining], dtype=np.intp)
        if avail.size == 0:
            break
        d_cur = dm[last, avail]
        if avail.size > 1:
            sub = dm[np.ix_(avail, avail)]
            np.fill_diagonal(sub, np.inf)
            d_onward = sub.min(axis=1)
            scores = d_cur + 0.5 * d_onward
        else:
            scores = d_cur
        nxt = int(avail[int(np.argmin(scores))])
        beam.append(np.append(last_tour, nxt))
        remaining.discard(nxt)

    # For each beam entry, score every possible next unvisited node by the
    # closed-tour length it would produce.  Pick the (beam, node) pair with
    # the smallest score and return that node.
    best_node = None
    best_score = np.inf
    for t in beam:
        if t.size == 0:
            continue
        last = int(t[-1])
        already_in = set(t.tolist())
        avail = np.fromiter([x for x in unv if x not in already_in], dtype=np.intp)
        if avail.size == 0:
            continue
        base = float(np.sum(dm[t[:-1], t[1:]])) if t.size > 1 else 0.0
        # candidate closed-tour length = base + d(last, cand) + d(cand, dest)
        cand_scores = base + dm[last, avail] + dm[avail, destination_node]
        min_idx = int(np.argmin(cand_scores))
        if cand_scores[min_idx] < best_score:
            best_score = cand_scores[min_idx]
            best_node = int(avail[min_idx])

    if best_node is None:
        # Fallback: nearest neighbour
        return int(unv[np.argmin(dm[current_node, unv])])
    return best_node
# EVOLVE-BLOCK-END
