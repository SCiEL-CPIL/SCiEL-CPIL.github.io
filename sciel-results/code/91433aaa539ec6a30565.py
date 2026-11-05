"""Program for Maximum Independent Set via parallel discrete-time exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores by simulating a parallel discrete-time exclusion process.
    In each round, vertices with the fewest active neighbors become eligible for selection.
    A maximal independent set is greedily chosen among the eligible set.
    Vertices selected earlier receive higher scores, and vertices excluded by
    neighbors receive scores reflecting the round at which they were blocked.
    This captures dynamic interference patterns rather than static topology.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    adj = adjacency.astype(bool)

    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    assigned = np.zeros(n, dtype=bool)

    round_id = 0
    base_score = float(n) + 1.0

    while active.any():
        # Compute active degree for each vertex (count active neighbors)
        active_int = active.astype(np.int32)
        deg = (adj * active_int).sum(axis=1)

        active_indices = np.where(active & ~assigned)[0]
        if len(active_indices) == 0:
            break

        active_deg = deg[active_indices]
        min_deg = active_deg.min()

        # Eligible: active vertices with minimum degree
        eligible_mask = np.zeros(n, dtype=bool)
        eligible_mask[active_indices[active_deg == min_deg]] = True

        eligible_indices = np.where(eligible_mask)[0]
        if len(eligible_indices) == 0:
            break

        # Tie-breaking: among eligible, prefer those with lower internal degree
        # (fewer neighbors also in the eligible set), so they block fewer candidates
        eligible_set_mask = eligible_mask
        internal_deg = (adj * eligible_set_mask.astype(np.int32)).sum(axis=1)
        eligible_internal = internal_deg[eligible_indices]

        # Sort eligible by internal degree ascending, then by global degree ascending
        sort_keys = np.lexsort((deg[eligible_indices], eligible_internal))
        sorted_eligible = eligible_indices[sort_keys]

        # Greedy maximal independent set on sorted_eligible
        selected = []
        for v in sorted_eligible:
            if not selected:
                selected.append(v)
            else:
                if not adj[v, selected].any():
                    selected.append(v)

        if not selected:
            break

        # Assign scores: earlier rounds get higher scores
        round_score = base_score - round_id

        for v in selected:
            scores[v] = round_score
            assigned[v] = True
            active[v] = False

        # Identify neighbors of selected vertices that are still active
        if selected:
            neighbor_mask = adj[selected, :].any(axis=0)
            # Only consider neighbors that are active and not yet assigned
            neighbor_candidates = np.where(neighbor_mask & active & ~assigned)[0]
            for u in neighbor_candidates:
                # Give a score slightly below the current round's score
                scores[u] = round_score - 0.5
                assigned[u] = True
                active[u] = False

        round_id += 1

    # Any vertices not yet assigned get a minimal score
    unassigned = np.where(~assigned)[0]
    if len(unassigned) > 0:
        for v in unassigned:
            scores[v] = 0.5

    return scores
# EVOLVE-BLOCK-END
