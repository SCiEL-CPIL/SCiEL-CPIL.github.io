"""Kissing number in dimension 11: Maximum Independent Set on norm-4 conflict graph."""
D = 11

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _build_candidates():
    """Build all nonzero integer vectors with norm^2 = 4."""
    cands = []

    # Type 1: weight-4 vectors in {-1, 0, 1}^11 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            cands.append(v)

    # Type 2: weight-1 vectors with entry ±2 (norm^2 = 4)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cands.append(v)

    return np.array(cands, dtype=np.int64)


def _greedy_mis(conflicts, adj, n, rng, deadline):
    """Greedy maximum independent set with min-degree selection."""
    alive = np.ones(n, dtype=bool)
    selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)

    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break

        # Pick vertex with minimum degree among alive
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]

        selected.append(chosen)

        # Remove chosen and its neighbors
        neighbors = adj[chosen]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False

        # Update degrees for remaining alive vertices
        rem = np.where(alive)[0]
        if len(rem) > 0:
            # For each remaining vertex, subtract number of neighbors that were removed
            # to_remove includes chosen and its neighbors
            # We need: for each r in rem, count how many of to_remove are neighbors of r
            # That's: conflicts[rem][:, to_remove].sum(axis=1)
            if len(to_remove) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

    return selected


def _local_improve(selected, conflicts, adj, n, rng, deadline):
    """Try to improve the independent set by swapping out vertices for better ones."""
    selected_set = set(selected)
    alive = np.ones(n, dtype=bool)
    for idx in selected:
        alive[idx] = False
        # Also mark neighbors as not available for swap-in
        for nb in adj[idx]:
            alive[nb] = False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # For each selected vertex, try to find a non-selected vertex that can replace it
        # A replacement v is valid if:
        #   - v is not in selected_set
        #   - v has no conflicts with any other selected vertex (other than the one being removed)
        for i, sel_idx in enumerate(list(selected)):
            # Temporarily remove sel_idx
            selected_set.remove(sel_idx)
            # Find candidates: alive vertices that don't conflict with remaining selected
            remaining_selected = list(selected_set)
            if remaining_selected:
                # A candidate v is valid if it doesn't conflict with any remaining selected
                # conflicts[remaining_selected, v] should all be False
                # Check: for each r in remaining_selected, conflicts[r, v] == False
                # This is equivalent to: v is not in the union of neighbors of remaining selected
                # Let's compute which vertices conflict with any remaining selected
                conflict_mask = np.zeros(n, dtype=bool)
                for r in remaining_selected:
                    conflict_mask |= conflicts[r]
                # Candidates are alive vertices not in conflict_mask and not in selected_set
                candidate_mask = alive & ~conflict_mask
                # Also exclude vertices that are in selected_set (already excluded from alive)
                candidates = np.where(candidate_mask)[0]
            else:
                candidates = np.where(alive)[0]

            if len(candidates) > 0:
                # Pick a random candidate to try swapping
                # Prefer candidates with low degree (more "flexible")
                degs = conflicts[candidates].sum(axis=1)
                min_d = np.min(degs)
                good_cands = candidates[degs == min_d]
                new_idx = good_cands[rng.integers(len(good_cands))]

                # Perform the swap
                selected[i] = new_idx
                selected_set.add(new_idx)
                alive[new_idx] = False
                for nb in adj[new_idx]:
                    alive[nb] = False

                # Re-mark neighbors of old sel_idx as alive (they might be usable now)
                for nb in adj[sel_idx]:
                    if nb not in selected_set:
                        alive[nb] = True

                improved = True

            # Restore sel_idx if we didn't swap
            if sel_idx not in selected_set:
                selected_set.add(sel_idx)

    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    candidates = _build_candidates()
    n = len(candidates)

    # Compute dot products: conflict if dot > 2
    dots = candidates @ candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Build adjacency lists
    adj = [np.where(conflicts[i])[0] for i in range(n)]

    # Run greedy MIS with multiple random seeds and keep the best
    best_selected = None
    best_size = 0

    # First, run a few greedy passes with different random seeds
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        rng = np.random.default_rng(seed)
        selected = _greedy_mis(conflicts, adj, n, rng, deadline - 20)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected

    if best_selected is None:
        # Fallback: return all weight-1 vectors
        v1 = []
        for i in range(D):
            for s in (2, -2):
                v = np.zeros(D, dtype=np.int64)
                v[i] = s
                v1.append(v)
        return np.array(v1, dtype=np.int64)

    # Try local improvement on the best solution
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        improved = _local_improve(list(best_selected), conflicts, adj, n, rng, deadline - 5)
        if len(improved) > best_size:
            best_selected = improved
            best_size = len(improved)

    return candidates[best_selected]
# EVOLVE-BLOCK-END
