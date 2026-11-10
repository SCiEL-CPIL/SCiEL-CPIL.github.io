"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix using a structural change:
    a budget-aware greedy lookahead estimate of marginal prize gain for each (i, j),
    blended with a return-cost penalty.

    The heuristic is computed once per instance and is intended to guide ACO toward
    moves that are both feasible under the tour-length constraint and likely to
    increase total collected prize.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin under 330s wall-clock limit

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    # Precompute distances to depot
    d0 = distance_matrix[:, 0]

    # Feasibility of taking edge i -> j and then returning to depot immediately:
    # ACO feasibility is len_so_far + d[i,j] + d[j,0] <= max_len.
    # If d[i,j] + d[j,0] > max_len, the edge is never feasible.
    edge_cost = distance_matrix + d0[None, :]
    feasible = edge_cost <= (max_len + 1e-9)

    # Base value: prize at destination
    prize_gain = prizes[None, :]

    # Greedy lookahead: after moving i -> j, estimate additional prize collectable
    # within the remaining budget before returning to depot.
    # Remaining budget after i->j and final return is:
    #   rem = max_len - (d[i,j] + d[j,0])
    # From j, we can visit a sequence of nodes k1, k2, ... such that
    #   d[j,k1] + d[k1,k2] + ... + d[km,0] <= rem
    # We approximate this with a short greedy chain (up to 3 extra nodes) to keep cost low.

    # Initialize lookahead gain (excluding the immediate prize at j)
    lookahead_gain = np.zeros((n, n), dtype=np.float64)

    # To keep runtime bounded, we compute lookahead only for feasible edges and
    # limit the number of candidate next nodes per step.
    # We use a simple vectorized-ish loop over current node i, but inner steps are
    # limited and we break early if time approaches deadline.

    # Precompute sorted candidate order per node by prize / distance-to-depot
    # This gives a cheap proxy for good next stops when budget is tight.
    # For each current node j, we want to greedily pick next node k maximizing
    # prize[k] / (d[j,k] + d[k,0]) while maintaining feasibility.
    # We'll compute a small lookup of top candidates per node to speed up.

    # For each node j, compute candidate scores for all k != j
    # score[j,k] = prize[k] / (d[j,k] + d[k,0]) if feasible else 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cand_score = np.where(
            feasible,
            prize_gain / np.maximum(edge_cost, 1e-12),
            0.0
        )
    # Do not allow self as next
    np.fill_diagonal(cand_score, 0.0)

    # For each node j, get top 8 candidate next nodes by score
    # This limits inner loop work.
    top_k = min(8, n)
    # argsort in descending order
    # We'll store top indices per node
    top_indices = np.argsort(-cand_score, axis=1)[:, :top_k]

    # Now compute lookahead for each feasible (i,j)
    # We'll iterate over i rows to avoid huge temporaries
    for i in range(n):
        # For each j, if edge i->j feasible, compute greedy chain from j
        # We can vectorize over j partially, but keep it simple for now.
        for j in range(n):
            if not feasible[i, j]:
                continue
            rem = max_len - (distance_matrix[i, j] + d0[j])
            if rem < 0:
                continue
            # Greedy chain from j
            cur = j
            gain = 0.0
            # Up to 3 extra nodes
            for _ in range(3):
                # Check time
                if time.time() > deadline:
                    break
                # Get top candidates from cur
                cands = top_indices[cur]
                best_k = -1
                best_val = 0.0
                best_cost = 0.0
                for k in cands:
                    if k == cur:
                        continue
                    # Check feasibility: d[cur,k] + d[k,0] <= rem
                    step_cost = distance_matrix[cur, k] + d0[k]
                    if step_cost > rem + 1e-9:
                        continue
                    val = prizes[k]
                    if val > best_val:
                        best_val = val
                        best_k = k
                        best_cost = step_cost
                if best_k < 0:
                    break
                gain += best_val
                rem -= best_cost
                cur = best_k
            lookahead_gain[i, j] = gain

    # If we hit deadline, we still have partial lookahead (zeros for uncomputed).
    # That's acceptable; we'll still produce a valid heuristic.

    # Combine immediate prize and lookahead
    total_gain = prize_gain + lookahead_gain

    # Penalize edges with high return cost relative to max_len
    # Use a soft penalty: 1 / (1 + (edge_cost / max_len)^2)
    # This keeps values positive and bounded.
    with np.errstate(divide='ignore', invalid='ignore'):
        cost_ratio = edge_cost / max(max_len, 1e-12)
        penalty = 1.0 / (1.0 + cost_ratio ** 2)

    # Heuristic value: total_gain * penalty
    heuristic = total_gain * penalty

    # Apply feasibility mask
    heuristic[~feasible] = 0.0
    np.fill_diagonal(heuristic, 0.0)

    # Normalize to [0, 1]
    max_h = heuristic.max()
    if max_h > 0:
        heuristic /= max_h
    else:
        heuristic[:] = 1e-9

    # Ensure minimum value
    heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
