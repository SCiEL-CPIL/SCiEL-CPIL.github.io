"""Structural change: Or-opt-style segment insertion heuristic for constructive TSP.
Instead of picking the nearest (or 2-step lookahead) next node, we consider inserting
the current 'tail' segment [current_node -> best continuation] at various positions.
Here we implement a practical variant:
- Compute the 2-step lookahead cost for each candidate as before, but
- Additionally compute a 'segment removal' regret: how much would the tour cost
  increase if we removed the best next edge?
- Use a combined score that penalizes candidates whose insertion would create
  a long 'detour' relative to the local structure.

A more structural approach: use a 3-lookahead with geometric angle awareness.
For each candidate, compute the angle between (current->candidate) and
(candidate->best_onward). Prefer small angles (smoother turns) as a proxy for
shorter tours.
"""
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node using a 3-step lookahead with geometric smoothness.
    
    For each candidate c in unvisited_nodes:
      1. d1 = dist(current, c)
      2. c2 = argmin_{v in unvisited \ {c}} dist(c, v)   (best onward from c)
      3. d2 = dist(c, c2)
      4. c3 = argmin_{v in unvisited \ {c, c2}} dist(c2, v)  (best onward from c2)
      5. d3 = dist(c2, c3)
      6. total_3step = d1 + d2 + d3
      7. angle_score = angle between vectors (c - current) and (c2 - c)
      8. score = total_3step + lambda * angle_penalty
    
    We also add a small destination bias to avoid getting stuck far from the
    return point.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    if n == 2:
        # Pick the one that minimizes dist(current, x) + dist(x, destination)
        dm = distance_matrix
        costs = dm[current_node, unv] + dm[unv, destination_node]
        return int(unv[int(np.argmin(costs))])

    dm = distance_matrix

    # Precompute distances from current to all unvisited
    d_cur = dm[current_node, unv]

    # Submatrix among unvisited nodes
    sub = dm[np.ix_(unv, unv)].copy()
    np.fill_diagonal(sub, np.inf)

    # For each candidate c (row in sub), find best onward (min in row)
    # onward_idx[c] = index in unv of best next after c
    onward_cost = sub.min(axis=1)
    onward_idx = sub.argmin(axis=1)

    # For 3-step: for each c, after going to onward_idx[c], find best onward from that
    # But we need to exclude c itself from the second onward
    # Build a copy of sub with c's row/col handled
    # For efficiency, compute for each c:
    #   c2 = onward_idx[c]
    #   Among unvisited \ {c, c2}, find min dist(c2, v)
    # We can do this by setting sub[c, c] = inf (already done) and sub[c2, c] = inf
    # but since we're working with a submatrix, let's handle it differently.

    # Create a full copy to modify per-candidate (expensive for large n, but n<=50)
    # Actually, let's vectorize: for each c, the second onward is min of sub[onward_idx[c], :]
    # excluding c and onward_idx[c] itself.
    # Let's create a mask-based approach.

    # For each c in 0..n-1:
    #   c2 = onward_idx[c]
    #   We want min over j != c, j != c2 of sub[c2, j]
    # Set up a temporary sub2 = sub.copy()
    # For each c, set sub2[c, c] = inf (already inf), sub2[c, onward_idx[c]] = inf
    # Then second_onward_cost[c] = sub2[onward_idx[c], :].min()
    # This is O(n^2) which is fine for n<=50.

    sub2 = sub.copy()
    for c in range(n):
        c2 = onward_idx[c]
        if c2 != c:
            sub2[c, c2] = np.inf
            sub2[c2, c] = np.inf

    # Now for each c, the best second onward from c2 (excluding c and c2)
    # We need to exclude c2 itself from sub2[c2, :]
    sub2_no_self = sub2.copy()
    for c2 in range(n):
        sub2_no_self[c2, c2] = np.inf

    # second_onward_cost[c] = sub2_no_self[onward_idx[c], :].min()
    second_onward_cost = sub2_no_self[onward_idx, :].min(axis=1)
    second_onward_idx = sub2_no_self[onward_idx, :].argmin(axis=1)

    # Total 3-step cost
    three_step = d_cur + onward_cost + second_onward_cost

    # Geometric smoothness: compute angle between (c - current) and (c2 - c)
    # We don't have coordinates, only distance matrix. So we use a proxy:
    # If dist(current, c2) is close to |d1 - d2| or d1+d2, the points are collinear.
    # Small angle => dist(current, c2) ≈ |d1 - d2| (if same direction) or d1+d2 (if opposite)
    # Actually, by law of cosines: dist(current,c2)^2 = d1^2 + d2^2 - 2*d1*d2*cos(angle)
    # cos(angle) = (d1^2 + d2^2 - dist(current,c2)^2) / (2*d1*d2)
    # Small angle => cos close to 1 => dist(current,c2) close to |d1-d2|
    # Large angle (180 deg) => cos close to -1 => dist(current,c2) close to d1+d2
    # We want to penalize large angles (sharp turns).
    # angle_penalty = max(0, dist(current,c2) - (d1+d2)) ... no, that's always <= 0 by triangle inequality.
    # Let's use: penalty = (d1 + d2 - dist(current, c2))  ... this is small when collinear same direction
    # Actually, we want to reward small angles. Small angle means the path is straight.
    # A good proxy: the "detour" = d1 + d2 - dist(current, c2). This is >= 0 by triangle inequality.
    # Small detour => nearly collinear (straight line) => good.
    # Large detour => sharp turn => bad.
    d_cur_c2 = dm[current_node, unv[onward_idx]]
    detour = d_cur + onward_cost - d_cur_c2
    # Normalize detour
    detour_span = detour.max() - detour.min()
    if detour_span > 1e-12:
        detour_norm = (detour - detour.min()) / detour_span
    else:
        detour_norm = np.zeros_like(detour)

    # Destination awareness
    d_dest = dm[unv, destination_node]
    d_dest_span = d_dest.max() - d_dest.min()
    if d_dest_span > 1e-12:
        d_dest_norm = (d_dest - d_dest.min()) / d_dest_span
    else:
        d_dest_norm = np.zeros_like(d_dest)

    # Normalize 3-step
    ts_span = three_step.max() - three_step.min()
    if ts_span > 1e-12:
        ts_norm = (three_step - three_step.min()) / ts_span
    else:
        ts_norm = np.zeros_like(three_step)

    # Combined score: lower is better
    # Weights: 0.60 for 3-step, 0.25 for detour, 0.15 for destination
    score = 0.60 * ts_norm + 0.25 * detour_norm + 0.15 * d_dest_norm

    idx = int(np.argmin(score))
    return int(unv[idx])


# Alias for backward compatibility if needed
_select_blended = select_next_node
