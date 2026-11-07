"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_heuristic(prizes, distance_matrix, max_len):
    """
    Build a heuristic matrix based on marginal prize density per unit of
    remaining path budget, with a geometric prior.
    
    The idea: for each pair (i, j), estimate how much 'slack' remains after
    visiting j from i and returning to 0. Nodes that can be visited with
    more remaining budget (i.e., cheaper to include) get higher heuristic
    weight when their prize is high.
    """
    n = len(prizes)
    d = distance_matrix
    prizes_arr = np.asarray(prizes, dtype=np.float64)
    
    # Slack for going i -> j -> 0: max_len - d[i,j] - d[j,0]
    # This represents remaining budget after this leg.
    # We want to reward nodes j that are "affordable" from i.
    
    # Compute slack matrix: slack[i, j] = max_len - d[i, j] - d[j, 0]
    d_j0 = d[:, 0]  # column 0: distance from each node to 0
    slack = max_len - d - d_j0[None, :]  # slack[i, j]
    
    # Where slack <= 0, the move is infeasible; give very low heuristic
    # But we still need finite positive values.
    
    # Prize density: prize[j] / max(slack, eps) -- but this is tricky.
    # Better: weight by prize[j] * f(slack) where f rewards positive slack.
    
    # Use: heuristic[i,j] = prize[j] * exp(slack / scale) for slack > 0
    # and a small value for slack <= 0.
    
    # Scale: use max_len as reference
    scale = max_len * 0.5
    
    # For j=0 (start node), prize is 0, so heuristic will be near 0.
    # That's fine; ants won't go back to 0 mid-tour typically.
    
    # Compute base heuristic
    # prize[j] broadcast over i
    prize_col = prizes_arr[None, :]  # shape (1, n)
    
    # For feasible moves (slack > 0), use prize * exp(slack/scale)
    # For infeasible, use a tiny value
    feasible = slack > 0
    exp_factor = np.exp(np.clip(slack, 0, max_len) / scale)
    
    heuristic = prize_col * exp_factor
    heuristic[~feasible] = 1e-9
    
    # Add a small geometric prior: prefer closer nodes (lower distance)
    # This helps break ties and guides exploration.
    # Use 1/(1 + d[i,j]) as a closeness bonus.
    closeness = 1.0 / (1.0 + d)
    # Blend: 70% prize-density, 30% closeness
    heuristic = 0.7 * heuristic + 0.3 * closeness * (prize_col + 1e-9)
    
    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


def heuristics(prizes, distance_matrix, max_len):
    """
    Returns an edge-attractiveness matrix (larger is more attractive) of the
    same shape as distance_matrix.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    
    # Deadline guard: if something goes wrong, fall back to a safe heuristic
    t0 = time.time()
    
    try:
        H = _build_heuristic(prizes, distance_matrix, max_len)
        
        # Validate
        if H.shape != distance_matrix.shape:
            raise ValueError("Shape mismatch")
        if not np.all(np.isfinite(H)):
            raise ValueError("Non-finite values in heuristic")
        if np.any(H <= 0):
            H = np.where(H > 0, H, 1e-9)
        
        # Safety margin: if we've used more than 250s, return immediately
        if time.time() - t0 > 250:
            return H
        
        return H
        
    except Exception:
        # Fallback: simple prize/distance heuristic
        d = distance_matrix + 1e-9
        H = (prizes[None, :] + 1e-9) / d
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
# EVOLVE-BLOCK-END
