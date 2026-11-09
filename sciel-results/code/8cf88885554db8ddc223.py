# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]

    # Precompute edge masks
    # cw_mask[d, e] is 1 if demand d uses edge e when routed clockwise
    # ccw_mask[d, e] is 1 if demand d uses edge e when routed counter-clockwise
    cw_masks = np.zeros((D, n), dtype=np.int8)
    ccw_masks = np.zeros((D, n), dtype=np.int8)
    weights = np.zeros(D)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            cw_masks[d, :] = 0
            ccw_masks[d, :] = 1
        else:
            if s < t:
                cw_mask = np.zeros(n, dtype=np.int8)
                cw_mask[s:t] = 1
            else:
                cw_mask = np.zeros(n, dtype=np.int8)
                cw_mask[s:] = 1
                cw_mask[:t] = 1
            cw_masks[d] = cw_mask
            ccw_masks[d] = 1 - cw_mask

    # ILP Formulation
    # Variables: x_0 ... x_{D-1} (binary), L (continuous)
    # Total variables: D + 1
    # Minimize L
    
    # Constraints for each edge e:
    # Load_e = sum_d w_d * (cw_mask[d,e] * (1-x_d) + ccw_mask[d,e] * x_d) <= L
    # Load_e = sum_d w_d * cw_mask[d,e] + sum_d w_d * (ccw_mask[d,e] - cw_mask[d,e]) * x_d <= L
    # Let B_e = sum_d w_d * cw_mask[d,e]
    # Let A_e,d = w_d * (ccw_mask[d,e] - cw_mask[d,e])
    # B_e + sum_d A_e,d * x_d - L <= 0
    # sum_d A_e,d * x_d - 1 * L <= -B_e
    
    # Matrix A for LinearConstraint:
    # Rows: n (one per edge)
    # Columns: D + 1 (x_0...x_{D-1}, L)
    
    A_mat = np.zeros((n, D + 1))
    b_vec = np.zeros(n)
    
    for e in range(n):
        # Coefficients for x_d
        A_mat[e, :D] = weights * (ccw_masks[:, e].astype(np.float64) - cw_masks[:, e].astype(np.float64))
        # Coefficient for L
        A_mat[e, D] = -1.0
        # RHS
        b_vec[e] = -np.sum(weights * cw_masks[:, e].astype(np.float64))
        
    # Objective: Minimize L
    # c vector: 0 for x, 1 for L
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Bounds
    # x_i in [0, 1]
    # L in [0, inf)
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    
    # Integrality
    # x_i are integers, L is continuous
    integrality = np.zeros(D + 1, dtype=np.int8)
    integrality[:D] = 1
    
    # Solve
    start_time = time.time()
    deadline = start_time + 580
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A_mat, -np.inf, b_vec),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0: # Optimal
            x_opt = res.x[:D]
            # Map to 0/1 routing
            # x > 0.5 means CCW (1), else CW (0)
            routing = (x_opt > 0.5).astype(np.float64)
            return routing
        else:
            # If ILP fails, fallback to exhaustive search (safe)
            pass
    except Exception:
        pass

    # Fallback: Exhaustive Search
    # This ensures we always return a valid result even if scipy fails
    base = np.zeros(n)
    for d in range(D):
        base += weights[d] * cw_masks[d]
        
    diff_mask = ccw_masks.astype(np.float64) - cw_masks.astype(np.float64)
    weighted_diff = weights[:, None] * diff_mask
    
    best_load = np.inf
    best_routing = np.zeros(D)
    
    for mask in range(1 << D):
        if time.time() > deadline:
            break
        dir_vec = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        loads = base + dir_vec @ weighted_diff
        max_load = np.max(loads)
        if max_load < best_load:
            best_load = max_load
            best_routing = dir_vec
            
    return best_routing

# EVOLVE-BLOCK-END
