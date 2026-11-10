# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy:
    - Blend multiple prize/distance exponent combinations to create sharp local attraction.
    - Cost[i,j] = d[i,j] + d[j,0] (step + return to origin).
    - Variants tested: prize/cost^3, prize^2/cost^3, prize/cost^4, prize^2/cost^4.
    - Normalize each variant and blend with tuned weights.
    - Add a small origin-proximity bonus to stabilize early moves.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Safety check for finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9
    
    # cost[i, j] = d[i, j] + d[j, 0]
    cost = distance_matrix + distance_matrix[:, 0][None, :]
    
    # prize[j] broadcast over rows
    prize_col = prizes[None, :]  # shape (1, n)
    
    # Ensure cost is positive to avoid division by zero
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # Ensure prize is non-negative for power operations
    prize_safe = np.maximum(prize_col, eps)
    
    # Compute multiple heuristic variants with different prize/distance exponents
    # V1: prize / cost^3  (current baseline)
    H1 = prize_safe / (cost_safe ** 3)
    
    # V2: prize^2 / cost^3  (sharper prize preference)
    H2 = (prize_safe ** 2) / (cost_safe ** 3)
    
    # V3: prize / cost^4  (stronger distance penalty)
    H3 = prize_safe / (cost_safe ** 4)
    
    # V4: prize^2 / cost^4  (both sharper)
    H4 = (prize_safe ** 2) / (cost_safe ** 4)
    
    # Check time
    if time.time() > deadline - 1.0:
        H = H1
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Normalize each variant to [0, 1] range for blending
    def normalize(H_mat):
        h = np.where(np.isfinite(H_mat) & (H_mat > 0), H_mat, 0.0)
        max_val = np.max(h)
        if max_val > 0:
            return h / max_val
        return h
    
    H1_n = normalize(H1)
    H2_n = normalize(H2)
    H3_n = normalize(H3)
    H4_n = normalize(H4)
    
    # Blend with weights tuned for sharp local attraction
    # Heavier weight on higher-exponent variants for contrast
    w1, w2, w3, w4 = 0.15, 0.25, 0.30, 0.30
    H = w1 * H1_n + w2 * H2_n + w3 * H3_n + w4 * H4_n
    
    # Refinement: Add a small bonus for nodes that are close to the origin
    dist_to_origin = distance_matrix[:, 0]  # shape (n,)
    origin_bonus = 1.0 / (1.0 + dist_to_origin[None, :])  # shape (1, n)
    
    # Combine: H = H * (1 + 0.15 * origin_bonus)
    H = H * (1.0 + 0.15 * origin_bonus)
    
    # Check time
    if time.time() > deadline - 0.5:
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Ensure no negative or non-finite values
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Final safety: ensure shape is correct
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9
    
    return H
# EVOLVE-BLOCK-END
