"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy:
    1. Local Prize Density: Compute a score S_j for each node j based on the
       sum of prizes of its neighbors, weighted by a Gaussian kernel. This
       encourages ants to visit clusters of high-value nodes.
    2. Cost-Aware Attractiveness: The attractiveness of moving from i to j is
       proportional to S_j / (d_ij + d_j0). The term d_j0 accounts for the
       cost of returning to the start, which is critical for respecting the
       max_len budget.
    3. Feasibility Mask: Nodes that are too far from the start to be visited
       and returned from within max_len (2 * d_j0 > max_len) are heavily
       penalized.
    """
    start_time = time.time()
    # Safety guard: if computation takes too long, fallback to simple heuristic
    if time.time() - start_time > 290.0:
        return (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)

    n = len(prizes)
    if n == 0:
        return np.array([])

    # 1. Compute local prize density S
    # beta controls the radius of the neighborhood. 
    # With 50 points in unit square, avg dist ~ 0.5. beta=5 gives radius ~ 0.2.
    beta = 5.0
    kernel = np.exp(-beta * distance_matrix)
    # S_j = sum_i prizes[i] * exp(-beta * d_ij)
    S = prizes @ kernel

    # 2. Compute denominator: d_ij + d_j0
    # d_j0 is the distance from each node j to the start node 0.
    d_j0 = distance_matrix[:, 0]
    # Add d_j0 to each column j of distance_matrix
    denom = distance_matrix + d_j0[None, :]
    
    # 3. Compute base heuristic
    H = S[None, :] / (denom + 1e-9)

    # 4. Apply feasibility mask
    # If 2 * d_j0 > max_len, the node is infeasible for a direct round trip.
    # We set these to a very small value to discourage selection.
    mask = (2.0 * d_j0 <= max_len).astype(float)
    H = H * mask[None, :]

    # 5. Ensure all values are at least 1e-9
    H = np.maximum(H, 1e-9)

    return H
# EVOLVE-BLOCK-END
