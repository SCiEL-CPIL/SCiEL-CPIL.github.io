# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=float)

    # Degrees
    deg = A.sum(axis=1)

    # Mean degree of neighbors
    neighbor_deg_sum = A @ deg
    safe_deg = np.maximum(deg, 1.0)
    mean_neighbor_deg = neighbor_deg_sum / safe_deg

    # A2 = A @ A
    A2 = A @ A

    # 2-step paths (off-diagonal)
    two_step_paths = A2.sum(axis=1) - deg
    mean_two_step = two_step_paths / safe_deg

    # Neighbor clustering: sum of A2[v, u] where A[v, u] = 1
    # This equals sum over neighbors u of (number of common neighbors of v and u)
    # which counts edges within N(v) with multiplicity
    neighbor_clustering = (A * A2).sum(axis=1)
    mean_neighbor_connectivity = neighbor_clustering / safe_deg

    # Variance of neighbor degrees: low variance means neighbors are similar
    # High variance means some neighbors are much more connected than others
    neighbor_deg_var = (A * (deg[None, :] ** 2)).sum(axis=1) / safe_deg - mean_neighbor_deg ** 2
    neighbor_deg_var = np.maximum(neighbor_deg_var, 0.0)

    # Number of common neighbors with other vertices (off-diagonal A2 row sum already computed)
    # For vertex v, how many other vertices share neighbors with v
    # This is related to the "redundancy" of v's neighborhood

    # Score components:
    # 1. -deg: prefer low degree (less blocking of potential MIS members)
    # 2. + mean_neighbor_deg: prefer vertices whose neighbors are high-degree
    #    (blocking high-degree neighbors is "cheaper" since they'd block many anyway)
    # 3. + mean_neighbor_connectivity: prefer vertices whose neighbors are densely connected
    #    (clique-like neighbors: at most 1 can be in MIS, so blocking them costs little)
    # 4. - mean_two_step: prefer vertices with fewer 2-step neighbors
    #    (fewer indirect interactions)
    # 5. - neighbor_deg_var: prefer vertices whose neighbors have similar degrees
    #    (uniform neighborhood structure is more "efficient" to block)

    score = (-1.0 * deg
             + 1.0 * mean_neighbor_deg
             + 0.5 * mean_neighbor_connectivity
             - 0.3 * mean_two_step
             - 0.2 * neighbor_deg_var)

    # Check deadline
    if time.time() > deadline:
        return score.astype(float)

    return score.astype(float)
# EVOLVE-BLOCK-END
