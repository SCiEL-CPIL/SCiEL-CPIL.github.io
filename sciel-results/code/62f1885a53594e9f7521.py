"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]

    # Compute degrees
    degrees = adj.sum(axis=1)

    # Compute 2-hop neighborhood density (sum of degrees of neighbors)
    # This penalizes vertices connected to high-degree vertices
    neighbor_degree_sums = adj @ degrees

    # Compute 2-hop counts (number of vertices at distance 2)
    # A2 = adj @ adj gives counts of length-2 paths
    A2 = adj @ adj
    # Sum over rows gives total 2-paths from each vertex
    # Subtract direct neighbors to get only distance-2 vertices
    two_hop_counts = A2.sum(axis=1) - degrees

    # Check time
    if time.time() > deadline - 5:
        # Not enough time, fall back to simple degree
        return -degrees.astype(float)

    # Combine signals:
    # 1. Low degree (primary signal for MIS)
    # 2. Low neighbor degree sum (avoid vertices in dense regions)
    # 3. Low 2-hop connectivity (prefer isolated/sparse regions)
    
    # Normalize each component to [0, 1] range for balanced weighting
    def normalize(x):
        mn, mx = np.min(x), np.max(x)
        if mx - mn > 1e-10:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    deg_norm = normalize(degrees)
    nbs_norm = normalize(neighbor_degree_sums)
    hop_norm = normalize(two_hop_counts)

    # Weighted combination: degree is most important, then neighbor sum, then 2-hop
    # We want to MINIMIZE these, so score = negative weighted sum
    score = -(1.0 * deg_norm + 0.5 * nbs_norm + 0.2 * hop_norm)

    return score.astype(float)
# EVOLVE-BLOCK-END
