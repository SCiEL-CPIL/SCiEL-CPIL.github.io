"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    # Ensure adjacency is a numpy array
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Compute degree
    degree = adj.sum(axis=1)

    # Compute inverse degree of neighbors: sum_{j in N(i)} 1/(d_j + 1)
    # This measures the "independence value" gained by selecting i,
    # as it removes neighbors from contention, weighted by how cheap they are to exclude.
    inv_degree = 1.0 / (degree + 1.0)
    neighbor_inv_degree_sum = adj @ inv_degree

    # Normalize degree to [0, 1]
    dmin = degree.min()
    dmax = degree.max()
    if dmax > dmin:
        degree_norm = (degree - dmin) / (dmax - dmin)
    else:
        degree_norm = np.zeros_like(degree)

    # Normalize neighbor inverse degree sum to [0, 1]
    nid_min = neighbor_inv_degree_sum.min()
    nid_max = neighbor_inv_degree_sum.max()
    if nid_max > nid_min:
        nid_norm = (neighbor_inv_degree_sum - nid_min) / (nid_max - nid_min)
    else:
        nid_norm = np.zeros_like(neighbor_inv_degree_sum)

    # Composite score:
    # We want to select vertices that are:
    # 1. Low degree (easier to keep independent) -> low degree_norm
    # 2. High neighbor inverse degree sum (removes many low-cost neighbors, 
    #    which have high relative independence value) -> high nid_norm
    
    # Weighted combination:
    # score = -w1 * degree_norm + w2 * nid_norm
    # Primary: Low Degree. Secondary: High neighbor inverse degree sum.
    w1 = 1.0
    w2 = 0.8

    score = -w1 * degree_norm + w2 * nid_norm

    return score
# EVOLVE-BLOCK-END
