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

    # Compute neighbor-degree-sum (opportunity cost signal)
    # For each vertex i, sum of degrees of its neighbors
    neighbor_degree_sum = adj @ degree

    # Compute the Laplacian: L = D - A
    L = np.diag(degree) - adj

    # Compute eigenvalues and eigenvectors of the Laplacian
    # We want the Fiedler vector (second smallest eigenvalue's eigenvector)
    try:
        if time.time() < deadline:
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
            fiedler_abs = np.abs(fiedler)
        else:
            # Fallback: use degree-based scoring
            return -degree.astype(float)
    except Exception:
        return -degree.astype(float)

    # Normalize fiedler values to [0, 1] range
    fmin = fiedler_abs.min()
    fmax = fiedler_abs.max()
    if fmax > fmin:
        fiedler_norm = (fiedler_abs - fmin) / (fmax - fmin)
    else:
        fiedler_norm = np.zeros_like(fiedler_abs)

    # Normalize degree to [0, 1]
    dmin = degree.min()
    dmax = degree.max()
    if dmax > dmin:
        degree_norm = (degree - dmin) / (dmax - dmin)
    else:
        degree_norm = np.zeros_like(degree)

    # Normalize neighbor-degree-sum to [0, 1]
    nds_min = neighbor_degree_sum.min()
    nds_max = neighbor_degree_sum.max()
    if nds_max > nds_min:
        nds_norm = (neighbor_degree_sum - nds_min) / (nds_max - nds_min)
    else:
        nds_norm = np.zeros_like(neighbor_degree_sum)

    # Composite score:
    # We want to select vertices that are:
    # 1. Low degree (easier to keep independent) -> low degree_norm
    # 2. High Fiedler value (peripheral, less entangled) -> high fiedler_norm
    # 3. Low neighbor-degree-sum (selecting it removes low-value neighbors) -> low nds_norm
    
    # Weighted combination:
    # score = -w1 * degree_norm + w2 * fiedler_norm - w3 * nds_norm
    w1 = 1.0
    w2 = 0.8
    w3 = 0.5

    score = -w1 * degree_norm + w2 * fiedler_norm - w3 * nds_norm

    return score
# EVOLVE-BLOCK-END
