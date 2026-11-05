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

    # Compute the Laplacian: L = D - A
    # D is diagonal with degrees
    L = np.diag(degree) - adj

    # Compute eigenvalues and eigenvectors of the Laplacian
    # We want the Fiedler vector (second smallest eigenvalue's eigenvector)
    # For efficiency, use eigh which works for symmetric matrices
    try:
        if time.time() < deadline:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Sort by eigenvalue (eigh already returns sorted for symmetric)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # eigvecs[:, 1] is the Fiedler vector
            fiedler = eigvecs[:, 1]
            fiedler_abs = np.abs(fiedler)
        else:
            # Fallback: use degree-based scoring
            return -degree.astype(float)
    except Exception:
        return -degree.astype(float)

    # Normalize fiedler values to [0, 1] range for better combination
    fmin = fiedler_abs.min()
    fmax = fiedler_abs.max()
    if fmax > fmin:
        fiedler_norm = (fiedler_abs - fmin) / (fmax - fmin)
    else:
        fiedler_norm = np.zeros_like(fiedler_abs)

    # Compute neighbor degree sum: sum of degrees of neighbors
    # This measures how much "capacity" we free up by removing neighbors
    # Picking a vertex with high neighbor degree sum is beneficial
    neighbor_degree_sum = adj @ degree

    # Normalize degree to [0, 1]
    dmin = degree.min()
    dmax = degree.max()
    if dmax > dmin:
        degree_norm = (degree - dmin) / (dmax - dmin)
    else:
        degree_norm = np.zeros_like(degree)

    # Normalize neighbor degree sum to [0, 1]
    nds_min = neighbor_degree_sum.min()
    nds_max = neighbor_degree_sum.max()
    if nds_max > nds_min:
        nds_norm = (neighbor_degree_sum - nds_min) / (nds_max - nds_min)
    else:
        nds_norm = np.zeros_like(neighbor_degree_sum)

    # Composite score:
    # 1. Low degree: we want to pick low degree vertices (negative weight)
    # 2. High neighbor degree sum: we want to pick vertices that remove high-degree neighbors (positive weight)
    # 3. High Fiedler: peripheral vertices are safer (positive weight)
    
    # Weights tuned for Erdos-Renyi n=200
    # The "efficiency" of picking a node is roughly (neighbors removed) / (nodes lost)
    # Here we approximate value of removing neighbors vs cost of node itself.
    
    w_deg = 1.0
    w_nds = 0.8
    w_fiedler = 0.5

    score = -w_deg * degree_norm + w_nds * nds_norm + w_fiedler * fiedler_norm

    return score
# EVOLVE-BLOCK-END
