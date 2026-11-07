"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    # Set diagonal to a large value so 1/d^alpha becomes very small
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    # Use inverse cubic distance for sharper guidance toward short edges
    alpha = 3.0
    eta = 1.0 / (d_safe ** alpha + 1e-30)
    
    # Degree-based penalty: penalize edges to nodes that are already
    # highly connected to many other close nodes, to avoid over-concentration
    # and encourage more balanced tour structure
    # Compute a "popularity" score for each node based on how many close neighbors it has
    # Use a soft-min approach: for each node j, compute sum of exp(-k * d[i,j]) over all i
    # This gives higher weight to closer neighbors
    k = 5.0  # temperature for the soft count
    # d_safe has diagonal set to 1e10, so exp(-k*1e10) ~ 0, which is fine
    popularity = np.sum(np.exp(-k * d_safe), axis=0)  # shape (n,)
    
    # Normalize popularity so that the mean is 1
    pop_mean = np.mean(popularity)
    if pop_mean > 0:
        popularity_norm = popularity / pop_mean
    else:
        popularity_norm = np.ones(n)
    
    # Apply a mild penalty: divide by (1 + beta * (popularity_norm - 1))
    # Nodes with above-average popularity get slightly penalized
    beta = 0.5
    penalty = 1.0 + beta * (popularity_norm - 1.0)
    penalty = np.where(penalty > 0, penalty, 1e-9)
    eta = eta / penalty[np.newaxis, :]
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta
# EVOLVE-BLOCK-END
