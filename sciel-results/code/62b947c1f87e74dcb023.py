import numpy as np

def priority(adjacency):
    A = np.asarray(adjacency)
    degree = A.sum(axis=1).astype(float)
    # Sum of neighbor degrees
    sum_neigh_deg = A @ degree  # matrix-vector product
    # Mean neighbor degree (avoid div by zero)
    mean_neigh_deg = sum_neigh_deg / np.maximum(degree, 1)
    # Score: prefer low degree, high mean neighbor degree
    score = -degree + mean_neigh_deg
    return score
