"""Spectral relaxation followed by vectorized local search for Maximum Cut."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to float64 for linear algebra operations
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Compute degrees
    degrees = A.sum(axis=1)
    
    # Construct Graph Laplacian L = D - A
    L = np.diag(degrees) - A
    
    # Compute leading eigenvector using power iteration
    # Start with a random vector
    rng = np.random.default_rng(42)
    v = rng.normal(size=n)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        v = np.ones(n)
    else:
        v = v / v_norm
        
    # Power iteration for L
    for _ in range(100):
        v = L @ v
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            break
        v = v / v_norm
        
    # Assign labels based on the sign of the eigenvector components
    # We need to ensure a balanced partition if possible, but sign is a good start.
    # If all signs are the same, it's a degenerate case, fallback to alternating.
    labels = (v >= 0).astype(int)
    
    # Check if partition is trivial (all 0s or all 1s)
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.arange(n) % 2

    # Local improvement phase
    # We will iterate until no improvement or time limit.
    # Deadline: 145 seconds (5s margin from 150s limit)
    deadline = time.time() + 145.0
    
    improved = True
    # Pre-compute row sums for quick gain calculation?
    # Gain of flipping v:
    # Let S_v be sum of neighbors in same set.
    # Let O_v be sum of neighbors in other set.
    # Total degree d_v = S_v + O_v.
    # Current contribution of v to cut: O_v.
    # New contribution if flipped: S_v.
    # Change: S_v - O_v = S_v - (d_v - S_v) = 2*S_v - d_v.
    # We flip if Change > 0 => 2*S_v > d_v.
    
    while improved and time.time() < deadline:
        improved = False
        # Vectorized computation of gains for all vertices
        # labels is 0/1 array.
        # For each v, we need sum of A[v, labels == labels[v]]
        
        # Create a mask of same labels
        # same_mask[i, j] is True if labels[i] == labels[j]
        # We can compute S_v for all v at once:
        # S_v = sum_j A[v, j] * (labels[i] == labels[j])
        
        # Efficient way:
        # Let L0 = labels == 0, L1 = labels == 1
        # For v with label 0: S_v = A[v, :].dot(L0)
        # For v with label 1: S_v = A[v, :].dot(L1)
        
        L0 = (labels == 0)
        L1 = (labels == 1)
        
        # Compute S for all vertices
        # S = A.dot(L0) for label 0, A.dot(L1) for label 1
        # We can compute both and select
        S0 = A.dot(L0)
        S1 = A.dot(L1)
        
        # S_v for vertex v
        # If label is 0, use S0[v], else S1[v]
        S_v = np.where(labels == 0, S0, S1)
        
        # Degrees
        d_v = degrees
        
        # Gain = 2 * S_v - d_v
        # Flip if Gain > 0
        to_flip = (2 * S_v > d_v)
        
        if np.any(to_flip):
            labels[to_flip] = 1 - labels[to_flip]
            improved = True

    return labels
# EVOLVE-BLOCK-END
