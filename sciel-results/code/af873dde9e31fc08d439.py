"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition (0/1 labels)."""
    # cut = sum_{v<u} adj[v,u] * (labels[v] != labels[u])
    # Using vectorized: for each v, count neighbors in opposite set
    # cut = 0.5 * sum_v (adj[v] @ (1 - labels)) where we only count once per edge
    # Actually: sum_v adj[v] @ (1 - labels) counts each crossing edge twice (once from each end)
    # So cut = 0.5 * np.sum(adj @ (1 - labels))
    return 0.5 * np.sum(adj @ (1 - labels))


def _spectral_init(adj, n):
    """
    Compute a spectral embedding inspired by Goemans-Williamson.
    For Max-Cut, the SDP relaxation leads to a matrix X where x_i are unit vectors.
    The leading eigenvectors of the Laplacian (or Laplacian-like matrix) give a good embedding.
    We use the second-smallest eigenvector (Fiedler vector) of the graph Laplacian,
    and also try random hyperplane projections in a low-dimensional spectral space.
    """
    # Build Laplacian: L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    # Get eigenvalues and eigenvectors
    # For n=100, this is fast
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # The Fiedler vector is the second smallest eigenvector
    # Use a few low-frequency eigenvectors for embedding
    k = min(10, n)  # use up to 10 dimensions
    embedding = eigvecs[:, 1:k+1]  # skip the trivial zero eigenvector
    
    # Normalize rows to unit length (like in GW)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding = embedding / norms
    
    return embedding


def _random_hyperplane_partition(embedding, rng):
    """
    Generate a partition by projecting onto a random hyperplane.
    This mimics the Goemans-Williamson rounding step.
    """
    d = embedding.shape[1]
    # Random unit vector in R^d
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    
    # Project and threshold
    projections = embedding @ v
    labels = (projections >= 0).astype(int)
    
    # Ensure both partitions are non-empty (usually they will be)
    if labels.sum() == 0 or labels.sum() == len(labels):
        # Flip some to ensure balance
        labels[len(labels)//2:] = 1 - labels[len(labels)//2:]
    
    return labels


def _greedy_flip_improve(adj, labels, max_rounds=50):
    """
    Local search: repeatedly flip vertices that improve the cut until no improvement.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degree
    deg = adj.sum(axis=1).astype(float)
    
    for round_num in range(max_rounds):
        improved = False
        # Compute for each vertex the gain of flipping
        # gain[v] = 2 * (number of neighbors in opposite set) - deg[v]
        # If gain > 0, flipping improves the cut
        # neighbors_in_opposite[v] = adj[v] @ (1 - labels)
        # neighbors_in_same[v] = adj[v] @ labels
        # gain[v] = 2 * (deg[v] - neighbors_in_same[v]) - deg[v] = deg[v] - 2 * neighbors_in_same[v]
        # Wait: if we flip v, edges to same become crossing, edges to opposite become non-crossing.
        # Current contribution of v to cut: adj[v] @ (1 - labels[v]) ... no
        # Let's think carefully.
        # Cut size = 0.5 * sum_v adj[v] @ (1 - labels)
        # If we flip v (labels[v] -> 1 - labels[v]):
        # The term for v changes from adj[v] @ (1 - labels[v]) to adj[v] @ (1 - (1 - labels[v])) = adj[v] @ labels[v]
        # But also for each neighbor u, the term adj[u] @ (1 - labels) changes because labels[v] changed.
        # This is getting complex. Let's use the simpler approach:
        # For vertex v, let s = sum of adj[v, u] where labels[u] == labels[v] (same side)
        # Let o = sum of adj[v, u] where labels[u] != labels[v] (opposite side)
        # s + o = deg[v]
        # Currently, v contributes o to the cut (edges from v to opposite side)
        # If we flip v, v contributes s to the cut (edges from v to what was same, now opposite)
        # Net change: s - o = s - (deg[v] - s) = 2s - deg[v]
        # We want to flip if 2s - deg[v] > 0, i.e., s > deg[v]/2
        # i.e., more neighbors on same side than opposite side
        # Equivalently, o < deg[v]/2, i.e., fewer neighbors on opposite side
        
        # Vectorized: 
        # s[v] = adj[v] @ labels (sum of adj to same-side neighbors, since labels are 0/1 and we include v itself but adj[v,v]=0)
        # Wait: adj[v] @ labels = sum_u adj[v,u] * labels[u]. This includes neighbors with label 1.
        # If labels[v] = 0, then same-side neighbors are those with label 0, so s = adj[v] @ (1 - labels)
        # If labels[v] = 1, then same-side neighbors are those with label 1, so s = adj[v] @ labels
        # So s[v] = adj[v] @ (labels == labels[v]) ... but vectorized:
        # s = np.where(labels == 0, adj @ (1 - labels), adj @ labels)
        # But adj @ (1 - labels) and adj @ labels can be computed once.
        
        prod1 = adj @ labels  # sum of adj to label-1 neighbors
        prod0 = adj @ (1 - labels)  # sum of adj to label-0 neighbors
        
        # For v with label 0: same-side sum = prod0[v]
        # For v with label 1: same-side sum = prod1[v]
        same_sum = np.where(labels == 0, prod0, prod1)
        
        # Gain from flipping: 2 * same_sum - deg[v]
        gain = 2 * same_sum - deg
        
        # Find vertices with positive gain
        # Flip all at once (parallel updates) or one at a time?
        # Let's flip all with positive gain simultaneously for speed, then recompute
        to_flip = gain > 0
        if not np.any(to_flip):
            break
        labels[to_flip] = 1 - labels[to_flip]
        improved = True
    
    return labels


def partition(adjacency):
    """
    Partition a graph into two sets to maximize the cut size.
    Uses spectral embedding + random hyperplane projections for initialization,
    followed by greedy local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    adj = np.array(adjacency, dtype=float)
    n = len(adj)
    
    # Make symmetric (in case)
    adj = (adj + adj.T) / 2.0
    
    rng = np.random.RandomState(42)
    
    # Compute spectral embedding once
    embedding = _spectral_init(adj, n)
    
    best_labels = None
    best_cut = -1
    
    # Generate multiple initial partitions using random hyperplanes
    # Also include a simple alternating start as baseline
    inits = []
    
    # Alternating start
    inits.append(np.arange(n) % 2)
    
    # Spectral sign (Fiedler vector)
    fiedler = embedding[:, 0]
    inits.append((fiedler >= 0).astype(int))
    
    # Random hyperplane projections
    n_hyperplanes = 20
    for i in range(n_hyperplanes):
        if time.time() > deadline - 5:
            break
        labels = _random_hyperplane_partition(embedding, rng)
        inits.append(labels)
    
    # Evaluate each initialization and improve
    for labels in inits:
        if time.time() > deadline - 10:
            break
        
        # Improve with greedy local search
        improved_labels = _greedy_flip_improve(adj, labels, max_rounds=50)
        
        # Compute cut size
        cut = _cut_size(adj, improved_labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = improved_labels.copy()
    
    # If we still have time, try more hyperplanes or more local search
    if best_labels is None:
        best_labels = inits[0]
    
    return best_labels
# EVOLVE-BLOCK-END
