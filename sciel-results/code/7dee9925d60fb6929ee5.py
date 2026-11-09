"""Rewritten program for Max-Cut on 100-node graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(labels, adj):
    """Compute the cut size (number of edges crossing the partition)."""
    # For each edge (i,j), it crosses if labels[i] != labels[j]
    # Cut size = sum over all i<j where adj[i][j] and labels[i] != labels[j]
    # Efficient: for each vertex, count edges to opposite partition
    # But double counts, so divide by 2
    # Actually, let's use a matrix approach
    # cut = sum_{i,j: labels[i]!=labels[j]} adj[i][j] / 2
    # More efficient: 
    # For each vertex v, the number of edges from v to the opposite side is:
    #   if labels[v] == 0: sum of adj[v][labels==1]
    #   if labels[v] == 1: sum of adj[v][labels==0]
    # Total cut = sum over all v of that / 2 (since each edge counted twice)
    
    # Vectorized approach:
    # Create a matrix M where M[i,j] = 1 if labels[i] != labels[j], else 0
    # Then cut_size = sum(adj * M) / 2
    
    # More efficient: 
    # Let's use the fact that cut = (total_edges - within_edges)
    # Or directly: for each vertex, count neighbors in opposite partition
    
    # Simple vectorized:
    # diff = (labels[:, None] != labels[None, :])  # n x n boolean
    # return (adj * diff).sum() / 2
    
    # Even more efficient using matrix multiplication:
    # Let x be a vector where x[i] = 1 if labels[i] == 0, -1 if labels[i] == 1
    # Then cut_size = (n * total_edges - x^T A x) / 4 ... let me derive
    # Actually: for edge (i,j), it crosses iff x[i] != x[j] iff x[i]*x[j] = -1
    # So sum over edges of (1 - x[i]*x[j])/2 = (num_edges - sum_{i,j} x[i]*x[j]*adj[i][j]) / 2
    # Wait, let me be more careful.
    
    # Let x[i] = 1 if labels[i] == 0, x[i] = -1 if labels[i] == 1
    # An edge (i,j) crosses the cut iff x[i] != x[j], i.e., x[i]*x[j] = -1
    # Number of crossing edges = (1/2) * sum_{i,j} adj[i][j] * (1 - x[i]*x[j])
    #   = (1/2) * (sum_{i,j} adj[i][j] - sum_{i,j} adj[i][j]*x[i]*x[j])
    #   = (1/2) * (2*E - x^T A x)  where E is number of edges
    #   = E - (1/2) * x^T A x
    
    # So cut_size = E - 0.5 * x^T A x
    
    x = np.where(labels == 0, 1.0, -1.0)
    # x^T A x
    xAx = x @ adj @ x
    E = adj.sum() / 2.0
    return E - 0.5 * xAx


def _local_search(labels, adj, n, max_iters=1000, time_limit=None, start_time=None):
    """Greedy local search: flip vertex if it improves cut size."""
    labels = labels.copy()
    best_cut = _cut_size(labels, adj)
    
    if time_limit is None:
        time_limit = 10.0
    if start_time is None:
        start_time = time.time()
    
    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit - 2.0:
            break
        
        improved = False
        # For each vertex, compute the gain from flipping
        # If we flip vertex v from label l to 1-l:
        # The change in cut size = (edges from v to same side) - (edges from v to opposite side)
        # = 2 * (edges from v to opposite side) - degree(v)
        # Wait, let me think again.
        
        # Current: v is on side s. 
        # Edges from v to side s (including itself? no, no self-loops): these are "within" edges from v's perspective
        # Edges from v to side 1-s: these are "crossing" edges from v's perspective
        
        # If we flip v to side 1-s:
        # The edges that were crossing (v to 1-s) become within
        # The edges that were within (v to s) become crossing
        # So change = (edges v->s) - (edges v->1-s)
        
        # Let same_count = number of neighbors of v on same side as v
        # Let other_count = number of neighbors of v on opposite side
        # Change in cut = same_count - other_count
        
        # We flip if same_count > other_count, i.e., gain > 0
        
        # Vectorized computation for all vertices:
        # For vertex v, same_count = sum of adj[v][labels == labels[v]]
        # other_count = sum of adj[v][labels != labels[v]]
        
        # Let's compute for all vertices at once
        # same_mask[v, u] = 1 if labels[v] == labels[u]
        # We want, for each v: sum_u adj[v,u] * (labels[v] == labels[u])
        
        # This is: (adj * (labels[:, None] == labels[None, :])).sum(axis=1)
        # But that's O(n^2) per iteration, which is fine for n=100
        
        same_eq = (labels[:, None] == labels[None, :])
        same_counts = (adj * same_eq).sum(axis=1)
        total_degrees = adj.sum(axis=1)
        other_counts = total_degrees - same_counts
        
        gains = same_counts - other_counts
        
        # Find the vertex with the best positive gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain <= 0:
            break
        
        # Flip the vertex
        labels[max_gain_idx] = 1 - labels[max_gain_idx]
        best_cut = _cut_size(labels, adj)
        improved = True
    
    return labels, best_cut


def partition(adjacency):
    """
    Find a good 2-partition (Max-Cut) of the graph.
    Uses spectral initialization + greedy local search with multiple random restarts.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Ensure symmetry (in case of slight asymmetry in input)
    adj = (adj + adj.T) // 2
    
    best_labels = None
    best_cut = -1
    
    # Spectral initialization: use Fiedler vector of the Laplacian
    # L = D - A
    degrees = adj.sum(axis=1).astype(float)
    L = np.diag(degrees) - adj.astype(float)
    
    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # For connected graphs, index 1. For disconnected, there might be multiple zero eigenvalues.
        # We'll take the eigenvector with the second smallest eigenvalue.
        fiedler = eigenvectors[:, 1]
        labels_spec = (fiedler < 0).astype(np.int8)
        
        # Make sure both sides are non-empty
        if labels_spec.sum() == 0 or labels_spec.sum() == n:
            labels_spec = np.arange(n) % 2
    except Exception:
        labels_spec = np.arange(n) % 2
    
    # Try spectral initialization
    labels, cut = _local_search(labels_spec, adj, n, time_limit=deadline - time.time(), start_time=start_time)
    best_labels = labels.copy()
    best_cut = cut
    
    # Multiple random restarts
    np.random.seed(42)
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        labels_rand = np.random.randint(0, 2, size=n).astype(np.int8)
        
        # Ensure both sides are non-empty
        if labels_rand.sum() == 0 or labels_rand.sum() == n:
            labels_rand[0] = 1 - labels_rand[0]
        
        labels, cut = _local_search(labels_rand, adj, n, time_limit=deadline - time.time(), start_time=start_time)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final polish: run local search once more on the best solution with more iterations
    if time.time() < deadline - 2.0:
        labels, cut = _local_search(best_labels, adj, n, max_iters=5000, time_limit=deadline - time.time(), start_time=start_time)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
