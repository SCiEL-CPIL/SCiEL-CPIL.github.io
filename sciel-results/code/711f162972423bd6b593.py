"""
Strategy: Spectral embedding + hierarchical clustering for phylogenetic topology recovery.
Instead of greedy NJ, we:
1. Compute JC distance matrix
2. Build a graph Laplacian from the distance matrix
3. Extract low-frequency eigenvectors (spectral coordinates)
4. Use agglomerative clustering on spectral coordinates to build the tree
This captures the hierarchical structure inherent in trees via spectral methods.
"""
import time
import math
import numpy as np
from scipy.linalg import eigh

_START_TIME = None
_TIME_LIMIT = 250.0  # Safety margin for 270s limit


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances using NumPy for speed."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    
    return dists


def _spectral_embed(dists, n_components=None):
    """
    Compute spectral embedding of the distance matrix.
    Uses the graph Laplacian's low-frequency eigenvectors.
    """
    n = dists.shape[0]
    if n_components is None:
        n_components = min(n - 1, max(3, n // 4))
    
    # Build a similarity/affinity matrix from distances
    # Use a Gaussian kernel: A_ij = exp(-d_ij^2 / sigma^2)
    # Choose sigma based on median distance
    d_upper = dists[np.triu_indices(n, k=1)]
    sigma = np.median(d_upper[d_upper > 0]) if np.any(d_upper > 0) else 1.0
    sigma = max(sigma, 1e-6)
    
    # Affinity matrix
    D = dists ** 2
    A = np.exp(-D / (2 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    
    # Graph Laplacian: L = D - A where D is degree matrix
    degrees = A.sum(axis=1)
    D_mat = np.diag(degrees)
    L = D_mat - A
    
    # For better conditioning, use normalized Laplacian
    # L_norm = I - D^(-1/2) A D^(-1/2)
    # But for simplicity, use unnormalized and handle small eigenvalues
    
    # Compute eigendecomposition
    # We want the smallest eigenvalues (excluding 0)
    try:
        eigenvalues, eigenvectors = eigh(L)
    except Exception:
        # Fallback: use the distance matrix directly
        eigenvalues, eigenvectors = eigh(dists)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Skip the first eigenvector (corresponds to eigenvalue ~0, constant vector)
    # Take the next n_components eigenvectors
    start = 1
    end = min(start + n_components, n)
    
    if end - start < 2:
        # Not enough components, take what we can
        embedding = eigenvectors[:, start:min(end, n)]
    else:
        embedding = eigenvectors[:, start:end]
    
    # Normalize the embedding
    # Center it
    embedding = embedding - embedding.mean(axis=0)
    
    return embedding


def _agglomerative_cluster(embedding, labels, n_taxa):
    """
    Build a tree using agglomerative clustering on spectral coordinates.
    Returns a Newick string representing the unrooted tree.
    """
    n = n_taxa
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]});"
    
    # Start with each taxon as a cluster
    clusters = [[i] for i in range(n)]
    cluster_ids = list(range(n))
    active_clusters = list(range(n))
    
    # Precompute pairwise distances between cluster centroids
    # Use the embedding for distance computation
    def cluster_centroid(cluster):
        return embedding[cluster].mean(axis=0)
    
    # Track the tree structure
    # Each merge creates a new internal node
    node_labels = {i: labels[i] for i in range(n)}
    children = {}
    next_node_id = n
    
    # Compute initial pairwise distances between all clusters
    def compute_pairwise_dists(clusters, active):
        n_active = len(active)
        d = np.zeros((n_active, n_active))
        centroids = {}
        for idx, c_id in enumerate(active):
            centroids[c_id] = cluster_centroid(clusters[c_id])
        for i in range(n_active):
            for j in range(i + 1, n_active):
                ci, cj = active[i], active[j]
                dist = np.linalg.norm(centroids[ci] - centroids[cj])
                d[i, j] = dist
                d[j, i] = dist
        return d, centroids
    
    d, centroids = compute_pairwise_dists(clusters, active_clusters)
    
    while len(active_clusters) > 3:
        # Find the pair of clusters with minimum distance
        n_active = len(active_clusters)
        min_dist = np.inf
        min_i, min_j = 0, 1
        for i in range(n_active):
            for j in range(i + 1, n_active):
                if d[i, j] < min_dist:
                    min_dist = d[i, j]
                    min_i, min_j = i, j
        
        # Merge clusters min_i and min_j
        cluster_a = active_clusters[min_i]
        cluster_b = active_clusters[min_j]
        
        new_cluster = cluster_a + cluster_b
        new_node = next_node_id
        next_node_id += 1
        
        children[new_node] = [cluster_a, cluster_b]
        node_labels[new_node] = None
        
        clusters.append(new_cluster)
        
        # Remove the two merged clusters from active
        # Be careful with indices
        if min_i > min_j:
            min_i, min_j = min_j, min_i
        active_clusters = [c for idx, c in enumerate(active_clusters) if idx not in (min_i, min_j)]
        active_clusters.append(new_node)
        
        # Recompute distances
        d, centroids = compute_pairwise_dists(clusters, active_clusters)
    
    # Final 3 clusters: connect them in a star
    a, b, c = active_clusters
    # Create a binary tree: ((a,b),c)
    newick_a = _build_newick(a, children, node_labels)
    newick_b = _build_newick(b, children, node_labels)
    newick_c = _build_newick(c, children, node_labels)
    
    return f"(({newick_a},{newick_b}),{newick_c});"


def _build_newick(node, children, node_labels):
    """Recursively build Newick string."""
    if node in children:
        left = _build_newick(children[node][0], children, node_labels)
        right = _build_newick(children[node][1], children, node_labels)
        return f"({left},{right})"
    else:
        return node_labels[node]


def _nj_run(dists, labels, seed=None):
    """
    Run one instance of Neighbor Joining.
    Returns Newick string.
    """
    import random as _random
    rng = _random.Random(seed) if seed is not None else _random
    
    n = len(labels)
    
    current_d = dists.copy()
    active = list(range(n))
    node_to_label = {i: labels[i] for i in range(n)}
    next_node_id = n
    
    children = {}
    
    while len(active) > 3:
        n_curr = len(active)
        
        # Calculate row sums
        r = np.zeros(n_curr)
        for i in range(n_curr):
            for j in range(n_curr):
                r[i] += current_d[i, j]
        
        # Calculate Q matrix
        q_matrix = np.zeros((n_curr, n_curr))
        for i in range(n_curr):
            for j in range(n_curr):
                if i != j:
                    q_matrix[i, j] = (n_curr - 2) * current_d[i, j] - r[i] - r[j]
                else:
                    q_matrix[i, j] = np.inf
        
        # Find minimum Q
        min_q = np.inf
        min_i, min_j = 0, 1
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                if q_matrix[i, j] < min_q:
                    min_q = q_matrix[i, j]
                    min_i, min_j = i, j
        
        # Randomized tie-breaking
        threshold = 1e-6
        candidates = []
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                if q_matrix[i, j] <= min_q + threshold:
                    candidates.append((i, j))
        
        if len(candidates) > 1:
            min_i, min_j = rng.choice(candidates)
        
        i, j = min_i, min_j
        
        new_node = next_node_id
        next_node_id += 1
        
        children[new_node] = [active[i], active[j]]
        
        other_indices = [k for k in range(n_curr) if k != i and k != j]
        d_ij = current_d[i, j]
        
        new_dists = {}
        for k in other_indices:
            new_d = 0.5 * (current_d[i, k] + current_d[j, k] - d_ij)
            new_dists[k] = max(0.0, new_d)
        
        new_active = [active[k] for k in other_indices] + [new_node]
        node_to_label[new_node] = None
        
        n_new = len(new_active)
        new_d = np.zeros((n_new, n_new))
        
        for pos, k in enumerate(other_indices):
            for pos2, k2 in enumerate(other_indices):
                new_d[pos, pos2] = current_d[k, k2]
        
        new_idx = len(other_indices)
        for pos, k in enumerate(other_indices):
            val = new_dists[k]
            new_d[new_idx, pos] = val
            new_d[pos, new_idx] = val
        
        new_d[new_idx, new_idx] = 0.0
        
        active = new_active
        current_d = new_d
    
    # Final 3 nodes
    a, b, c = active
    
    def build_newick(node):
        if node in children:
            left = build_newick(children[node][0])
            right = build_newick(children[node][1])
            return f"({left},{right})"
        else:
            return node_to_label[node]
    
    newick = f"(({build_newick(a)},{build_newick(b)}),{build_newick(c)});"
    return newick


def _spectral_tree(dists, labels):
    """
    Build a phylogenetic tree using spectral embedding + agglomerative clustering.
    """
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]});"
    
    # Compute spectral embedding
    embedding = _spectral_embed(dists)
    
    # Use agglomerative clustering on spectral coordinates
    tree = _agglomerative_cluster(embedding, labels, n)
    
    return tree


def _nj_tree(dists, labels):
    """Build a tree using NJ."""
    return _nj_run(dists, labels, seed=0)


def _choose_tree(spectral_newick, nj_newick, dists, labels, sequences):
    """
    Choose between spectral tree and NJ tree based on a simple criterion.
    For now, we'll use the spectral tree as the primary method.
    """
    # We could compare parsimony scores, but that's complex.
    # Let's just use the spectral tree.
    return spectral_newick


def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Try spectral approach first
    try:
        spectral_tree = _spectral_tree(dists, labels)
    except Exception:
        spectral_tree = None
    
    # Fallback to NJ if spectral fails
    if spectral_tree is None:
        nj_tree = _nj_tree(dists, labels)
        return nj_tree
    
    # For now, return the spectral tree
    return spectral_tree
