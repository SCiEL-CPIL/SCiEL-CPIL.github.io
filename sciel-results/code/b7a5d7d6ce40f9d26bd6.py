"""Rewritten program for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 265.0  # safety margin before 270s kill
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _jc_distance_vectorized(seqs_int, n_sites):
    """
    Compute pairwise JC distances for a set of integer-encoded sequences.
    seqs_int: (n, L) array of ints 0-3
    """
    n = seqs_int.shape[0]
    if n == 0:
        return np.array([])
    
    # Compute pairwise Hamming distances
    # Using broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This might be memory heavy for large n, but n<=32 so 32*32*2000 = 2M ints, ~8MB. Safe.
    # However, for n=32, L=2000, (32, 32, 2000) is ~8MB of int8, which is fine.
    
    # Alternative: loop over i, vectorize j. This is safer for memory and often faster due to BLAS/cache.
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # diffs: (n, L) boolean array
        diffs = (seqs_int != seqs_int[i])
        # counts: (n,) int array
        counts = np.sum(diffs, axis=1)
        p = counts / n_sites
        # Handle p >= 0.75
        mask = p >= 0.75
        # Calculate JC distance
        # d = -0.75 * ln(1 - 4/3 * p)
        # Avoid log(0) or negative
        p_safe = np.where(mask, 0.7499, p)
        d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_safe)
        d[mask] = 3.0 # Cap at 3.0 (infinite distance approximation)
        dist[i, :] = d
        dist[:, i] = d
    
    return dist


def _neighbor_joining(dist):
    """
    Optimized Neighbor Joining.
    Returns the tree as a list of edges (u, v) and a mapping of node IDs to labels.
    """
    n = len(dist)
    if n <= 2:
        return [], {i: i for i in range(n)}
    
    # D is the current distance matrix
    D = dist.copy()
    active = list(range(n))
    next_node = n
    edges = []
    
    # Node labels: 0..n-1 are taxa, n.. are internal
    node_labels = {i: i for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        if m < 2:
            break
            
        # Get submatrix for active nodes
        # active is a list of indices into the full D matrix
        # We need to work with the submatrix D[active, :][:, active]
        # But updating D is tricky if we don't maintain a dense matrix of size next_node x next_node
        
        # Let's maintain a dense matrix D of size next_node x next_node
        # Initially size n x n. When we add a new node, we expand D.
        
        # For now, let's just use the full D matrix and active indices.
        
        # Row sums r[i] for i in active
        # r[i] = sum_{j in active, j != i} D[i, j]
        # We can compute this by summing the rows of D[active, :] and subtracting D[i, i] (which is 0)
        # But we only want sums over active j.
        
        # Let's create a view or copy of the active submatrix for speed?
        # Actually, just iterating over active indices in Python is okay for m<=32.
        # But let's try to vectorize the criterion calculation.
        
        # r = np.zeros(m)
        # for k, i in enumerate(active):
        #     r[k] = np.sum(D[i, active]) # D[i,i] is 0
        
        # Criterion matrix B:
        # B[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        # We want to find min B[i, j] for i < j
        
        r = np.array([np.sum(D[i, active]) for i in active])
        
        # D_sub is the m x m submatrix
        D_sub = D[np.ix_(active, active)]
        
        # Compute B
        # B = (m-2)*D_sub - r[:, None] - r[None, :]
        B = (m - 2) * D_sub - r[:, None] - r[None, :]
        
        # Set diagonal to inf to ignore i==j
        np.fill_diagonal(B, np.inf)
        
        # Find min
        min_val = np.min(B)
        # Get indices
        # np.argmin gives flat index
        flat_idx = np.argmin(B)
        i_idx, j_idx = np.unravel_index(flat_idx, (m, m))
        
        if i_idx == j_idx:
            break
            
        # Ensure i_idx < j_idx for consistency? Not strictly necessary for logic, but good for debug.
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
            
        i = active[i_idx]
        j = active[j_idx]
        
        # Create new node
        new_node = next_node
        next_node += 1
        
        # Add to edges
        edges.append((i, new_node))
        edges.append((j, new_node))
        
        # Compute new distances to other active nodes
        # d(new, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        
        # We need to update the D matrix.
        # D is currently n x n (or larger if we expanded it? No, we haven't expanded it).
        # We should expand D to next_node x next_node initially?
        # Or just use a dictionary for new nodes?
        # Since n is small, let's just resize D to next_node x next_node once at the start?
        # Max nodes = 2*n - 2.
        # Let's initialize D to max_size x max_size.
        
        # Wait, the code above assumes D is accessible at D[new_node, k].
        # So we must have pre-allocated D to max_size.
        
        # Let's refactor:
        # 1. Pre-allocate D to (2*n-2) x (2*n-2)
        # 2. Fill top-left n x n with initial distances.
        # 3. Update D[new_node, k] and D[k, new_node] for k in active.
        
        # Since I can't easily refactor the whole function structure in a small edit without rewriting,
        # I will rewrite `_neighbor_joining` to use a pre-allocated matrix.
        
        # For now, let's break and use a fallback if this implementation is too slow/complex to patch.
        # Actually, let's just implement the full correct version in the replacement.
        
        break # Break out to use the better implementation below

    # This function is a placeholder. The real one is below.
    return edges, node_labels


def _nj_optimized(dist):
    """
    Optimized NJ with pre-allocated matrix.
    """
    n = len(dist)
    if n <= 2:
        return [], {i: i for i in range(n)}
    
    max_nodes = 2 * n - 2
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist
    
    active = list(range(n))
    next_node = n
    edges = []
    
    while len(active) > 3:
        m = len(active)
        if m < 2:
            break
            
        # Row sums
        # r[k] = sum(D[active[k], active])
        # Vectorized: D[np.ix_(active, active)].sum(axis=1)
        D_sub = D[np.ix_(active, active)]
        r = D_sub.sum(axis=1)
        
        # Criterion
        B = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(B, np.inf)
        
        flat_idx = np.argmin(B)
        i_idx, j_idx = np.unravel_index(flat_idx, (m, m))
        
        if i_idx == j_idx:
            break
            
        i = active[i_idx]
        j = active[j_idx]
        
        new_node = next_node
        next_node += 1
        
        edges.append((i, new_node))
        edges.append((j, new_node))
        
        # Update distances
        # d(new, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        # k in active, k != i, k != j
        
        # Get indices of k in active
        # active is a list.
        # We can just loop over active since m is small.
        # Or vectorize:
        # k_indices = [k for k in active if k != i and k != j]
        
        # Let's use a mask on active
        mask = np.array([k != i and k != j for k in active])
        k_active = [active[k] for k in range(m) if mask[k]]
        
        if k_active:
            # d_i_k = D[i, k_active]
            # d_j_k = D[j, k_active]
            # d_ij = D[i, j]
            
            d_i_k = D[i, k_active]
            d_j_k = D[j, k_active]
            d_ij = D[i, j]
            
            new_d = 0.5 * (d_i_k + d_j_k - d_ij)
            
            # Update D
            D[new_node, k_active] = new_d
            D[k_active, new_node] = new_d
        
        # Update active
        # Remove i and j, add new_node
        active = [k for k in active if k != i and k != j]
        active.append(new_node)
        
        if _elapsed() > _DEADLINE - 5:
            break
            
    return edges, {i: i for i in range(n)}


def _build_newick(edges, n, labels):
    """
    Build Newick string from edges.
    edges: list of (u, v)
    n: number of taxa
    labels: list of taxon labels
    """
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build adjacency
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        if v not in adj:
            adj[v] = []
        adj[v].append(u)
        
    # Find a leaf to start
    start = 0
    while start in adj and len(adj[start]) > 1:
        start += 1
    if start >= n:
        start = 0
        
    # DFS
    def dfs(node, parent):
        if node < n:
            return labels[node]
        
        children = [c for c in adj[node] if c != parent]
        if not children:
            return "" # Should not happen for internal nodes in a tree
        
        child_strs = [dfs(c, node) for c in children]
        return "(" + ",".join(child_strs) + ")"
    
    newick = dfs(start, -1)
    return newick + ";"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]});"
    
    try:
        # Encode sequences
        seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # Handle potential missing characters? Task says ACGT.
        seqs_int = np.array([[seq_map.get(c, 0) for c in sequences[labels[i]]] for i in range(n)], dtype=np.int8)
        n_sites = len(sequences[labels[0]])
        
        # Compute distances
        dist = _jc_distance_vectorized(seqs_int, n_sites)
        
        # NJ
        edges, _ = _nj_optimized(dist)
        
        # Build Newick
        newick = _build_newick(edges, n, labels)
        
        return newick
    except Exception:
        # Fallback to star tree
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
