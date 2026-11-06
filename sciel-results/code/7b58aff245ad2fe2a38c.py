"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # For each edge (i,j) where i<j, count if labels[i] != labels[j]
    # Using matrix: cut = sum over i<j where adj[i,j]=1 and labels[i]!=labels[j]
    # Efficient: cut = 0.5 * sum(adj * (labels[:,None] != labels[None,:]))
    # But we can do: for each node, count neighbors with different label
    # cut = 0.5 * sum_v (sum_u adj[v,u] * (labels[v] != labels[u]))
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)


def _flip_gain(adj, labels, v):
    """Compute the gain in cut size if we flip vertex v."""
    # Currently, edges from v to same-side neighbors are NOT in cut.
    # Edges from v to other-side neighbors ARE in cut.
    # If we flip v:
    #   - edges to same-side become cross-side (gain = count_same)
    #   - edges to other-side become same-side (loss = count_other)
    # Net gain = count_same - count_other
    same_side = (labels == labels[v]).astype(np.int8)
    other_side = (labels != labels[v]).astype(np.int8)
    count_same = int(adj[v] @ same_side)
    count_other = int(adj[v] @ other_side)
    return count_same - count_other


def _local_search(adj, labels, deadline):
    """Greedy local search: repeatedly flip vertices that improve the cut until no improvement."""
    n = len(labels)
    labels = labels.copy()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            gain = _flip_gain(adj, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
        if time.time() > deadline:
            break
    return labels


def _simulated_annealing(adj, labels, deadline, start_temp=1.0, cooling=0.999):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp
    
    while time.time() < deadline and temp > 1e-6:
        # Randomly pick a vertex to flip
        v = np.random.randint(n)
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        new_cut = _cut_size(adj, labels)
        delta = new_cut - current_cut
        
        if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-10)):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label
        
        temp *= cooling
    
    return best_labels


def _spectral_init(adj, n):
    """Initialize using the Fiedler vector of the graph Laplacian."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector with the second smallest eigenvalue
        # Sort eigenvalues in ascending order
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        # Partition based on the sign of the Fiedler vector
        labels = (fiedler > 0).astype(np.int8)
        # Ensure both sides are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        return labels
    except:
        return np.arange(n) % 2


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Set deadline with safety margin
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Phase 1: Multiple restarts with different initializations
    best_labels = None
    best_cut = -1
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # Reserve 30s for final polish
            break
        
        # Different initialization strategies
        if restart == 0:
            # Spectral initialization
            labels = _spectral_init(adj, n)
        elif restart == 1:
            # Alternating
            labels = np.arange(n) % 2
        elif restart == 2:
            # Random
            labels = np.random.randint(0, 2, n).astype(np.int8)
        elif restart == 3:
            # Random with bias
            labels = (np.random.random(n) > 0.3).astype(np.int8)
        else:
            # Random
            labels = np.random.randint(0, 2, n).astype(np.int8)
        
        # Local search
        labels = _local_search(adj, labels, deadline - 20)
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 2: Simulated annealing from the best solution
    if time.time() < deadline - 10:
        sa_labels = _simulated_annealing(adj, best_labels, deadline - 5)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2:
        best_labels = _local_search(adj, best_labels, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
