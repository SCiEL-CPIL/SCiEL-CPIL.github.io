"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # An edge (u,v) crosses the cut if labels[u] != labels[v]
    # Using vectorized computation
    return np.sum(adj * (labels[:, None] != labels[None, :])) // 2


def _local_search(adj, labels, deadline):
    """
    Greedy local search: flip a vertex if it improves the cut.
    Repeat until no improvement or deadline reached.
    """
    n = len(adj)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    while time.time() < deadline:
        improved = False
        for v in range(n):
            # Compute gain from flipping vertex v
            # Current contribution of v to cut: edges from v to opposite side
            # After flip: edges from v to same side become crossing
            same_side = (labels == labels[v]).astype(np.int8)
            other_side = 1 - same_side
            
            # Edges from v to same side (currently NOT crossing, would become crossing)
            edges_to_same = adj[v, :].dot(same_side) - adj[v, v]  # exclude self
            # Edges from v to other side (currently crossing, would become non-crossing)
            edges_to_other = adj[v, :].dot(other_side)
            
            # Gain = (new crossing) - (old crossing) = edges_to_same - edges_to_other
            gain = edges_to_same - edges_to_other
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _simulated_annealing(adj, labels, deadline, temperature=1.0):
    """
    Simulated annealing to escape local optima.
    """
    n = len(adj)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Cooling schedule
    T = temperature
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        same_side = (labels == labels[v]).astype(np.int8)
        other_side = 1 - same_side
        
        edges_to_same = adj[v, :].dot(same_side) - adj[v, v]
        edges_to_other = adj[v, :].dot(other_side)
        
        gain = edges_to_same - edges_to_other
        
        if gain >= 0 or np.random.random() < np.exp(gain / T):
            labels[v] = 1 - labels[v]
            current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        T *= cooling_rate
    
    return best_labels, best_cut


def _spectral_init(adj, n):
    """
    Generate initial partition using spectral method (Fiedler vector).
    """
    try:
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the second smallest eigenvalue's eigenvector
        fiedler = eigvecs[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except Exception:
        return np.arange(n) % 2


def _qaoa_init(adj, n, deadline):
    """
    QAOA-inspired initialization: use a simple variational approach
    to find a good initial partition.
    """
    # Start with random initializations and pick the best
    best_labels = None
    best_cut = -1
    
    num_inits = min(50, max(10, int((deadline - time.time()) * 10)))
    
    for i in range(num_inits):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if i == 0:
            labels = _spectral_init(adj, n)
        elif i == 1:
            labels = np.arange(n) % 2
        elif i == 2:
            labels = (np.random.random(n) > 0.5).astype(np.int8)
        else:
            # Random with some structure
            labels = (np.random.random(n) > np.random.uniform(0.3, 0.7)).astype(np.int8)
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj, best_labels)
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Find a maximum cut in the graph using a combination of:
    1. Spectral initialization
    2. QAOA-inspired multi-start
    3. Greedy local search
    4. Simulated annealing for refinement
    """
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Set deadline with safety margin
    deadline = time.time() + 140.0  # 10s margin before 150s kill
    
    # Phase 1: Get a good initial solution via multi-start
    labels, current_cut = _qaoa_init(adj, n, deadline - 10)
    
    # Phase 2: Greedy local search to find local optimum
    labels, current_cut = _local_search(adj, labels, deadline - 5)
    
    # Phase 3: Simulated annealing to potentially escape local optimum
    if time.time() < deadline - 2:
        sa_labels, sa_cut = _simulated_annealing(adj, labels, deadline - 2, temperature=2.0)
        if sa_cut > current_cut:
            labels = sa_labels
            current_cut = sa_cut
    
    # Phase 4: Final local search polish
    labels, current_cut = _local_search(adj, labels, deadline)
    
    return labels
# EVOLVE-BLOCK-END
