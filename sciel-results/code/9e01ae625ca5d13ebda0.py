"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    cross = np.zeros(n, dtype=np.int64)
    for u in range(n):
        if labels[u] == 0:
            cross[u] = adjacency[u, labels == 1].sum()
        else:
            cross[u] = adjacency[u, labels == 0].sum()
    return int(np.sum(cross) // 2)

def _local_search_flip(adjacency, labels, max_passes=50):
    """Single-flip local search with vectorized delta computation. Returns improved labels and cut size."""
    n = len(labels)
    labels = labels.copy()
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # For each vertex u, compute delta_cut if we flip u
        # delta_cut = 2 * (sum of edges to other side) - (sum of edges to same side)
        # Actually: flipping u changes cut by:
        #   - lose edges from u to other side (they become same-side)
        #   + gain edges from u to same side (they become other-side)
        # delta = (edges to same side) - (edges to other side)
        # If delta > 0, flipping improves cut
        
        # Vectorized: for label 0 vertices, edges to label 1
        # For label 1 vertices, edges to label 0
        same_mask = (labels == labels)  # This is all True, not useful directly
        # Better: for each u, count neighbors with same label vs different label
        # Use matrix multiplication approach
        # cut_contribution[u] = sum_v adj[u,v] * (labels[u] != labels[v])
        # If we flip u, new contribution = sum_v adj[u,v] * (1-labels[u] != labels[v])
        # delta = new - old = sum_v adj[u,v] * ((1-labels[u] != labels[v]) - (labels[u] != labels[v]))
        # If labels[u]=0: old = adj[u, labels==1].sum(), new = adj[u, labels==0].sum()
        # delta = adj[u, labels==0].sum() - adj[u, labels==1].sum()
        # If labels[u]=1: old = adj[u, labels==0].sum(), new = adj[u, labels==1].sum()
        # delta = adj[u, labels==1].sum() - adj[u, labels==0].sum()
        
        # Vectorized computation:
        # For all vertices, compute:
        # sum_same[u] = adj[u, labels==labels[u]]
        # sum_diff[u] = adj[u, labels!=labels[u]]
        # delta[u] = sum_same[u] - sum_diff[u]
        
        # Using matrix ops:
        # sum_diff = adjacency @ (labels != labels).T ... not quite
        # Let's do it efficiently:
        mask0 = (labels == 0)
        mask1 = (labels == 1)
        
        # For vertices with label 0:
        # sum_same = adjacency[mask0, :][mask0, mask0].sum(axis=1) -> not straightforward
        # Simpler: 
        # sum_diff[u] = adjacency[u, :].sum() - adjacency[u, labels==labels[u]].sum()
        # Let's just compute directly with vectorized ops
        
        # Create a matrix where M[u,v] = 1 if labels[u] != labels[v], else 0
        # M = (labels[:, None] != labels[None, :]).astype(np.int8)
        # sum_diff = adjacency @ M.sum(axis=0) ... no
        # sum_diff[u] = (adjacency * M).sum(axis=1)
        
        M = (labels[:, None] != labels[None, :]).astype(np.int8)
        sum_diff = (adjacency * M).sum(axis=1)
        total_deg = adjacency.sum(axis=1)
        sum_same = total_deg - sum_diff
        delta = sum_same - sum_diff  # = 2*sum_same - total_deg
        
        # Find vertices where flipping improves cut
        flip_mask = delta > 0
        if flip_mask.any():
            labels[flip_mask] = 1 - labels[flip_mask]
            improved = True
    
    return labels, _cut_size(adjacency, labels)

def _spectral_init(adjacency):
    """Use Fiedler vector of graph Laplacian for initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
    except Exception:
        labels = np.random.randint(0, 2, n).astype(np.int8)
    
    return labels

def _random_init(n, seed=None):
    """Random initial partition."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.randint(0, 2, n).astype(np.int8)

def _quantum_annealing(adjacency, initial_labels, n_steps=500, beta_start=1.0, beta_end=10.0, gamma_max=5.0, seed=42):
    """
    Quantum Annealing-inspired update for Maximum Cut.
    
    Uses a transverse-field schedule: H(t) = -A(t) * X - B(t) * H_problem
    where X is the transverse field (encourages tunneling) and H_problem encodes the cut.
    
    We simulate this by maintaining a "spin" vector and updating it with a probability
    that depends on the transverse field strength (which allows tunneling through barriers).
    
    Parameters:
    - adjacency: int8 adjacency matrix
    - initial_labels: int8 array of initial 0/1 labels
    - n_steps: number of annealing steps
    - beta_start, beta_end: inverse temperature schedule
    - gamma_max: maximum transverse field strength
    - seed: random seed
    
    Returns:
    - labels: int8 array of final 0/1 labels
    - cut: final cut size
    """
    rng = np.random.RandomState(seed)
    n = len(initial_labels)
    labels = initial_labels.copy().astype(np.int8)
    
    # Precompute for fast delta computation
    # We'll maintain the cut and update it incrementally
    
    A = adjacency.astype(np.float64)
    
    # Current cut
    cut = _cut_size(adjacency, labels)
    
    # Transverse field schedule: gamma(t) decreases from gamma_max to 0
    # This allows quantum tunneling early on, then settles into classical behavior
    
    for step in range(n_steps):
        t = step / max(n_steps - 1, 1)
        
        # Transverse field strength: starts high, decays to 0
        gamma = gamma_max * (1 - t) ** 1.5
        
        # Temperature: starts high, decays
        beta = beta_start + (beta_end - beta_start) * t
        temp = 1.0 / max(beta, 1e-10)
        
        # With high transverse field, we allow more aggressive moves (tunneling)
        # With low transverse field, we behave like classical SA
        
        # Choose a random vertex to flip
        u = rng.randint(n)
        
        # Compute delta cut if we flip u
        # delta = 2 * (edges to same side) - degree[u]
        # Or: delta = (edges to same side) - (edges to different side)
        
        # Efficient delta computation:
        # For vertex u with label l:
        # edges_to_same = adjacency[u, labels==l].sum()
        # edges_to_diff = adjacency[u, labels!=l].sum()
        # delta = edges_to_same - edges_to_diff
        # If we flip, cut changes by delta
        
        l = labels[u]
        edges_to_same = adjacency[u, labels == l].sum()
        edges_to_diff = adjacency[u, labels != l].sum()
        delta = edges_to_same - edges_to_diff
        
        # Acceptance probability:
        # Classical SA: accept if delta > 0, or with prob exp(delta/temp) if delta < 0
        # Quantum: the transverse field allows tunneling, so we use a modified acceptance
        # that's more permissive when gamma is high
        
        if delta >= 0:
            accept = True
        else:
            # Classical acceptance
            p_classical = np.exp(delta / max(temp, 1e-10))
            
            # Quantum tunneling bonus: when gamma is high, we add extra probability
            # This represents the ability to tunnel through energy barriers
            tunneling_bonus = gamma * 0.1  # Scale factor
            p_quantum = min(1.0, p_classical + tunneling_bonus * (1 - t))
            
            accept = rng.random() < p_quantum
        
        if accept:
            labels[u] = 1 - labels[u]
            cut += delta  # Note: delta was computed for the flip, so cut changes by delta
    
    return labels, cut

def _quantum_annealing_multi(adjacency, n_restarts=10, seed=42):
    """Run multiple quantum annealing restarts from different initial states."""
    n = len(adjacency)
    best_labels = None
    best_cut = -1
    
    for i in range(n_restarts):
        # Different initialization strategies
        if i == 0:
            init = _spectral_init(adjacency)
        elif i == 1:
            init = _random_init(n, seed=seed + 100)
        else:
            init = _random_init(n, seed=seed + 100 + i)
        
        labels, cut = _quantum_annealing(
            adjacency, init,
            n_steps=800,
            beta_start=0.5,
            beta_end=15.0,
            gamma_max=8.0,
            seed=seed + i
        )
        
        # Polish with classical local search
        polished_labels, polished_cut = _local_search_flip(adjacency, labels, max_passes=50)
        
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished_labels.copy()
    
    return best_labels, best_cut

def partition(adjacency):
    """
    Maximum Cut via quantum annealing + classical local search polish.
    
    Parameters:
    - adjacency: bool matrix (n x n)
    
    Returns:
    - 0/1 label array of length n
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = len(adjacency)
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral init + classical local search (fast baseline)
    spectral_labels = _spectral_init(adjacency)
    refined_labels, cut_val = _local_search_flip(adjacency, spectral_labels, max_passes=100)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels
    
    # Strategy 2: Quantum Annealing with multiple restarts
    # Use the remaining time budget for QA restarts
    time_remaining = deadline - time.time() - 5.0  # 5s safety margin
    
    if time_remaining > 10.0:
        # Estimate how many QA restarts we can afford
        # Each QA restart takes roughly n_steps * n / some_constant
        # For n=100, n_steps=800, each restart is ~80000 operations
        # Let's be conservative and try up to 15 restarts
        max_qa_restarts = min(15, int(time_remaining / 2.0))  # ~2s per restart estimate
        
        qa_labels, qa_cut = _quantum_annealing_multi(
            adjacency,
            n_restarts=max_qa_restarts,
            seed=12345
        )
        
        if qa_cut > best_cut:
            best_cut = qa_cut
            best_labels = qa_labels.copy()
    
    # Check time
    if time.time() > deadline - 10.0:
        return best_labels
    
    # Strategy 3: Additional classical local search from random starts
    max_random_starts = 10
    for i in range(max_random_starts):
        if time.time() > deadline - 10.0:
            break
        
        random_labels = _random_init(n, seed=42 + i)
        refined_labels, cut_val = _local_search_flip(adjacency, random_labels, max_passes=100)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
    
    # Final polish: more iterations on the best solution
    if best_labels is not None and time.time() < deadline - 2.0:
        polished_labels, cut_val = _local_search_flip(adjacency, best_labels, max_passes=200)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = polished_labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
