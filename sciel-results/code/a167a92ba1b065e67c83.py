# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (eigenvector of Laplacian)."""
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler > 0).astype(int)
    except:
        labels = np.arange(n) % 2
    return labels


def _admm_maxcut(adj, n, max_iter=200, tol=1e-6, max_time=None):
    """
    ADMM (Alternating Direction Method of Multipliers) for Maximum Cut.
    This is a convex relaxation approach that often finds good solutions.
    
    The problem is:
    max (1/4) * x^T (D - A) x  subject to x_i = ±1
    where D is the degree matrix and A is the adjacency matrix.
    
    Using ADMM with the constraint x_i^2 = 1:
    min -1/4 * x^T L x + ρ/2 * ||x - z||^2 + u^T (x - z)
    s.t. z_i^2 = 1
    
    Where L = D - A is the Laplacian.
    """
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    # Initialize
    x = np.ones(n)
    z = np.ones(n)
    u = np.zeros(n)
    rho = 1.0
    rho_max = 10.0
    
    best_x = x.copy()
    best_val = -np.inf
    
    for it in range(max_iter):
        if time.time() > max_time:
            break
        
        # x-update: minimize -1/4 x^T L x + ρ/2 ||x - z + u/ρ||^2
        # Gradient: -1/4 L x + ρ (x - z + u/ρ) = 0
        # (ρ I - 1/4 L) x = ρ (z - u/ρ)
        # x = (ρ I - 1/4 L)^{-1} ρ (z - u/ρ)
        # x = (I - (1/(4ρ)) L)^{-1} (z - u/ρ)
        
        # Solve (ρ I - 0.25 L) x_new = ρ (z - u/ρ)
        A_mat = rho * np.eye(n) - 0.25 * L
        b_vec = rho * (z - u / rho)
        
        try:
            x_new = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            x_new = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        # z-update: project x_new + u/ρ onto {±1}
        z_new = np.sign(x_new + u / rho)
        z_new[z_new == 0] = 1  # Handle zeros
        
        # u-update
        u = u + rho * (x_new - z_new)
        
        x = x_new
        z = z_new
        
        # Check convergence
        primal_res = np.linalg.norm(x - z)
        dual_res = rho * np.linalg.norm(z_new - z)
        if primal_res < tol and dual_res < tol:
            break
        
        # Adaptive rho
        if primal_res > 10 * dual_res:
            rho = min(rho * 2, rho_max)
        elif dual_res > 10 * primal_res:
            rho = max(rho / 2, 0.1)
        
        # Track best integer solution
        z_int = (z_new > 0).astype(float) * 2 - 1  # ±1
        val = -0.25 * z_int @ L @ z_int
        if val > best_val:
            best_val = val
            best_x = z_int.copy()
    
    # Convert to 0/1 labels
    labels = ((best_x > 0).astype(int))
    return labels


def _local_search_1flip(adj, labels, max_time=None):
    """Greedy 1-flip local search with vectorized gain computation."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    improved = True
    while improved:
        if time.time() > max_time:
            break
        improved = False
        
        # Compute all flip gains at once
        # gain[v] = sum_u adj[v,u] * (1 if labels[v]==labels[u] else -1)
        #        = sum_u adj[v,u] - 2 * sum_u adj[v,u] * (labels[v] != labels[u])
        #        = deg[v] - 2 * (adj @ (1 - labels))[v] if labels[v]==1
        #        = deg[v] - 2 * (adj @ labels)[v] if labels[v]==0
        
        # More directly:
        # For vertex v with label l:
        #   same_count = sum_u adj[v,u] * (labels[u] == l)
        #   opp_count = sum_u adj[v,u] * (labels[u] != l)
        #   gain = same_count - opp_count = 2*same_count - deg[v]
        
        # Vectorized:
        # same_count[v] = adj[v] @ (labels == labels[v])
        # We can compute: for each v, sum of adj[v] where labels match
        
        # Efficient approach:
        # match = (labels[:, None] == labels[None, :])  # n x n boolean
        # same_counts = adj @ match.sum(axis=0)  # This doesn't work directly
        
        # Better:
        # gain[v] = sum_u adj[v,u] * (2*labels[u] - 1) * (2*labels[v] - 1)
        # Let s = 2*labels - 1 (±1 vector)
        # gain[v] = s[v] * (adj @ s)[v]
        
        s = 2 * labels - 1
        adj_s = adj @ s
        gains = s * adj_s
        
        # Find the vertex with the largest positive gain
        max_gain_idx = np.argmax(gains)
        if gains[max_gain_idx] > 0:
            labels[max_gain_idx] = 1 - labels[max_gain_idx]
            improved = True
    
    return labels


def _local_search_2swap(adj, labels, max_time=None):
    """2-swap local search: flip pairs of vertices with different labels."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    improved = True
    while improved:
        if time.time() > max_time:
            break
        improved = False
        
        # Compute flip gains for all vertices
        s = 2 * labels - 1
        adj_s = adj @ s
        flip_gains = s * adj_s
        
        # For 2-swap of i and j (labels[i] != labels[j]):
        # gain = flip_gains[i] + flip_gains[j] - 2 * adj[i,j]
        
        # We want to find the pair (i,j) with labels[i]!=labels[j] that maximizes this.
        # Let's compute a matrix of 2-swap gains for all pairs.
        
        # 2swap_gain[i,j] = flip_gains[i] + flip_gains[j] - 2*adj[i,j]
        # We only consider pairs where labels differ.
        
        # Efficient approach: for each vertex i, find the best j with different label.
        # For fixed i, gain(i,j) = flip_gains[i] + flip_gains[j] - 2*adj[i,j]
        # = flip_gains[i] + (flip_gains[j] - 2*adj[i,j])
        
        # Let's iterate:
        for i in range(n):
            if time.time() > max_time:
                return labels
            
            # Find best j with labels[j] != labels[i]
            opp_mask = (labels != labels[i])
            if not np.any(opp_mask):
                continue
            
            # gain(i,j) for j in opp_mask:
            # = flip_gains[i] + flip_gains[j] - 2*adj[i,j]
            # We want to maximize over j in opp_mask.
            
            # Create a vector of gains for all j, set same-label ones to -inf
            j_gains = flip_gains.copy()
            j_gains[labels == labels[i]] = -np.inf
            j_gains[i] = -np.inf  # i != j
            
            # Subtract 2*adj[i,:]
            j_gains -= 2 * adj[i, :]
            
            best_j = np.argmax(j_gains)
            if j_gains[best_j] > 0:
                labels[i] = 1 - labels[i]
                labels[best_j] = 1 - labels[best_j]
                improved = True
                break  # Restart since labels changed
        
    return labels


def _full_local_search(adj, labels, max_time=None):
    """Alternate between 1-flip and 2-swap local search."""
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    prev_cut = -1
    current_cut = _cut_size(adj, labels)
    
    for _ in range(100):
        if time.time() > max_time:
            break
        
        # 1-flip
        labels = _local_search_1flip(adj, labels, max_time)
        new_cut = _cut_size(adj, labels)
        
        if new_cut > current_cut:
            current_cut = new_cut
            prev_cut = new_cut
            continue
        
        # 2-swap
        labels = _local_search_2swap(adj, labels, max_time)
        new_cut = _cut_size(adj, labels)
        
        if new_cut <= current_cut:
            break
        
        current_cut = new_cut
    
    return labels


def _tabu_search(adj, labels, tabu_len=7, max_iters=500, max_time=None):
    """Cerny tabu search for maximum cut."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    current_cut = best_cut
    
    # Tabu list: stores iteration until which vertex is tabu
    tabu_until = np.zeros(n, dtype=int)
    
    for it in range(max_iters):
        if time.time() > max_time:
            break
        
        # Compute all flip gains
        s = 2 * labels - 1
        adj_s = adj @ s
        gains = s * adj_s
        
        # Find best non-tabu move
        best_gain = -np.inf
        best_v = -1
        
        for v in range(n):
            is_tabu = (it < tabu_until[v])
            
            # Aspiration: accept tabu move if it improves best
            if is_tabu and gains[v] > best_cut:
                is_tabu = False
            
            if is_tabu:
                continue
            
            if gains[v] > best_gain:
                best_gain = gains[v]
                best_v = v
        
        if best_v == -1:
            # All tabu, pick best anyway
            for v in range(n):
                if gains[v] > best_gain:
                    best_gain = gains[v]
                    best_v = v
            if best_v == -1:
                break
        
        # Perform flip
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = it + tabu_len
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_labels = None
    best_cut = -1
    
    # Generate diverse initializations
    inits = []
    
    # 1. Spectral initialization
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # 2. ADMM initialization
    if time.time() < deadline - 30:
        admm_labels = _admm_maxcut(adj, n, max_iter=100, max_time=deadline - 30)
        inits.append(admm_labels)
    
    # 3. Alternating pattern
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # 4. Random initializations
    np.random.seed(42)
    for _ in range(8):
        rand_labels = np.random.randint(0, 2, n)
        inits.append(rand_labels)
    
    # For each initialization, run local search + tabu
    for init_idx, init_labels in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        labels = init_labels.copy()
        
        # Local search to local optimum
        labels = _full_local_search(adj, labels, max_time=deadline - 10)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Tabu search from local optimum
        if time.time() < deadline - 10:
            ts_labels, ts_cut = _tabu_search(adj, labels.copy(), tabu_len=7, 
                                             max_iters=200, max_time=deadline - 10)
            
            # Final local search on tabu result
            ts_labels = _full_local_search(adj, ts_labels, max_time=deadline - 5)
            ts_cut = _cut_size(adj, ts_labels)
            
            if ts_cut > best_cut:
                best_cut = ts_cut
                best_labels = ts_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
