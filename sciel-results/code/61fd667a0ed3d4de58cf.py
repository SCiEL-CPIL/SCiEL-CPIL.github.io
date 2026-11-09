# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _all_flip_gains(adj, labels):
    """Compute the gain from flipping each vertex, vectorized."""
    n = len(labels)
    s = 2 * labels - 1
    As = adj @ s
    gains = s * As
    return gains


def _local_search_1flip(adj, labels, deadline=None):
    """Improve partition by greedy 1-flip until no improvement."""
    n = len(labels)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        gains = _all_flip_gains(adj, labels)
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        improved = True
    return labels


def _two_swap_gain(adj, labels, i, j):
    """Compute the gain from swapping labels of vertices i and j."""
    li = labels[i]
    lj = labels[j]
    
    if li == lj:
        return 0.0
    
    s = 2 * labels - 1
    As = adj @ s
    
    sum_i = As[i]
    sum_i_excl_j = sum_i - adj[i, j] * s[j]
    
    sum_j = As[j]
    sum_j_excl_i = sum_j - adj[j, i] * s[i]
    
    change_i = s[i] * sum_i_excl_j
    change_j = s[j] * sum_j_excl_i
    
    total_change = change_i + change_j
    return total_change


def _local_search_2swap(adj, labels, deadline=None):
    """Improve partition by greedy 2-swap until no improvement (vectorized)."""
    n = len(labels)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        s = 2 * labels - 1
        As = adj @ s
        gains = s * As
        
        G = gains[:, np.newaxis] + gains[np.newaxis, :] + 2 * adj
        same_mask = (labels[:, np.newaxis] == labels[np.newaxis, :])
        G[same_mask] = 0
        np.fill_diagonal(G, 0)
        
        triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        G_search = np.where(triu_mask, G, -np.inf)
        max_gain = np.max(G_search)
        
        if max_gain > 0:
            flat_idx = np.argmax(G_search)
            i, j = divmod(flat_idx, n)
            labels[i], labels[j] = labels[j], labels[i]
            improved = True
    return labels


def _full_local_search(adj, labels, deadline=None):
    """Alternate 1-flip and 2-swap local search."""
    if deadline is None:
        deadline = time.time() + 10
    labels = _local_search_1flip(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_2swap(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_1flip(adj, labels, deadline)
    return labels


def _tabu_search(adj, labels, tabu_len=10, max_iter=500, deadline=None):
    """Tabu search with 1-flip moves and proper aspiration."""
    n = len(labels)
    if deadline is None:
        deadline = time.time() + 10
    
    current_labels = labels.copy()
    current_cut = _cut_size(adj, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    tabu_until = np.zeros(n, dtype=int)
    iteration = 0
    
    while iteration < max_iter:
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adj, current_labels)
        
        is_tabu = tabu_until > iteration
        allowed = ~is_tabu | (gains > best_cut)
        
        if not np.any(allowed):
            break
        
        gains_allowed = np.where(allowed, gains, -np.inf)
        best_v = np.argmax(gains_allowed)
        best_gain = gains_allowed[best_v]
        
        if best_gain == -np.inf:
            break
        
        current_labels[best_v] = 1 - current_labels[best_v]
        tabu_until[best_v] = iteration + tabu_len
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
        
        iteration += 1
    
    return best_labels


def _simulated_annealing(adj, labels, deadline=None):
    """Simulated annealing with batched 1-flip and 2-swap moves."""
    n = len(labels)
    if deadline is None:
        deadline = time.time() + 10
    
    current_labels = labels.copy()
    current_cut = _cut_size(adj, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    temperature = 8.0
    cooling = 0.97
    batch_size = 5
    
    while temperature > 0.01:
        if time.time() > deadline:
            break
        
        s = 2 * current_labels - 1
        As = adj @ s
        gains = s * As
        
        for _ in range(batch_size):
            v = np.random.randint(n)
            g = gains[v]
            if g > 0 or np.random.random() < np.exp(g / temperature):
                current_labels[v] = 1 - current_labels[v]
                current_cut += g
                s = 2 * current_labels - 1
                As = adj @ s
                gains = s * As
        
        if np.random.random() < 0.4:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j and current_labels[i] != current_labels[j]:
                gain2 = _two_swap_gain(adj, current_labels, i, j)
                if gain2 > 0 or np.random.random() < np.exp(gain2 / temperature):
                    current_labels[i], current_labels[j] = current_labels[j], current_labels[i]
                    current_cut += gain2
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
        
        temperature *= cooling
    
    return best_labels


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


def _continuous_relaxation(adj, n, max_iters=200, deadline=None):
    """
    Subgradient ascent on the continuous relaxation of max-cut.
    
    We maximize f(w) = w^T A w / 4 over w in [-1, 1]^n.
    The gradient is grad_f(w) = A @ w / 2.
    
    We use projected subgradient ascent with adaptive step size,
    then hard-threshold to get a binary partition.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    # Initialize with spectral method for good starting point
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        w = eigvecs[:, idx[1]].copy()
        # Normalize to [-1, 1]
        w_max = np.max(np.abs(w))
        if w_max > 0:
            w = w / w_max
    except:
        w = np.random.randn(n)
        w = np.clip(w, -1, 1)
    
    best_w = w.copy()
    best_val = w @ adj @ w
    
    # Adaptive step size
    eta = 1.0
    momentum = np.zeros(n)
    beta = 0.9  # Momentum coefficient
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gradient: grad = A @ w
        grad = adj @ w
        
        # Momentum update
        momentum = beta * momentum + (1 - beta) * grad
        momentum_norm = np.linalg.norm(momentum)
        if momentum_norm > 1e-10:
            momentum = momentum / momentum_norm
        
        # Adaptive step size: decrease if we're oscillating
        w_new = w + eta * momentum
        w_new = np.clip(w_new, -1, 1)
        
        # Evaluate
        val_new = w_new @ adj @ w_new
        
        if val_new > best_val:
            best_val = val_new
            best_w = w_new.copy()
        
        # Update step size: if we made progress, keep eta; else decrease
        if val_new > w @ adj @ w:
            w = w_new
        else:
            eta *= 0.95
            w = w_new  # Still take the step but reduce eta for next time
        
        # Also try a few different step sizes from current w
        for eta_try in [eta * 0.5, eta * 2.0]:
            w_try = np.clip(w + eta_try * momentum, -1, 1)
            val_try = w_try @ adj @ w_try
            if val_try > best_val:
                best_val = val_try
                best_w = w_try.copy()
    
    # Hard threshold to get binary partition
    labels = (best_w > 0).astype(int)
    
    # Ensure balanced partition if needed
    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)
    if n0 == 0 or n1 == 0:
        # Degenerate case; use sign of mean-zero projection
        w_centered = best_w - np.mean(best_w)
        labels = (w_centered > 0).astype(int)
    
    return labels


def _admm_maxcut(adj, n, max_iters=300, deadline=None):
    """
    ADMM-based approach for max-cut.
    
    Solve: maximize -1/4 * s^T (I - A) s subject to s_i in {-1, 1}
    
    Using ADMM with variable splitting:
    s = z, minimize f(s) + g(z) where f(s) = -1/4 * s^T (I-A) s, g(z) = 0 if z in {-1,1}^n else inf
    """
    if deadline is None:
        deadline = time.time() + 10
    
    # Initialize
    s = np.random.choice([-1.0, 1.0], size=n)
    z = s.copy()
    u = np.zeros(n)  # Dual variable
    
    rho = 1.0  # Penalty parameter
    best_s = s.copy()
    best_val = s @ adj @ s
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # s-update: minimize -1/4 * s^T (I-A) s + rho/2 * ||s - z + u||^2
        # Gradient: -(I-A)s/2 + rho*(s - z + u) = 0
        # (rho*I - (I-A)/2) s = rho*(z - u)
        # (rho - 0.5)*I + A/2) s = rho*(z - u)
        
        M = (rho - 0.5) * np.eye(n) + 0.5 * adj
        rhs = rho * (z - u)
        
        try:
            s_new = np.linalg.solve(M, rhs)
        except:
            s_new = z - u
        
        # z-update: proximal operator of g(z) = 0 if z in {-1,1}^n else inf
        # z_new = sign(s_new + u)
        z_new = np.sign(s_new + u)
        # Handle zeros: assign randomly
        zero_mask = z_new == 0
        if np.any(zero_mask):
            z_new[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
        
        # Dual update
        u = u + s_new - z_new
        
        s = s_new
        z = z_new
        
        # Track best integer solution
        val = z @ adj @ z
        if val > best_val:
            best_val = val
            best_s = z.copy()
        
        # Adaptive rho
        if it % 10 == 9:
            residual_primal = np.linalg.norm(s - z)
            residual_dual = rho * np.linalg.norm(z_new - z)
            if residual_primal > 10 * residual_dual:
                rho *= 2
            elif residual_dual > 10 * residual_primal:
                rho *= 0.5
    
    return best_s.astype(int)


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Generate initial solutions
    inits = []
    
    # Spectral init
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Continuous relaxation init
    cont_labels = _continuous_relaxation(adj, n, max_iters=150, deadline=deadline)
    inits.append(cont_labels)
    
    # ADMM init
    admm_labels = _admm_maxcut(adj, n, max_iters=200, deadline=deadline)
    inits.append(admm_labels)
    
    # Alternating init
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # Random inits
    np.random.seed(42)
    for _ in range(4):
        rand_labels = np.random.randint(0, 2, n)
        inits.append(rand_labels)
    
    for init_labels in inits:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # Full local search (1-flip + 2-swap)
        labels = _full_local_search(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Tabu search from the locally-optimal solution
        if time.time() > deadline:
            break
        tabu_labels = _tabu_search(adj, labels, tabu_len=12, max_iter=400, deadline=deadline)
        tabu_cut = _cut_size(adj, tabu_labels)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
        
        # Simulated annealing from the best so far
        if time.time() > deadline:
            break
        sa_labels = _simulated_annealing(adj, best_labels.copy(), deadline=deadline)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
