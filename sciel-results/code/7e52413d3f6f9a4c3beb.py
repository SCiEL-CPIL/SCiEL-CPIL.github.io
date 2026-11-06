"""Maximum Cut: SDP relaxation via projected gradient + hyperplane rounding + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _cut_size(labels, adj):
    """Compute cut size for given 0/1 labels."""
    s = 2.0 * labels - 1.0
    E = 0.5 * np.sum(adj)
    sAs = s @ (adj @ s)
    return E - 0.5 * sAs

def _greedy_improve(labels, adj, deadline):
    """First-improvement local search: flip vertices if it improves cut."""
    curr_labels = labels.copy()
    curr_cut = _cut_size(curr_labels, adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s = 2.0 * curr_labels - 1.0
        As = adj @ s
        deltas = s * As
        pos_indices = np.where(deltas > 1e-9)[0]
        if len(pos_indices) > 0:
            best_v = pos_indices[np.argmax(deltas[pos_indices])]
            curr_labels[best_v] = 1 - curr_labels[best_v]
            curr_cut += deltas[best_v]
            improved = True
    return curr_labels, curr_cut

def _simulated_annealing(labels, adj, deadline, T0=10.0, alpha=0.995):
    """Simulated annealing to escape local optima."""
    curr_labels = labels.copy().astype(np.float64)
    curr_cut = _cut_size(curr_labels.astype(int), adj)
    T = T0
    n = len(labels)
    
    while time.time() < deadline and T > 0.01:
        v = random.randint(0, n - 1)
        s = 2.0 * curr_labels - 1.0
        As = adj @ s
        delta = s[v] * As[v]
        
        if delta > 0 or random.random() < np.exp(delta / T):
            curr_labels[v] = 1.0 - curr_labels[v]
            curr_cut += delta
        
        T *= alpha
    
    curr_labels_int = (curr_labels >= 0.5).astype(int)
    curr_labels_int, curr_cut = _greedy_improve(curr_labels_int, adj, deadline)
    return curr_labels_int, curr_cut

def _spectral_init(adj, n):
    """Spectral initialization using Fiedler vector."""
    A_sp = sp.csr_matrix(adj)
    D = np.sum(adj, axis=1)
    L = sp.diags(D) - A_sp
    
    try:
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
        labels = (fiedler_vec >= 0).astype(int)
        if np.all(labels == 0) or np.all(labels == 1):
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2

def _sdp_projected_gradient(adj, n, deadline, n_iter=2000):
    """
    Solve the Goemans-Williamson SDP relaxation via projected gradient ascent.
    Maximize Tr(X L) subject to X >= 0, diag(X) = 1.
    L = D - A is the graph Laplacian.
    
    The objective Tr(X L) = sum_{i,j} L_{ij} X_{ij} = 2 * sum_{(i,j) in E} (1 - X_{ij})
    Wait, let me reconsider. For max cut, the SDP is:
    Maximize sum_{(i,j) in E} (1 - x_i . x_j) / 2 = |E|/2 - (1/4) sum_{(i,j) in E} x_i . x_j
    Equivalently: Maximize Tr(X (D - A)) where X = [x_i . x_j], diag(X)=1, X PSD.
    Actually the standard form: maximize (1/4) sum_{i,j} (D_{ii} delta_{ij} - A_{ij}) X_{ij}
    which is (1/4) Tr(X L) where L = D - A.
    
    So we maximize Tr(X L) subject to X PSD, diag(X) = 1.
    Gradient of Tr(X L) w.r.t. X is L (symmetric).
    Projected gradient ascent: X_{k+1} = proj(X_k + eta * L)
    Projection onto {X PSD, diag(X)=1}: set diagonal to 1, project onto PSD cone.
    """
    D = np.sum(adj, axis=1)
    L = np.diag(D) - adj
    L = 0.5 * (L + L.T)  # ensure symmetric
    
    # Initialize X as identity (satisfies diag=1, PSD)
    X = np.eye(n)
    
    # Adaptive step size
    eta = 1.0 / (2.0 * np.max(np.abs(np.diag(L))) + 1e-6)
    eta = min(eta, 0.5)
    
    best_X = X.copy()
    best_obj = np.trace(X @ L)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Gradient ascent step
        grad = L  # gradient of Tr(XL) is L
        X_new = X + eta * grad
        
        # Project: set diagonal to 1
        np.fill_diagonal(X_new, 1.0)
        X_new = 0.5 * (X_new + X_new.T)  # symmetrize
        
        # Project onto PSD cone
        try:
            eigvals, eigvecs = np.linalg.eigh(X_new)
            eigvals_clipped = np.maximum(eigvals, 0.0)
            X_new = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        except np.linalg.LinAlgError:
            X_new = X  # keep old if projection fails
        
        # Re-fix diagonal after PSD projection (it may have drifted)
        np.fill_diagonal(X_new, 1.0)
        X_new = 0.5 * (X_new + X_new.T)
        
        obj = np.trace(X_new @ L)
        if obj > best_obj:
            best_obj = obj
            best_X = X_new.copy()
        
        # Adaptive step: if objective decreased, shrink eta
        if obj < best_obj - 1e-6:
            eta *= 0.5
        else:
            # slowly increase eta
            eta = min(eta * 1.01, 1.0 / (2.0 * np.max(np.abs(np.diag(L))) + 1e-6))
        
        X = X_new
    
    return best_X

def _hyperplane_rounding(X, n, n_trials=30, deadline=None):
    """
    Randomized hyperplane rounding: pick random unit vector u,
    label_i = 0 if x_i . u >= 0 else 1.
    Returns list of label arrays.
    """
    labels_list = []
    rng = np.random.RandomState(789)
    
    # Extract rows of X (each row is x_i)
    X_rows = X  # X[i] is x_i
    
    for t in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        u = rng.randn(n)
        u /= (np.linalg.norm(u) + 1e-12)
        
        projections = X_rows @ u
        labels = (projections >= 0).astype(int)
        
        if not (np.all(labels == 0) or np.all(labels == 1)):
            labels_list.append(labels)
    
    if not labels_list:
        labels_list.append(np.random.randint(0, 2, n))
    
    return labels_list

def _pairwise_swap_search(labels, adj, deadline):
    """Try pairwise swaps (flip two vertices) to escape 1-flip local optima."""
    curr_labels = labels.copy()
    curr_cut = _cut_size(curr_labels, adj)
    n = len(curr_labels)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s = 2.0 * curr_labels - 1.0
        As = adj @ s
        deltas = s * As
        # For a pair (i,j), the change in cut is:
        # delta_pair = -deltas[i] - deltas[j] + 2*s[i]*s[j]*adj[i,j]
        # Actually let's compute it properly.
        # Flipping i changes cut by deltas[i]. Flipping j changes by deltas[j].
        # But if both are flipped, the interaction term matters.
        # Let's just try: for each pair, compute the actual cut change.
        # For efficiency, try a random subset of pairs.
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        n_pairs_to_try = min(500, n * (n - 1) // 2)
        tried = set()
        for _ in range(n_pairs_to_try):
            if time.time() > deadline:
                break
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i == j or (min(i,j), max(i,j)) in tried:
                continue
            tried.add((min(i,j), max(i,j)))
            
            # Compute cut change for flipping both i and j
            # Original contribution of edges involving i or j
            # New labels: s[i] -> -s[i], s[j] -> -s[j]
            # Change in sum_{k} s[i]*s[k]*A[i,k] for edges from i:
            # Old: s[i] * sum_k s[k] A[i,k] = s[i] * As[i]
            # New: -s[i] * (sum_{k!=j} s[k] A[i,k] + (-s[j]) A[i,j])
            #      = -s[i] * (As[i] - s[j]*A[i,j] - s[j]*A[i,j]) ... this is getting complex
            # Let's just compute directly for small n
            new_labels = curr_labels.copy()
            new_labels[i] = 1 - new_labels[i]
            new_labels[j] = 1 - new_labels[j]
            new_cut = _cut_size(new_labels, adj)
            if new_cut > curr_cut + 1e-9:
                curr_labels = new_labels
                curr_cut = new_cut
                improved = True
                break
    
    return curr_labels, curr_cut

def partition(adjacency):
    """
    Maximum Cut via SDP relaxation + hyperplane rounding + local search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: SDP relaxation via projected gradient
    sdp_deadline = start_time + 60.0  # Give SDP up to 60s
    X = _sdp_projected_gradient(adj, n, sdp_deadline, n_iter=5000)
    
    # Phase 2: Hyperplane rounding to get diverse initializations
    rounding_deadline = start_time + 80.0
    init_labels_list = _hyperplane_rounding(X, n, n_trials=50, deadline=rounding_deadline)
    
    # Add spectral init
    spec_labels = _spectral_init(adj, n)
    init_labels_list.append(spec_labels)
    
    # Add a couple random inits
    rng = np.random.RandomState(123)
    for _ in range(3):
        rand_labels = rng.randint(0, 2, n)
        init_labels_list.append(rand_labels)
    
    # Phase 3: For each initialization, run greedy + SA + pairwise swap
    for init_labels in init_labels_list:
        if time.time() > deadline - 10.0:
            break
        
        # Run greedy first
        greedy_labels, greedy_cut = _greedy_improve(init_labels, adj, deadline)
        
        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()
        
        # Run pairwise swap search
        if time.time() < deadline - 5.0:
            swap_labels, swap_cut = _pairwise_swap_search(greedy_labels, adj, deadline)
            if swap_cut > best_cut:
                best_cut = swap_cut
                best_labels = swap_labels.copy()
        
        # Run SA from best local result
        if time.time() < deadline - 5.0:
            sa_labels, sa_cut = _simulated_annealing(best_labels, adj, deadline)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    # Phase 4: Iterated local search with random perturbations
    if best_labels is not None and time.time() < deadline - 5.0:
        rng2 = np.random.RandomState(456)
        for _ in range(15):
            if time.time() > deadline - 3.0:
                break
            
            # Perturb best solution
            perturbed = best_labels.copy()
            num_flips = rng2.randint(2, 6)
            for _ in range(num_flips):
                v = rng2.randint(0, n)
                perturbed[v] = 1 - perturbed[v]
            
            # Run greedy + pairwise swap
            g_labels, g_cut = _greedy_improve(perturbed, adj, deadline)
            if g_cut > best_cut:
                best_cut = g_cut
                best_labels = g_labels.copy()
            
            if time.time() < deadline - 3.0:
                s_labels, s_cut = _pairwise_swap_search(g_labels, adj, deadline)
                if s_cut > best_cut:
                    best_cut = s_cut
                    best_labels = s_labels.copy()
            
            if time.time() < deadline - 3.0:
                sa_labels, sa_cut = _simulated_annealing(g_labels, adj, deadline)
                if sa_cut > best_cut:
                    best_cut = sa_cut
                    best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
