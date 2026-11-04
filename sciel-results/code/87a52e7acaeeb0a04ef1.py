"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _compute_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    n = len(p)
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    Uses vectorized delta calculation for all j > i at each step i.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            # diff_A = A_sym[i] - A_sym[js]  (shape: (n-i-1, n))
            # diff_D = D[js] - D[i]         (shape: (n-i-1, n))
            # S = sum(diff_A * diff_D, axis=1)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms for the diagonal and i-j interactions
            # The formula S includes k=i and k=j terms which are incorrect for the swap delta.
            # We need to subtract the contributions from k=i and k=j in the sum, 
            # and add the correct terms for the swap.
            
            # Term for k=i in sum:
            # (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # Term for k=j in sum:
            # (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms for the swap of i and j:
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            # cross_term: (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    The surrogate cost C[i, j] approximates the contribution of placing facility i at location j.
    C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    This is equivalent to (A @ B)[i, j] + (A.T @ B.T)[i, j]
    """
    n = len(a)
    # Calculate surrogate cost matrix
    cost_matrix = a @ b + a.T @ b.T
    
    # Solve assignment problem
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

def _momentum_init(a, b, time_limit, start_time):
    """
    Momentum-based continuous relaxation initialization.
    
    Formulates QAP as a discrete dynamical system on the permutation group where
    each facility holds a velocity vector in R^n. Positions are updated using a
    gradient-driven inertia term that allows facilities to "slide" through intermediate
    locations to escape local optima, rather than performing instantaneous pairwise swaps.
    
    The continuous relaxation uses a soft assignment matrix X where X[i,j] represents
    the probability that facility i is at location j. We minimize a smooth surrogate:
    F(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    subject to X being a doubly stochastic matrix (relaxation of permutation matrix).
    
    We use gradient descent with momentum on the log-domain to maintain positivity,
    then project to the Birkhoff polytope via Sinkhorn-Knopp or Hungarian.
    """
    n = len(a)
    
    # Initialize with a doubly stochastic matrix (uniform)
    # Use log-domain for numerical stability
    log_X = np.zeros((n, n), dtype=np.float64) / n  # uniform initialization
    
    # Precompute for gradient
    # F(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    # = sum_{i,j} a[i,j] * (X @ B @ X.T)[i,j]
    # Gradient: dF/dX[i,k] = sum_j a[i,j] * (X @ B)[j,k] + sum_j a[j,i] * (X @ B.T)[j,k]
    #           = (A @ (X @ B))[i,k] + (A.T @ (X @ B.T))[i,k]
    
    velocity = np.zeros((n, n), dtype=np.float64)
    lr = 0.01
    momentum = 0.9
    
    # Precompute A and A.T
    A = a.astype(np.float64)
    AT = a.T.astype(np.float64)
    B = b.astype(np.float64)
    BT = b.T.astype(np.float64)
    
    # Scale for numerical stability
    A_scale = np.max(np.abs(A))
    B_scale = np.max(np.abs(B))
    if A_scale > 0 and B_scale > 0:
        A = A / (A_scale * B_scale)
        AT = AT / (A_scale * B_scale)
        B = B / (A_scale * B_scale)
        BT = BT / (A_scale * B_scale)
    
    max_iters = 500
    for iter in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Convert log_X to X (softmax normalization per row and column for doubly stochastic)
        X = np.exp(log_X)
        # Row normalization
        X = X / X.sum(axis=1, keepdims=True)
        # Column normalization (Sinkhorn step)
        X = X / X.sum(axis=0, keepdims=True)
        # Renormalize rows
        X = X / X.sum(axis=1, keepdims=True)
        
        # Compute gradient
        XB = X @ B
        XBT = X @ BT
        
        grad = A @ XB + AT @ XBT
        
        # Update with momentum
        velocity = momentum * velocity - lr * grad
        log_X = log_X + velocity
        
        # Periodic re-normalization to prevent numerical issues
        if iter % 10 == 0:
            # Softmax with temperature to keep values in reasonable range
            log_X = log_X - np.max(log_X, axis=1, keepdims=True)
            log_X = log_X - np.max(log_X, axis=0, keepdims=True)
    
    # Convert final X to a permutation using Hungarian
    # Use -X as cost (we want to maximize X[i,j] where assignment is likely)
    row_ind, col_ind = opt.linear_sum_assignment(-X)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

def _spectral_init(a, b):
    """
    Phase-field-inspired initialization:
    1. Embed locations into 2D using MDS on distance matrix B.
    2. Embed facilities into 2D using MDS on flow matrix A (treated as similarity).
    3. Assign facilities to locations via Hungarian on Euclidean distance in embedding space.
    """
    n = len(a)
    
    # Embed locations: minimize sum_{ij} b[i,j] * ||x_i - x_j||^2
    # This is equivalent to finding the best low-dim representation of the distance matrix.
    # Using MDS with dissimilarities b.
    try:
        from scipy.spatial.distance import squareform
        # b is a distance matrix (symmetric, zero diagonal).
        # MDS wants dissimilarities.
        loc_coords = opt._linear_sum_assignment  # placeholder to ensure import
        # Use scipy.spatial.distance or manual MDS
        # Simpler: use the eigendecomposition of the centered kernel matrix.
        # Kernel K = -0.5 * J * B * J where J is centering matrix.
        # But B might not be negative type. Let's just use a simple heuristic or standard MDS.
        
        # Standard MDS implementation for 2D
        def mds(dissim, n_dim=2):
            n = len(dissim)
            J = np.eye(n) - np.ones((n, n)) / n
            M = -0.5 * J @ (dissim ** 2) @ J
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            # Take top n_dim
            V = eigenvectors[:, :n_dim]
            D = np.sqrt(np.maximum(eigenvalues[:n_dim], 0))
            coords = V * D
            return coords
        
        loc_coords = mds(b, n_dim=2)
        
        # For facilities, A is flow. We want facilities with high flow to be "close" in some sense?
        # Or rather, the structure of A dictates which facilities should be close.
        # If a[i,j] is high, facility i and j should be close in location space (small b).
        # So we can treat A as a similarity matrix.
        # Dissimilarity for MDS: 1 - A (normalized) or -A?
        # Let's use -A as dissimilarity for MDS (higher flow -> smaller distance).
        # Or simply use the eigenvectors of A directly as coordinates (Laplacian eigenmap style).
        # Let's stick to MDS with dissimilarity = -A (assuming A is non-negative).
        # If A has negative values, this might fail. Let's assume non-negative.
        # To be safe, let's use the row sums or a different embedding.
        
        # Alternative: Use the eigenvectors of the normalized Laplacian of A?
        # Let's try MDS with dissimilarity = -A.
        # Note: MDS requires dissimilarities. If we use -A, high A -> negative dissim -> "closer".
        # This is valid for metric MDS if -A is a valid dissimilarity matrix (which it's not strictly, but the algorithm just minimizes stress).
        
        fac_coords = mds(-a, n_dim=2)
        
        # Now assign facilities to locations to minimize sum of squared Euclidean distances
        # This is an assignment problem.
        # Cost matrix C[i, j] = ||fac_coords[i] - loc_coords[j]||^2
        cost_matrix = np.sum((fac_coords[:, np.newaxis, :] - loc_coords[np.newaxis, :, :]) ** 2, axis=2)
        
        row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=int)
        p[row_ind] = col_ind
        
        return p
        
    except Exception:
        # Fallback to random if MDS fails
        return np.random.permutation(n)

def _greedy_init(a, b, n):
    """
    Greedy constructive heuristic: build permutation by placing facilities 
    in order of increasing 'importance' (flow row sum) to locations with 
    matching 'importance' (distance row sum), but with some randomization.
    """
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by flow sum (ascending - place less important first)
    fac_order = np.argsort(flow_sums)
    # Sort locations by distance sum (ascending)
    loc_order = np.argsort(dist_sums)
    
    p = np.zeros(n, dtype=int)
    used_locs = set()
    
    for fac in fac_order:
        # Find the best unused location for this facility
        # Consider only unused locations, pick one with min distance sum
        candidates = [loc for loc in loc_order if loc not in used_locs]
        # Add some randomness: pick from top 5 candidates
        top_k = min(5, len(candidates))
        chosen = random.choice(candidates[:top_k])
        p[fac] = chosen
        used_locs.add(chosen)
    
    return p

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _polish_pass(a, b, p, time_limit, start_time):
    """
    High-precision polish: run local search with multiple restarts from 
    small perturbations of the current solution to escape shallow local optima.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Try a few small perturbations and local search
    for k in [1, 2, 3, 5]:
        if time.time() - start_time > time_limit:
            break
        for trial in range(3):
            if time.time() - start_time > time_limit:
                break
            p_pert = p.copy()
            for _ in range(k):
                i, j = random.sample(range(n), 2)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            # Give this LS run a small time budget
            remaining = time_limit - (time.time() - start_time)
            if remaining < 1:
                break
            ls_time = min(remaining * 0.3, 10)  # cap at 10s per polish LS
            p_ls, cost_ls = _local_search(a, b, p_pert, ls_time, start_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple initializations
    # Use a mix of Hungarian, Spectral (Phase-field), Greedy, and random starts
    
    # 1. Hungarian Initialization (structurally distinct, respects global correlation)
    p_hung = _hungarian_init(a, b)
    p_ls, cost_ls = _local_search(a, b, p_hung, time_budget * 0.12, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 1.5. Momentum-based continuous relaxation initialization
    if time.time() - start_time < time_budget * 0.22:
        p_mom = _momentum_init(a, b, time_budget * 0.05, start_time)
        p_ls, cost_ls = _local_search(a, b, p_mom, time_budget * 0.1, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # 2. Spectral / Phase-field Initialization
    if time.time() - start_time < time_budget * 0.32:
        p_spec = _spectral_init(a, b)
        p_ls, cost_ls = _local_search(a, b, p_spec, time_budget * 0.1, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # 3. Greedy Initialization
    if time.time() - start_time < time_budget * 0.42:
        p_greedy = _greedy_init(a, b, n)
        p_ls, cost_ls = _local_search(a, b, p_greedy, time_budget * 0.1, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # 4. Random Initials
    num_randoms = 3
    for _ in range(num_randoms):
        if time.time() - start_time > time_budget * 0.52:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_budget * 0.07, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: ILS
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 3:
        # Check time
        if time.time() - start_time > time_budget - 3:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 8:
            perturb_k = min(20, base_perturb + consecutive_no_improve // 2)
        elif consecutive_no_improve > 4:
            perturb_k = base_perturb + 2
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        duration = (time_budget - 3) - (time.time() - start_time)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Phase 3: Final polish pass with remaining time
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time > 5:
        polish_time = min(remaining_time - 2, 20)  # cap polish at 20s
        best_p, best_cost = _polish_pass(a, b, best_p, polish_time, start_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
