"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)

    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False

    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]

    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]

    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)

    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))

    return int(delta_k + delta_self)


def _local_search(a, b, p, deadline, max_passes=100):
    """
    First-improvement local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break  # First improvement per i, then move to next i
        if not improved:
            break

    return p, cost


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align facility ordering with location ordering
    based on eigenvectors of the flow and distance matrices.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(10, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]

    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

    fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
    fac_norm[fac_norm == 0] = 1
    fac_emb = fac_emb / fac_norm

    loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
    loc_norm[loc_norm == 0] = 1
    loc_emb = loc_emb / loc_norm

    cost_matrix = -fac_emb @ loc_emb.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    if rng.random() < 0.3:
        subset_size = rng.integers(5, 20)
        subset = rng.choice(n, size=subset_size, replace=False)
        p[subset] = p[subset][::-1]

    return p


def _quantum_perturb(p, rng, n, temperature=0.5):
    """
    Quantum-inspired perturbation: probabilistic multi-swap moves.
    """
    perturbed_p = p.copy()

    base_swaps = 2
    max_swaps = 8
    n_swaps = base_swaps + int(temperature * (max_swaps - base_swaps))
    n_swaps = max(1, min(n_swaps, n // 2))

    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

    if rng.random() < 0.1 * temperature:
        block_len = rng.integers(3, min(10, n // 3))
        start = rng.integers(0, n - block_len + 1)
        perturbed_p[start:start + block_len] = perturbed_p[start:start + block_len][::-1]

    return perturbed_p


def _frank_wolfe(a, b, X_init, deadline, max_iters=500):
    """
    Frank-Wolfe (Conditional Gradient) method for QAP on Birkhoff polytope.
    
    The QAP cost in continuous form is:
        f(X) = sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
             = trace(X^T A X B^T)  [for the specific QAP formulation]
    
    More precisely:
        f(X) = sum_{i,j,k,l} a[i,j] * X[i,k] * b[k,l] * X[j,l]
             = sum_{k,l} b[k,l] * sum_{i,j} a[i,j] * X[i,k] * X[j,l]
             = sum_{k,l} b[k,l] * (X^T A X)[k,l]
             = <B, X^T A X>
             = trace(B^T X^T A X)
    
    Gradient:
        grad = (A + A^T) X B^T + (A^T + A) X B
        For symmetric A and B: grad = 2 * A X B
    
    Frank-Wolfe step:
        1. Compute gradient G
        2. Solve linear min over Birkhoff polytope: min_{Y in Birkhoff} <G, Y>
           This is the Hungarian algorithm on G (or -G for minimization)
        3. Update: X = (1 - gamma) * X + gamma * Y
           where gamma = 2/(t+2) or line search
    
    Returns: X (doubly stochastic matrix), p (extracted permutation)
    """
    n = X_init.shape[0]
    X = X_init.copy()
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    for t in range(1, max_iters + 1):
        if time.time() > deadline:
            break
        
        # Compute gradient: grad = 2 * A_sym @ X @ B_sym
        G = 2.0 * (a_sym @ X @ b_sym)
        
        # Solve linear problem over Birkhoff polytope: min <G, Y>
        # This is the assignment problem with cost matrix G
        row_ind, col_ind = linear_sum_assignment(G)
        Y = np.zeros((n, n), dtype=np.float64)
        Y[row_ind, col_ind] = 1.0
        
        # Frank-Wolfe step size
        gamma = 2.0 / (t + 2.0)
        
        # Update X
        X = (1.0 - gamma) * X + gamma * Y
        
        # Project back to Birkhoff polytope (Sinkhorn-Knopp)
        for _ in range(30):
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            X = X / col_sums
    
    # Extract permutation
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return X, p


def _frank_wolfe_with_line_search(a, b, X_init, deadline, max_iters=300):
    """
    Frank-Wolfe with backtracking line search for better convergence.
    """
    n = X_init.shape[0]
    X = X_init.copy()
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    def _fw_cost(X):
        return np.sum(b_sym * (X.T @ a_sym @ X))
    
    current_cost = _fw_cost(X)
    
    for t in range(1, max_iters + 1):
        if time.time() > deadline:
            break
        
        # Compute gradient
        G = 2.0 * (a_sym @ X @ b_sym)
        
        # Linear minimization over Birkhoff polytope
        row_ind, col_ind = linear_sum_assignment(G)
        Y = np.zeros((n, n), dtype=np.float64)
        Y[row_ind, col_ind] = 1.0
        
        # Backtracking line search
        gamma = 1.0
        best_cost = current_cost
        best_X = X.copy()
        
        for _ in range(20):
            X_trial = (1.0 - gamma) * X + gamma * Y
            # Project to Birkhoff
            for _ in range(20):
                rs = X_trial.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1
                X_trial = X_trial / rs
                cs = X_trial.sum(axis=0, keepdims=True)
                cs[cs == 0] = 1
                X_trial = X_trial / cs
            
            trial_cost = _fw_cost(X_trial)
            if trial_cost < best_cost:
                best_cost = trial_cost
                best_X = X_trial.copy()
            
            if trial_cost >= current_cost:
                gamma *= 0.5
            else:
                gamma *= 0.5  # Continue searching
            if gamma < 1e-6:
                break
        
        X = best_X
        current_cost = best_cost
    
    # Extract permutation
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return X, p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(12345)

    # Phase 1: Multiple initializations
    initializations = []
    
    # Init 1: Rank-matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    initializations.append(p1)
    
    # Init 2: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    initializations.append(p2)
    
    # Init 3: Spectral
    p3 = _spectral_init(a, b, n, rng)
    initializations.append(p3)
    
    # Init 4: Random
    p4 = rng.permutation(n).astype(np.int64)
    initializations.append(p4)
    
    # Phase 2: Frank-Wolfe optimization from each initialization
    for idx, p_init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Create initial doubly stochastic matrix
        X_init = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            X_init[i, p_init[i]] = 1.0
        
        # Run Frank-Wolfe
        fw_deadline = deadline - 15.0 - idx * 5.0
        X_fw, p_fw = _frank_wolfe(a, b, X_init, fw_deadline, max_iters=200)
        
        # Local search from FW result
        p_ls, c_ls = _local_search(a, b, p_fw, deadline - 10.0)
        
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()
    
    # Phase 3: Frank-Wolfe with line search from best solution
    if time.time() < deadline - 20.0 and best_p is not None:
        X_init = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            X_init[i, best_p[i]] = 1.0
        
        fw_deadline = deadline - 10.0
        X_fw, p_fw = _frank_wolfe_with_line_search(a, b, X_init, fw_deadline, max_iters=150)
        
        p_ls, c_ls = _local_search(a, b, p_fw, deadline - 5.0)
        
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()
    
    # Phase 4: Iterated Local Search with FW-based perturbations
    current_p = best_p.copy()
    current_cost = best_cost
    ilse_deadline = deadline - 5.0
    
    iteration = 0
    while time.time() < ilse_deadline:
        iteration += 1
        temperature = max(0.1, 1.0 - 0.05 * iteration)
        
        r = rng.random()
        if r < 0.3 and time.time() < ilse_deadline - 10.0:
            # FW-based perturbation: run few FW steps from current solution
            X_init = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                X_init[i, current_p[i]] = 1.0
            
            # Add noise to X
            noise_scale = 0.05 * temperature
            X_noisy = X_init + noise_scale * rng.standard_normal((n, n))
            X_noisy = np.maximum(X_noisy, 0)
            # Normalize to doubly stochastic
            for _ in range(50):
                rs = X_noisy.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1
                X_noisy = X_noisy / rs
                cs = X_noisy.sum(axis=0, keepdims=True)
                cs[cs == 0] = 1
                X_noisy = X_noisy / cs
            
            # Run few FW steps
            a_sym = (a + a.T) / 2.0
            b_sym = (b + b.T) / 2.0
            X = X_noisy
            for _ in range(10):
                if time.time() > ilse_deadline - 5.0:
                    break
                G = 2.0 * (a_sym @ X @ b_sym)
                row_ind, col_ind = linear_sum_assignment(G)
                Y = np.zeros((n, n), dtype=np.float64)
                Y[row_ind, col_ind] = 1.0
                gamma = 0.5
                X = (1.0 - gamma) * X + gamma * Y
                for _ in range(20):
                    rs = X.sum(axis=1, keepdims=True)
                    rs[rs == 0] = 1
                    X = X / rs
                    cs = X.sum(axis=0, keepdims=True)
                    cs[cs == 0] = 1
                    X = X / cs
            
            row_ind, col_ind = linear_sum_assignment(-X)
            perturbed_p = np.empty(n, dtype=np.int64)
            perturbed_p[row_ind] = col_ind
        elif r < 0.6:
            perturbed_p = _quantum_perturb(current_p, rng, n, temperature)
        else:
            k = 2 + int(temperature * 4)
            perturbed_p = current_p.copy()
            for _ in range(k):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]
        
        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ilse_deadline)
        
        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost < current_cost:
            delta = perturbed_cost - current_cost
            accept_prob = np.exp(-delta / (temperature * 1000.0))
            if rng.random() < accept_prob:
                current_p = perturbed_p
                current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost
    
    # Phase 5: Final polish
    if time.time() < deadline - 1.0 and best_p is not None:
        final_p = best_p.copy()
        final_p, final_cost = _local_search(a, b, final_p, deadline - 0.1, max_passes=10)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
