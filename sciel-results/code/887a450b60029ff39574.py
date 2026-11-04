# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta_vec(a, b, p, i, j):
    """Compute exact delta for swapping facilities at positions i and j."""
    n = len(p)
    pi = p[i]
    pj = p[j]
    
    # Vectorized computation of the four sums
    diff_row = b[pj, p] - b[pi, p]
    diff_col = b[p, pj] - b[p, pi]
    
    S1 = int(np.dot(a[i, :], diff_row))
    S2 = int(np.dot(a[j, :], -diff_row))
    S3 = int(np.dot(a[:, i], diff_col))
    S4 = int(np.dot(a[:, j], -diff_col))
    
    total = S1 + S2 + S3 + S4
    
    b_pi_pi = b[pi, pi]
    b_pj_pj = b[pj, pj]
    b_pi_pj = b[pi, pj]
    b_pj_pi = b[pj, pi]
    
    Rem_ii = a[i, i] * (b_pj_pi - b_pi_pi + b_pi_pj - b_pi_pi)
    Rem_ij = a[i, j] * (b_pj_pj - b_pi_pj + b_pi_pi - b_pi_pj)
    Rem_ji = a[j, i] * (b_pi_pi - b_pj_pi + b_pj_pj - b_pj_pi)
    Rem_jj = a[j, j] * (b_pi_pj - b_pj_pj + b_pj_pi - b_pj_pj)
    
    Rem = Rem_ii + Rem_ij + Rem_ji + Rem_jj
    Direct = a[i, j] * (b_pj_pi - b_pi_pj) + a[j, i] * (b_pi_pj - b_pj_pi)
    
    return total - Rem + Direct

def _local_search(a, b, p, deadline, best_cost, best_p):
    """Fast 2-opt first-improvement local search."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta_vec(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p

def _frank_wolfe(a, b, deadline, rng):
    """
    Frank-Wolfe (Conditional Gradient) on the Birkhoff polytope.
    Minimizes F(X) = sum_{i,j} A_ij X_ij^2 where A = a*b (element-wise).
    """
    n = len(a)
    A = a * b  # Element-wise product matrix for the quadratic term
    
    # Initialize with a random doubly stochastic matrix
    # Start with identity, then mix with random permutation matrices
    X = np.eye(n)
    # Add some noise to break symmetry
    for _ in range(3):
        perm = rng.permutation(n)
        X += np.eye(n)[perm]
    X /= X.sum() # Normalize rows? No, X is a sum of permutation matrices, so it's already DS if scaled properly?
    # Actually, a sum of permutation matrices is DS.
    # Let's just start with a random DS matrix properly.
    # Simple way: Start with all 1/n, then project? No, 1/n is DS.
    # Let's start with 1/n + small noise projected?
    # Better: Start with identity, then do a few FW steps.
    
    X = np.eye(n)
    
    # Gradient of F(X) = sum A_ij X_ij^2 is G = 2 * A * X
    # We want to minimize F.
    
    step_size = 0.0
    last_improve = time.time()
    best_X = X.copy()
    best_val = np.sum(A * X * X)
    
    while time.time() < deadline:
        G = 2.0 * A * X
        
        # Linear minimization problem: min_{Y in Birkhoff} <G, Y>
        # Equivalent to assignment problem with cost matrix G.
        # scipy.optimize.linear_sum_assignment minimizes sum C_ij X_ij
        row_ind, col_ind = linear_sum_assignment(G)
        Y = np.zeros((n, n))
        Y[row_ind, col_ind] = 1.0
        
        # Check if Y is the same as X (convergence)
        if np.allclose(X, Y):
            break
            
        # Update X = (1 - gamma) * X + gamma * Y
        # Standard FW uses gamma_k = 2 / (k + 2)
        step_size += 1
        gamma = 2.0 / (step_size + 2.0)
        
        X_new = (1.0 - gamma) * X + gamma * Y
        
        # Evaluate objective
        val = np.sum(A * X_new * X_new)
        
        if val < best_val:
            best_val = val
            best_X = X_new.copy()
            last_improve = time.time()
            
        X = X_new
        
        # Early stopping if no improvement for a while?
        # The step size decreases, so it should converge.
        # For QAP, we might get stuck in a local min of the relaxed problem,
        # but the landscape is different from the discrete one.
        
    return best_X

def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill

    rng = np.random.default_rng(12345)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # 1. Initial Discrete Solution
    # Use Hungarian on the sum of rows as a heuristic start
    row_sums_a = a.sum(axis=1)
    col_sums_b = b.sum(axis=1)
    
    # Create a cost matrix for initial assignment
    # Heuristic: match high flow facilities to low distance locations?
    # Or use the FW result directly as the first start?
    
    # Let's run FW for a short time to get a good continuous start
    # Then round it to a permutation.
    
    fw_time = min(10.0, (deadline - t0) * 0.1)
    X_frank = _frank_wolfe(a, b, t0 + fw_time, rng)
    
    # Convert X_frank to a permutation
    # Minimize -<X_frank, P> or similar?
    # We want to find the permutation matrix P closest to X_frank.
    # This is equivalent to maximizing sum X_frank[i, p[i]].
    # Which is minimizing sum -X_frank[i, p[i]].
    
    row_ind, col_ind = linear_sum_assignment(-X_frank)
    p_fw = np.empty(n, dtype=np.int64)
    p_fw[row_ind] = col_ind
    
    c_fw = _qap_cost(a, b, p_fw)
    if c_fw < best_cost:
        best_cost = c_fw
        best_p = p_fw.copy()
        
    # Also try a standard Hungarian on a heuristic matrix
    # cost[i, j] = a.sum(axis=1)[i] * b.sum(axis=1)[j] ?? No.
    # Let's just use the FW result. It's a structural change.
    
    # 2. Polish the FW-derived solution
    p = best_p.copy()
    polish_time = min(15.0, (deadline - t0) * 0.2)
    best_cost, best_p = _local_search(a, b, p, t0 + polish_time, best_cost, best_p)
    
    # 3. Diversification: Run FW again with a different random seed/init and polish
    # Or run FW on a perturbed A?
    # Let's run multiple FW instances if time permits.
    
    while time.time() < deadline - 5.0:
        # Perturb the current best solution to create a new continuous start?
        # Or just run FW from scratch with a different random seed?
        # FW is deterministic given the init. Let's randomize the init.
        
        # Random Init for FW
        X_init = np.eye(n)
        for _ in range(5):
            perm = rng.permutation(n)
            X_init += np.eye(n)[perm]
        X_init /= X_init.sum() # This is not strictly DS if not normalized correctly?
        # Sum of permutation matrices is DS.
        # But we want a random point in the polytope.
        # A uniform random point in Birkhoff polytope is hard to sample.
        # Let's just use the identity + noise approach or just restart FW with different seeds.
        
        # Actually, FW is quite fast for n=100.
        # Let's run FW for a fixed duration, get a permutation, polish it.
        
        fw_deadline = time.time() + 5.0
        X_next = _frank_wolfe(a, b, fw_deadline, rng)
        
        row_ind, col_ind = linear_sum_assignment(-X_next)
        p_next = np.empty(n, dtype=np.int64)
        p_next[row_ind] = col_ind
        
        c_next = _qap_cost(a, b, p_next)
        
        if c_next < best_cost:
            # Polish it
            p_to_polish = p_next.copy()
            polish_deadline = min(time.time() + 5.0, deadline - 2.0)
            best_cost, best_p = _local_search(a, b, p_to_polish, polish_deadline, best_cost, best_p)
        else:
            # If FW didn't find something better immediately, maybe polish the FW solution anyway?
            # It might be in a different basin.
            if time.time() < deadline - 2.0:
                p_to_polish = p_next.copy()
                polish_deadline = min(time.time() + 2.0, deadline - 1.0)
                new_cost, new_p = _local_search(a, b, p_to_polish, polish_deadline, best_cost, best_p)
                # _local_search returns best so far, so if it improved, best_cost is updated.
                # We need to be careful: _local_search updates best_cost/best_p in place (conceptually)
                # but in my implementation it returns them.
                # Let's ensure we use the returned values.
                best_cost, best_p = new_cost, new_p

    return best_p.tolist()

# EVOLVE-BLOCK-END
