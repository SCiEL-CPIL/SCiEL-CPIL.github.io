"""
Frank-Wolfe (Conditional Gradient) solver for QAPLIB wil100.
Strategy:
1. Relax the permutation constraint to the Birkhoff polytope (doubly stochastic matrices).
2. Optimize the quadratic form X^T A X B (interpreted appropriately) using Frank-Wolfe.
   The gradient of the QAP objective f(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l] 
   with respect to X is G = A X B + B^T X^T A^T.
   Since A and B are symmetric in standard QAP, G = A X B + B X A.
3. The linear minimization oracle (vertex of Birkhoff polytope) is the Hungarian algorithm.
4. Discretize the resulting doubly stochastic matrix into a permutation.
5. Polish with vectorized Simulated Annealing.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _frank_wolfe_qap(a, b, max_iter=100, tol=1e-4, time_limit=60.0):
    """
    Frank-Wolfe algorithm for QAP on the Birkhoff polytope.
    Objective: min f(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
    Subject to: X in Birkhoff Polytope (doubly stochastic, non-negative).
    
    Note: Standard QAP is min sum A[i,j] D[p[i], p[j]].
    If X is assignment matrix where X[i,k]=1 if facility i at location k,
    then cost = sum_{i,j} A[i,j] sum_{k,l} B[k,l] X[i,k] X[j,l]
             = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
             = trace(X^T A X B^T) if A,B symmetric? 
             Let's check: X^T A X B. (X^T A X)_{k,l} = sum_i X[i,k] A[i,j] X[j,l]? No.
             (X^T A X)_{k,l} = sum_{i,j} X[i,k] A[i,j] X[j,l].
             Then trace((X^T A X) B^T) = sum_{k,l} (X^T A X)_{k,l} B[l,k]
             = sum_{k,l,i,j} X[i,k] A[i,j] X[j,l] B[l,k].
             This matches sum_{i,j,k,l} A[i,j] B[l,k] X[i,k] X[j,l].
             If B is symmetric, B[l,k]=B[k,l]. So f(X) = trace(X^T A X B).
             
    Gradient of f(X) = A X B + B^T X^T A^T.
    If A, B symmetric: G = A X B + B X A.
    """
    n = len(a)
    # Initialize X as a random doubly stochastic matrix or identity
    # Identity is a vertex. Better to start with a uniform matrix or random DS.
    # Let's start with a random permutation matrix (vertex) to ensure feasibility, 
    # but FW usually benefits from a non-vertex start if we project.
    # However, starting at a vertex is safe. Let's use a random permutation.
    p_init = np.random.permutation(n)
    X = np.zeros((n, n))
    X[np.arange(n), p_init] = 1.0
    
    # Precompute symmetric parts if not symmetric (though QAPLIB is usually symmetric)
    # Assume A and B are symmetric for gradient simplification, but handle generally.
    # General gradient of trace(X^T A X B):
    # f(X) = sum_{i,j,k,l} X[i,k] A[i,j] X[j,l] B[l,k]
    # df/dX[p,q] = sum_{j,l} A[p,j] X[j,l] B[l,q] + sum_{i,k} X[i,k] A[i,p] B[q,k]
    #             = (A X B)[p,q] + (B^T X^T A^T)[p,q]
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    start_time = time.time()
    best_cost = np.inf
    best_x = X.copy()
    
    for t in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute Gradient
        # G = A X B + B X^T A^T (if symmetric, G = A X B + B X A)
        # Using general form for safety:
        G = a @ X @ b + b.T @ X.T @ a.T
        
        # Linear Minimization Oracle:
        # min <G, Y> s.t. Y in Birkhoff
        # This is the Hungarian algorithm on matrix G.
        row_ind, col_ind = linear_sum_assignment(G)
        Y = np.zeros((n, n))
        Y[row_ind, col_ind] = 1.0
        
        # Check convergence (gap)
        # Gap = f(X) - <G, Y>
        # f(X) = trace(X.T @ A @ X @ B)
        f_x = np.trace(X.T @ a @ X @ b)
        lin_val = np.sum(G * Y)
        gap = f_x - lin_val
        
        if abs(gap) < tol:
            break
        
        # Step size (line search or diminishing)
        # Standard FW uses step size gamma_t = 2/(t+2)
        gamma = 2.0 / (t + 2.0)
        
        # Update
        X_new = (1 - gamma) * X + gamma * Y
        
        # Evaluate new cost to track best
        f_new = np.trace(X_new.T @ a @ X_new @ b)
        if f_new < best_cost:
            best_cost = f_new
            best_x = X_new.copy()
        
        X = X_new
        
    return best_x

def _discretize(x):
    """
    Convert a doubly stochastic matrix to a permutation.
    Uses Hungarian on the negative matrix to find the closest permutation.
    """
    row_ind, col_ind = linear_sum_assignment(-x)
    p = np.empty(len(x), dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _sa_polish(a, b, p, time_limit):
    """Simulated annealing polish phase with vectorized delta."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Cooling schedule
    t0 = 5000.0
    t_final = 0.1
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = elapsed / time_limit
        temp = t0 * (t_final / t0) ** frac
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Strategy 1: Frank-Wolfe Relaxation
    # We allocate a significant portion of time to FW to find a good basin.
    fw_time_limit = 40.0
    time_remaining = deadline - (time.time() - start)
    fw_limit = min(fw_time_limit, time_remaining * 0.5)
    
    if fw_limit > 5:
        x_relax = _frank_wolfe_qap(a, b, max_iter=200, time_limit=fw_limit)
        p_fw = _discretize(x_relax)
        cost_fw = _qap_cost(a, b, p_fw)
        
        if cost_fw < best_cost:
            best_cost = cost_fw
            best_p = p_fw.copy()
    
    # Strategy 2: Fallback/Comparison with Hungarian on Row Sums (Quick check)
    if best_p is None:
        # If FW failed or didn't produce a valid permutation (shouldn't happen), use simple heuristic
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        C = -np.outer(a_row_sums, b_row_sums)
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        cost_hung = _qap_cost(a, b, p_hung)
        best_p = p_hung.copy()
        best_cost = cost_hung

    # Strategy 3: SA Polish
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5:
        p_sa, cost_sa = _sa_polish(a, b, best_p.copy(), time_remaining - 2)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
            
    return best_p.tolist()
