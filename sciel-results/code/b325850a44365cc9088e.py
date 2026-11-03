"""Frank-Wolfe / Birkhoff relaxation for QAP tai256c, with Hungarian projection and 2-opt polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the exact integer QAP cost for permutation p."""
    n = len(p)
    # b_p = b[p[:, None], p[None, :]] is the permuted distance matrix
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _frank_wolfe_qap(a, b, init_perm, max_iters=300, time_budget=100.0):
    """
    Frank-Wolfe (conditional gradient) on the Birkhoff polytope.
    We minimize f(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    over doubly stochastic X.
    
    The gradient of f w.r.t. X[i,k] is:
    grad[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    (since f is quadratic and symmetric in the two indices for the cost structure)
    
    Actually, let's be more careful. The cost is:
    C(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    
    grad_{i,k} C = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    
    This is because X[i,k] appears in terms where it's the first factor (i,k) and 
    where it's the second factor (j,l) with j=i, l=k.
    
    At each iteration:
    1. Compute gradient G
    2. Solve linear assignment on G to get vertex V (a permutation matrix)
    3. Line search: X_new = (1 - gamma) * X + gamma * V
    4. Project back to Birkhoff polytope (Hungarian on -X_new, or just keep convex combo)
       Actually, for Frank-Wolfe on Birkhoff, the step is X_{t+1} = (1-gamma_t) X_t + gamma_t V_t
       and we don't need to re-project because the convex combo of doubly stochastic matrices
       is doubly stochastic.
    """
    n = len(a)
    t_start = time.time()
    
    # Initialize X as the permutation matrix of init_perm
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), init_perm] = 1.0
    
    best_cost = _qap_cost(a, b, init_perm)
    best_perm = list(init_perm)
    
    prev_grad = None
    step_size = 1.0
    
    for it in range(max_iters):
        elapsed = time.time() - t_start
        if elapsed > time_budget:
            break
        
        # Compute gradient: grad[i,k] = sum_j a[i,j] * (B @ X)[j, k] ... let me re-derive
        # C = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
        # dC/dX[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
        
        # First term: for fixed i,k, sum_j a[i,j] * sum_l b[k,l] X[j,l] = sum_j a[i,j] (B X)[k,j]... 
        # Wait, let me define M = X @ B^T, so M[j,l] = sum_k X[j,k] B[k,l]? No.
        # Let's use matrix notation. Let S = X (n x n).
        # Term 1: (A @ S @ B)[i, k] where we need to check indices.
        # (A @ S)[i, l] = sum_j A[i,j] S[j,l]
        # (A @ S @ B)[i, k] = sum_l (A@S)[i,l] B[l,k] = sum_j,l A[i,j] S[j,l] B[l,k]
        # But we want sum_j,l A[i,j] B[k,l] S[j,l]. Note B[k,l] vs B[l,k].
        # If B is symmetric, B[k,l] = B[l,k], so Term 1 = (A @ S @ B)[i,k].
        # 
        # Term 2: sum_{j,l} A[j,i] B[l,k] S[j,l]
        # = sum_l B[l,k] sum_j A[j,i] S[j,l]
        # (A^T @ S^T)[l, i] = sum_j A^T[l,j] S^T[i,j] = sum_j A[j,l] S[j,i]... this is getting messy.
        # Let's just do it explicitly with numpy for clarity, then optimize if needed.
        
        # For n=256, explicit computation might be slow. Let's use matrix mults.
        # Assume A and B are symmetric (common in QAP). We'll handle general case too.
        
        # Term 1: T1[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l]
        # = sum_j A[i,j] * (sum_l B[k,l] X[j,l])
        # Let Y = X @ B^T, so Y[j,l'] = sum_l X[j,l] B[l,l']... no.
        # Y[j,k] = sum_l X[j,l] B[l,k] = (X @ B^T)[j,k] if B is not symmetric... 
        # Actually (X @ B)[j,k] = sum_l X[j,l] B[l,k]. So Y = X @ B.
        # Then T1[i,k] = sum_j A[i,j] Y[j,k] = (A @ Y)[i,k] = (A @ X @ B)[i,k].
        
        # Term 2: T2[i,k] = sum_{j,l} A[j,i] B[l,k] X[j,l]
        # = sum_l B[l,k] * (sum_j A[j,i] X[j,l])
        # Z[i,l] = sum_j A[j,i] X[j,l] = (A^T @ X^T)[l,i]^T... 
        # (A^T @ X)[l,i] = sum_j A^T[l,j] X[j,i] = sum_j A[j,l] X[j,i]. Not quite.
        # Let W = A^T @ X^T. W[l,i] = sum_j A^T[l,j] X^T[i,j] = sum_j A[j,l] X[j,i].
        # We want sum_j A[j,i] X[j,l]. That's (A^T @ X)[l,i]? 
        # (A^T @ X)[l,i] = sum_j A^T[l,j] X[j,i] = sum_j A[j,l] X[j,i]. 
        # We want sum_j A[j,i] X[j,l] = sum_j A^T[i,j] X[j,l] = (A^T @ X)[i,l].
        # So sum_j A[j,i] X[j,l] = (A^T @ X)[i,l].
        # Then T2[i,k] = sum_l B[l,k] * (A^T @ X)[i,l] = ((A^T @ X) @ B^T)[i,k].
        
        # So grad = A @ X @ B + (A^T @ X) @ B^T
        
        # For symmetric A and B: grad = A @ X @ B + A @ X @ B = 2 * A @ X @ B
        
        AX = a @ X
        if np.allclose(a, a.T) and np.allclose(b, b.T):
            grad = 2.0 * (AX @ b)
        else:
            ATX = a.T @ X
            grad = (AX @ b) + (ATX @ b.T)
        
        # Solve linear assignment on grad to get next vertex
        row_ind, col_ind = linear_sum_assignment(grad)
        V = np.zeros((n, n), dtype=np.float64)
        V[row_ind, col_ind] = 1.0
        
        # Line search (simple backtracking or fixed step)
        # For Frank-Wolfe, gamma_t = 2/(t+2) is a standard choice
        gamma = 2.0 / (it + 2.0)
        
        X_new = (1.0 - gamma) * X + gamma * V
        
        # Check convergence: if X_new is close to a permutation, extract it
        # Simple check: max deviation from 0/1
        # For now, just update X
        X = X_new
        
        # Periodically extract a permutation and evaluate
        if it % 5 == 0:
            # Extract permutation from X using Hungarian on -X
            r_idx, c_idx = linear_sum_assignment(-X)
            perm = np.empty(n, dtype=np.int64)
            perm[r_idx] = c_idx
            cost = _qap_cost(a, b, perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        
        # Also try the current vertex V as a permutation
        perm_v = np.empty(n, dtype=np.int64)
        perm_v[row_ind] = col_ind
        cost_v = _qap_cost(a, b, perm_v)
        if cost_v < best_cost:
            best_cost = cost_v
            best_perm = list(perm_v)
    
    return best_perm, best_cost


def _local_search_2opt(a, b, p, time_budget=20.0, max_passes=50):
    """2-opt local search: try swapping pairs of positions in the permutation."""
    n = len(p)
    p = list(p)
    best_cost = _qap_cost(a, b, p)
    t_start = time.time()
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if time.time() - t_start > time_budget:
            break
        
        for i in range(n):
            if time.time() - t_start > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - t_start > time_budget:
                    break
                
                # Swap p[i] and p[j]
                p[i], p[j] = p[j], p[i]
                new_cost = _qap_cost(a, b, p)
                if new_cost < best_cost:
                    best_cost = new_cost
                    improved = True
                else:
                    # Swap back
                    p[i], p[j] = p[j], p[i]
    
    return p, best_cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    total_time_budget = 130.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    p_init = p_init.tolist()
    
    # Phase 2: Frank-Wolfe relaxation
    fw_time_budget = 80.0
    p_fw, cost_fw = _frank_wolfe_qap(a, b, p_init, max_iters=500, time_budget=fw_time_budget)
    
    elapsed = time.time() - t_start
    remaining = total_time_budget - elapsed
    
    if remaining < 5.0:
        return p_fw
    
    # Phase 3: Local search polish
    ls_time_budget = min(remaining - 2.0, 40.0)
    p_ls, cost_ls = _local_search_2opt(a, b, p_fw, time_budget=ls_time_budget)
    
    return p_ls
# EVOLVE-BLOCK-END
