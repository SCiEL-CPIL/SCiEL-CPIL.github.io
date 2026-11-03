"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    def hungarian_solve(C):
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def project_birkhoff(X, iters=50):
        X = X.copy()
        X = np.maximum(X, 1e-10)
        for _ in range(iters):
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
        return X

    def run_pgd(X_init, max_time, lr_factor=1.0):
        X = X_init.copy()
        last_check = time.time()
        
        # Estimate gradient scale for learning rate
        # G = A X B. Norm approx ||A|| ||X|| ||B||.
        # ||X||_F = 1 (approx for doubly stochastic).
        # ||A||_F, ||B||_F.
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        grad_scale = norm_a * norm_b
        if grad_scale == 0:
            grad_scale = 1.0
        lr = (1.0 / (5.0 * grad_scale)) * lr_factor
        
        while time.time() < max_time:
            G = a @ X @ b
            X = X - lr * G
            X = project_birkhoff(X, iters=30)
            
            if time.time() - last_check > 3.0:
                last_check = time.time()
                p = hungarian_solve(G)
                return p, X
        
        G = a @ X @ b
        p = hungarian_solve(G)
        return p, X

    def local_search(p, max_time):
        current_cost = calc_cost(p)
        Bp = b[:, p]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Vectorized-ish 2-opt
            # We can speed this up by precomputing row sums or using matrix tricks,
            # but for n=150, python loops with numpy dot are ok-ish.
            # Let's try a faster approach:
            # Delta for swapping i, j:
            # d = (a[i] - a[j]) @ (Bp[j] - Bp[i])
            
            # We can compute all deltas at once?
            # Deltas[i, j] = sum_k (a[i,k] - a[j,k]) * (Bp[j,k] - Bp[i,k])
            # = a[i] @ Bp[j] - a[i] @ Bp[i] - a[j] @ Bp[j] + a[j] @ Bp[i]
            # Let M = a @ Bp. Then M[i,k] = a[i] @ Bp[k].
            # Delta(i,j) = M[i,j] - M[i,i] - M[j,j] + M[j,i]
            # This is O(n^3) to compute M, but M is n x n.
            # M = a @ Bp is O(n^3) = 150^3 = 3.3M ops. Very fast.
            
            M = a @ Bp
            # We want to find i < j such that Delta(i,j) < 0
            # Delta(i,j) = M[i,j] + M[j,i] - M[i,i] - M[j,j]
            # Let's create a matrix D where D[i,j] = M[i,j] + M[j,i]
            # Then subtract diagonal terms?
            # Actually, just compute D = M + M.T
            # Then Delta(i,j) = D[i,j] - M[i,i] - M[j,j]
            
            D = M + M.T
            diag_M = np.diag(M)
            
            # We want to find i < j minimizing D[i,j] - diag_M[i] - diag_M[j]
            # Create a matrix C where C[i,j] = D[i,j] - diag_M[i] - diag_M[j]
            # C = D - diag_M[:, None] - diag_M[None, :]
            
            C = D - diag_M[:, None] - diag_M[None, :]
            
            # Mask out i >= j and i == j
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            C_masked = np.where(mask, C, 0) # 0 is not negative, so it won't be picked if best_delta < 0
            
            # Find min
            min_val = np.min(C_masked)
            if min_val < 0:
                idx = np.unravel_index(np.argmin(C_masked), C_masked.shape)
                i, j = idx
                # Apply swap
                p[i], p[j] = p[j], p[i]
                current_cost += int(min_val)
                Bp = b[:, p]
                improved = True
        
        return p, current_cost

    best_p = None
    best_cost = np.inf

    # 1. Spectral Initialization + PGD
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        
        # Use top eigenvectors
        idx_a = np.argmax(np.abs(w_a))
        idx_b = np.argmax(np.abs(w_b))
        vec_a = v_a[:, idx_a]
        vec_b = v_b[:, idx_b]
        if np.dot(vec_a, vec_b) < 0:
            vec_b = -vec_b
        
        X0 = np.outer(vec_a, vec_b)
        X0 = project_birkhoff(X0, iters=100)
        
        p1, X1 = run_pgd(X0, deadline - 20.0, lr_factor=1.0)
        if p1 is not None:
            p1, c1 = local_search(p1, deadline - 10.0)
            if c1 < best_cost:
                best_cost = c1
                best_p = p1
    except Exception:
        pass

    # 2. Random Restart + PGD (if time)
    if best_p is not None and time.time() < deadline - 20.0:
        try:
            X2 = np.random.rand(n, n)
            X2 = project_birkhoff(X2, iters=100)
            p2, X2_final = run_pgd(X2, deadline - 20.0, lr_factor=0.5)
            if p2 is not None:
                p2, c2 = local_search(p2, deadline - 10.0)
                if c2 < best_cost:
                    best_cost = c2
                    best_p = p2
        except Exception:
            pass

    # 3. Another random restart or noise-based restart
    if best_p is not None and time.time() < deadline - 20.0:
        try:
            # Perturb best solution? Or new random?
            # Let's try a new random init with different lr
            X3 = np.random.rand(n, n)
            X3 = project_birkhoff(X3, iters=100)
            p3, X3_final = run_pgd(X3, deadline - 20.0, lr_factor=2.0)
            if p3 is not None:
                p3, c3 = local_search(p3, deadline - 10.0)
                if c3 < best_cost:
                    best_cost = c3
                    best_p = p3
        except Exception:
            pass

    if best_p is None:
        # Fallback: Hungarian on A? Or identity?
        # Identity is a valid permutation.
        best_p = np.arange(n)
        best_cost = calc_cost(best_p)

    return best_p.tolist()
# EVOLVE-BLOCK-END
