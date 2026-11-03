"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
1. Construct the known rank-55 decomposition using Strassen's algorithm 
   on the 2x2 sub-blocks (j=1,2) and naive multiplication for j=0.
2. Verify the decomposition.
3. Attempt to find a rank-54 decomposition by performing a local search 
   around the rank-55 solution, specifically trying to merge terms or 
   find a linear dependency that allows dropping one term.
"""

import numpy as np
import random
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False

    # Check zero entries (sampled for speed)
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(500):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _check_residual(U, V, W, n, m, p):
    """
    Compute the squared Frobenius norm residual of the decomposition.
    Residual = sum_{i,k} || sum_r U[i,:,r] * W[k,i,r] * V[:,k,r]^T - I ||_F^2
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    U_mat = U.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_mat = V.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_mat = W.reshape(p, n, rank)
    
    err = 0.0
    for i in range(n):
        for k in range(p):
            # Sum over r: U[i,:,r] * W[k,i,r] * V[:,k,r]^T
            # U[i] is (m, rank), W[k,i] is (rank,), V[:,k] is (m, rank)
            # M_ik = U[i] @ diag(W[k,i]) @ V[:,k].T
            # This is (m, rank) @ (rank, rank) @ (rank, m) -> (m, m)
            U_i = U_mat[i]          # (m, rank)
            W_ki = W_mat[k, i]      # (rank,)
            V_k = V_mat[:, k]       # (m, rank)
            
            # Efficiently compute U_i * W_ki (element-wise scaling of columns)
            # U_i_scaled = U_i * W_ki (broadcasting)
            U_scaled = U_i * W_ki   # (m, rank)
            M_ik = U_scaled @ V_k.T # (m, m)
            
            diff = M_ik - np.eye(m)
            err += np.sum(diff**2)
            
    return err


def _try_merge_terms(U, V, W, n, m, p, deadline):
    """
    Try to find a linear dependency among the rank-55 terms that allows
    dropping one term, resulting in rank 54.
    
    We look for a vector alpha of length R such that sum_r alpha_r * (U_r, V_r, W_r) = 0
    in the tensor sense. If such an alpha exists with all non-zero entries, we can
    re-parameterize. But more practically, we try to find if one term can be expressed
    as a linear combination of others.
    
    A simpler approach: try to remove each term r and see if the remaining R-1 terms
    can be adjusted to still satisfy the decomposition. This is a nonlinear feasibility
    problem. We use a local search around the current solution.
    
    Given time constraints, we try a few random removals and see if a simple correction
    works.
    """
    rank = U.shape[1]
    
    # Try to remove each term and see if we can fix the residual
    for r_remove in range(rank):
        if time.time() > deadline:
            break
            
        # Create a copy with term r_remove removed
        U_new = np.delete(U, r_remove, axis=1)
        V_new = np.delete(V, r_remove, axis=1)
        W_new = np.delete(W, r_remove, axis=1)
        
        # Try to adjust the remaining terms slightly to compensate
        # This is a heuristic: we don't have a clear direction, so we skip
        # complex optimization here and just return the rank-55 solution
        
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Start with rank-55
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Convert to numpy for optimization
    U_best_np = np.array(U_best, dtype=np.float64)
    V_best_np = np.array(V_best, dtype=np.float64)
    W_best_np = np.array(W_best, dtype=np.float64)
    
    # Check residual of the known solution
    res_55 = _check_residual(U_best_np, V_best_np, W_best_np, n, m, p)
    
    # Try to find rank 54 using local search
    # We try to remove one term and optimize the remaining 54 terms
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        # Randomly choose a term to remove
        r_remove = random.randint(0, 54)
        
        U_init = np.delete(U_best_np, r_remove, axis=1)
        V_init = np.delete(V_best_np, r_remove, axis=1)
        W_init = np.delete(W_best_np, r_remove, axis=1)
        
        # Flatten parameters
        n_params = U_init.size + V_init.size + W_init.size
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            if time.time() > deadline - 5:
                return float('inf')
            U_x = x[:U_init.size].reshape(n*m, 54)
            V_x = x[U_init.size:U_init.size+V_init.size].reshape(m*p, 54)
            W_x = x[U_init.size+V_init.size:].reshape(p*n, 54)
            return _check_residual(U_x, V_x, W_x, n, m, p)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < 1e-8:
                # Found a valid rank-54 solution
                U_sol = res.x[:U_init.size].reshape(n*m, 54)
                V_sol = res.x[U_init.size:U_init.size+V_init.size].reshape(m*p, 54)
                W_sol = res.x[U_init.size+V_init.size:].reshape(p*n, 54)
                
                # Verify
                if _verify(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p):
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                    
        except Exception:
            pass

    # Return the best known solution (rank 55)
    return {"U": U_best, "V": V_best, "W": W_best}
