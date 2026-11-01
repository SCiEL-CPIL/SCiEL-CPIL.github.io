"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    # T[i, j, k] = A_{i1, i2} * B_{j1, j2} * C_{k1, k2}
    # Indices:
    # i: 0..3 (A row, col) -> i = r_A * m + c_A
    # j: 0..9 (B row, col) -> j = r_B * p + c_B
    # k: 0..9 (C row, col) -> k = r_C * n + c_C
    # Product is sum over c_A, r_B, r_C, c_B, c_C?
    # (A B)_{i1, j2} = sum_{r} A_{i1, r} B_{r, j2}
    # Output C_{i1, j2} = sum_{r} A_{i1, r} B_{r, j2}
    # So T_{i, j, k} is non-zero if:
    # i = i1 * m + r
    # j = r * p + j2
    # k = i1 * n + j2
    # Value is 1.
    for i1 in range(n):
        for r in range(m):
            for j2 in range(p):
                i = i1 * m + r
                j = r * p + j2
                k = i1 * n + j2
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-6):
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # Check if sum_r U[:,r] V[:,r] W[:,r] == T
    # Use einsum
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    return err < tol

def _is_half_integer(mat, tol=1e-6):
    # Check if 2 * mat is close to integers
    return np.allclose(2 * mat, np.round(2 * mat), atol=tol)

def _try_scaling(U, V, W, T, tol=1e-6):
    # We can scale U, V, W such that U * V * W is invariant?
    # No, the product is invariant under U->cU, V->V/c, W->W.
    # We want all three to be half-integers.
    # This is a constraint on the relative scales.
    # Let's try to find a scale s such that s*U, (1/s)*V, W are half-integers?
    # Or s*U, V, W/s?
    # Actually, we can just check if there exists a common scaling for the set.
    # Since the problem is small, we can try to normalize one entry to 1.
    # If we find a real solution, it might be a rational multiple of a half-integer solution.
    
    # Heuristic: Check if U, V, W are already half-integers.
    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
        return U, V, W
    
    # Try to find a scale for U such that s*U is half-int, and (1/s)*V is half-int?
    # This is tricky. Let's just return the real solution if it's very close to half-integers
    # after some simple scaling attempts.
    
    # Try scaling U by 1/2, 2, 1/4, 4
    for s in [1, 0.5, 2, 0.25, 4]:
        if _is_half_integer(s * U) and _is_half_integer(V / s) and _is_half_integer(W):
            return s * U, V / s, W
        if _is_half_integer(U * s) and _is_half_integer(V) and _is_half_integer(W / s):
            return U * s, V, W / s
        if _is_half_integer(U / s) and _is_half_integer(V * s) and _is_half_integer(W):
            return U / s, V * s, W
            
    return None

def _als_step_U(U, V, W, T):
    # Minimize || T - sum_r U_r V_r W_r ||_F^2 with respect to U
    # For each i, T[i, :, :] = sum_r U[i, r] (V[:, r] * W[:, r])
    # Let M_r = V[:, r] * W[:, r] (outer product vectorized)
    # T[i, vec] = sum_r U[i, r] M_r[vec]
    # This is a linear system A x = b for each i
    # A is (mp*pn, R) where A[j, r] = V[j, r] * W[j, r] ? No.
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Fix i. We want to find U[i, :].
    # b[j, k] = T[i, j, k]
    # A[j, k, r] = V[j, r] W[k, r]
    # Vectorize b and A.
    n, m, p = DIMENSIONS
    R = U.shape[1]
    U_new = np.zeros_like(U)
    for i in range(n * m):
        b = T[i].flatten()
        # Construct A
        # A shape: (m*p * p*n, R)
        # A[idx, r] = V[j, r] * W[k, r]
        # j from 0..mp-1, k from 0..pn-1
        # idx = j * pn + k
        A = np.zeros((m * p * p * n, R))
        for r in range(R):
            # V[:, r] is (mp, 1), W[:, r] is (pn, 1)
            # Outer product is (mp, pn)
            outer = np.outer(V[:, r], W[:, r])
            A[:, r] = outer.flatten()
        # Solve A x = b
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        U_new[i] = x
    return U_new

def _als_step_V(U, V, W, T):
    n, m, p = DIMENSIONS
    R = V.shape[1]
    V_new = np.zeros_like(V)
    for j in range(m * p):
        b = T[:, j, :].transpose(1, 0).flatten() # T[i, j, k] -> b[k, i]?
        # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Fix j. We want V[j, :].
        # b[i, k] = T[i, j, k]
        # A[i, k, r] = U[i, r] W[k, r]
        A = np.zeros((n * m * p * n, R))
        for r in range(R):
            outer = np.outer(U[:, r], W[:, r])
            A[:, r] = outer.flatten()
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        V_new[j] = x
    return V_new

def _als_step_W(U, V, W, T):
    n, m, p = DIMENSIONS
    R = W.shape[1]
    W_new = np.zeros_like(W)
    for k in range(p * n):
        b = T[:, :, k].flatten() # T[i, j, k]
        # Fix k. We want W[k, :].
        # b[i, j] = T[i, j, k]
        # A[i, j, r] = U[i, r] V[j, r]
        A = np.zeros((n * m * m * p, R))
        for r in range(R):
            outer = np.outer(U[:, r], V[:, r])
            A[:, r] = outer.flatten()
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        W_new[k] = x
    return W_new

def _search_rank(R, T, max_iters=20, attempts=50):
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100 # 100s limit for search
    
    for attempt in range(attempts):
        if time.time() > deadline:
            break
        
        # Initialize
        U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
        
        # Avoid zero columns
        for r in range(R):
            if np.all(U[:, r] == 0): U[0, r] = 1
            if np.all(V[:, r] == 0): V[0, r] = 1
            if np.all(W[:, r] == 0): W[0, r] = 1
            
        for it in range(max_iters):
            U = _als_step_U(U, V, W, T)
            V = _als_step_V(U, V, W, T)
            W = _als_step_W(U, V, W, T)
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-5:
                # Found a real solution. Try to convert to half-integers
                sol = _try_scaling(U, V, W, T)
                if sol is not None:
                    U_f, V_f, W_f = sol
                    # Final verification
                    if _check_decomposition(U_f, V_f, W_f, T):
                        return U_f, V_f, W_f
                else:
                    # Maybe it is a valid solution but scaling failed.
                    # Check if it's close to half-integers directly
                    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                         if _check_decomposition(U, V, W, T):
                             return U, V, W

    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Try to find rank 15
    for R in [15, 16, 17]:
        result = _search_rank(R, T)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Rank 18
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
