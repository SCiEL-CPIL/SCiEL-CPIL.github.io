import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposition tensor
    # T_dec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_dec, T, atol=tol)

def _build_rank_39():
    """
    Build the known rank 39 decomposition.
    5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1).
    """
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal r
        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n     # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n     # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]

        # Strassen terms
        # m1 = (a00+a11)(b00+b11)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (a10+a11)b00
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a00(b01-b11)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = a11(b10-b00)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a00+a01)b11
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (a10-a00)(b00+b01)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (a01-a11)(b10+b11)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    # Last column k=10: Schoolbook
    k2 = 10
    v_e2 = k2      # B[0, 10]
    v_g2 = p + k2  # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]

    # Q1: a00 * B[0, 10]
    U[0, r] += 1
    V[v_e2, r] += 1
    W[w_c02, r] += 1
    r += 1

    # Q2: a01 * B[1, 10]
    U[1, r] += 1
    V[v_g2, r] += 1
    W[w_c02, r] += 1
    r += 1

    # Q3: a10 * B[0, 10]
    U[2, r] += 1
    V[v_e2, r] += 1
    W[w_c12, r] += 1
    r += 1

    # Q4: a11 * B[1, 10]
    U[3, r] += 1
    V[v_g2, r] += 1
    W[w_c12, r] += 1
    r += 1
    
    return U, V, W

def _try_rank_38_search(start_time, deadline):
    """
    Attempt to find a rank-38 decomposition using ALS and random restarts.
    """
    n, m, p = DIMENSIONS
    rank_target = 38
    U39, V39, W39 = _build_rank_39()
    
    # Build target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Reshape T for ALS
    # T[a,b,c] -> T_a[b,c] etc.
    T_a = np.zeros((n*m, m*p * p*n))
    T_b = np.zeros((m*p, n*m * p*n))
    T_c = np.zeros((p*n, n*m * m*p))
    
    # This is inefficient to build every time, but we do it once.
    # Actually, we can just use the einsum in the cost function.
    
    def cost(U, V, W):
        T_dec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_dec - T)
    
    # Try a few random restarts
    # Initialize from rank 39 by dropping a column
    # And add some noise
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        # Drop a random column from rank 39
        drop_idx = random.randint(0, 38)
        indices = [i for i in range(39) if i != drop_idx]
        U_init = U39[:, indices]
        V_init = V39[:, indices]
        W_init = W39[:, indices]
        
        # Add small noise to escape local minima?
        # Or just start with random if random restarts are better?
        # Let's try random initialization with integer/half-integer bias
        # But ALS is better with a good start.
        
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        # ALS iterations
        # Fix U, optimize V:
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed U, W: T[a,b,c] is linear in V[b,r].
        # Minimize sum_{a,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
        # This is a linear least squares problem for each b.
        # Let M_b be matrix of size (n*m * p*n) x rank
        # M_b[ (a,c), r ] = U[a,r] * W[c,r]
        # Target vector y_b[ (a,c) ] = T[a,b,c]
        
        # Precompute M_b for each b? No, U and W change.
        
        max_iter = 50
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
                
            # Update V
            for b in range(m*p):
                # Construct RHS: y_b
                y_b = T[:, b, :].flatten() # shape (n*m * p*n)
                
                # Construct A: shape (n*m * p*n, rank)
                # A[ (a,c), r ] = U[a,r] * W[c,r]
                A = np.zeros((n*m * p*n, rank_target))
                for a in range(n*m):
                    for c in range(p*n):
                        for r in range(rank_target):
                            A[a * p*n + c, r] = U_curr[a, r] * W_curr[c, r]
                
                # Solve A * V_curr[:, b] = y_b
                # Using least squares
                V_curr[:, b], residuals, rank, sv = np.linalg.lstsq(A, y_b, rcond=None)
            
            # Update W
            for c in range(p*n):
                y_c = T[:, :, c].flatten() # shape (n*m * m*p)
                A = np.zeros((n*m * m*p, rank_target))
                for a in range(n*m):
                    for b in range(m*p):
                        for r in range(rank_target):
                            A[a * m*p + b, r] = U_curr[a, r] * V_curr[b, r]
                W_curr[:, c], residuals, rank, sv = np.linalg.lstsq(A, y_c, rcond=None)
            
            # Update U
            for a in range(n*m):
                y_a = T[a, :, :].flatten() # shape (m*p * p*n)
                A = np.zeros((m*p * p*n, rank_target))
                for b in range(m*p):
                    for c in range(p*n):
                        for r in range(rank_target):
                            A[b * p*n + c, r] = V_curr[b, r] * W_curr[c, r]
                U_curr[:, a], residuals, rank, sv = np.linalg.lstsq(A, y_a, rcond=None)
            
            # Check convergence
            err = cost(U_curr, V_curr, W_curr)
            if err < 1e-6:
                # Try to round
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                    return U_rounded, V_rounded, W_rounded
                # Also check if the unrounded is exact enough?
                # The task requires integer/half-integer entries.
                # So we must round and verify.

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Build the known rank 39 solution
    U, V, W = _build_rank_39()
    
    # 2. Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 44
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # 3. Try to find rank 38 (or lower)
    result = _try_rank_38_search(start_time, deadline)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
