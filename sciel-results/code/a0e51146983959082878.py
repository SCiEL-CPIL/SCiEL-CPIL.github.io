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

def _try_als_search(rank, start_time, deadline):
    """
    Attempt to find a decomposition of the given rank using ALS.
    """
    n, m, p = DIMENSIONS
    U39, V39, W39 = _build_rank_39()
    
    # Initialize U, V, W with random values
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Flatten T for ALS
    T_au = T.reshape(n*m, m*p * p*n) # a, bc
    T_av = T.reshape(n*m * m*p, p*n) # ab, c
    T_aw = T.reshape(n*m * m*p, p*n) # ab, c (same as T_av for W update? No, indices differ)
    # T[a,b,c]. 
    # U update: T[a, b,c] = sum_r U[a,r] (V[b,r] W[c,r]).
    # Let Z_r = V[:,r] \otimes W[:,r]. Z is (m*p * p*n) x rank.
    # T_au[a, :] = sum_r U[a,r] Z[r, :].
    # U = T_au Z^T (Z Z^T)^{-1}.
    
    # V update: T[a,b, c] = sum_r (U[a,r] V[b,r]) W[c,r].
    # Let Y_r = U[:,r] \otimes V[:,r]. Y is (n*m * m*p) x rank.
    # T_av[:, c] = sum_r Y[:, r] W[c, r].
    # T_av = Y W^T.
    # W^T = Y^T T_av^T (T_av T_av^T)^{-1}? No.
    # T_av[:, c] = sum_r Y[:, r] W[c, r].
    # This is T_av = Y W^T.
    # W^T = (Y^T Y)^{-1} Y^T T_av^T.
    # W = T_av Y (Y^T Y)^{-1}.
    
    # W update: T[a,b, c] = sum_r (U[a,r] V[b,r]) W[c, r].
    # Let X_r = U[:,r] \otimes V[:,r]. X is (n*m * m*p) x rank.
    # T_aw[:, c] = sum_r X[:, r] W[c, r].
    # T_aw = X W^T.
    # W^T = (X^T X)^{-1} X^T T_aw^T.
    # W = T_aw X (X^T X)^{-1}.
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Update U
        Z = np.einsum('br,cr->bc r', V, W).reshape(m*p * p*n, rank)
        # T_au = U Z^T
        # U = T_au Z^T (Z Z^T)^{-1}
        try:
            U = np.linalg.lstsq(Z.T, T_au.T, rcond=None)[0].T
        except:
            pass
            
        # Update V
        # T_av = Y W^T where Y = U \otimes V
        # This is coupled. We need to solve for V.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        # For fixed U, W, we want V.
        # T[a,b,c] = sum_r (U[a,r] W[c,r]) V[b,r].
        # Let K_r = U[:,r] \otimes W[:,r]. K is (n*m * p*n) x rank.
        # T_auv? No.
        # T[a, b, c] = sum_r K_{ac, r} V[b, r].
        # Flatten T to T_ac, b.
        # T_ac, b = sum_r K_ac, r V_b, r.
        # T_ac, : = K_ac, : V^T.
        # V^T = K_ac, :^T T_ac, :^T (T_ac, : T_ac, :)^+?
        # V = T_ac, : K_ac, :^T (K_ac, : K_ac, :)^+?
        
        # Let's use a simpler iterative step:
        # V[b, r] = sum_{a,c} T[a,b,c] U[a,r] W[c,r] / (sum_{a,c} (U[a,r] W[c,r])^2)
        # This is the gradient step for V.
        
        # Gradient descent / ALS hybrid
        # Let's stick to least squares if possible.
        # For V:
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        # Fix U, W.
        # T[:, b, :] = sum_r U[:, r] V[b, r] W[:, r].
        # This is a linear system for V[b, :].
        # Let M_b = sum_r U[:, r] W[:, r] V[b, r]? No.
        # T[:, b, :] = sum_r (U[:, r] \otimes W[:, r]) V[b, r].
        # Let P_r = U[:, r] \otimes W[:, r]. P is (n*m * p*n) x rank.
        # T[:, b, :] flattened is T_b.
        # T_b = sum_r P_r V[b, r].
        # T_b = P V[b, :].
        # V[b, :] = P^+ T_b.
        
        P = np.einsum('ar,cr->ac r', U, W).reshape(n*m * p*n, rank)
        for b in range(m*p):
            T_b = T[:, b, :].reshape(-1)
            try:
                V[:, b] = np.linalg.lstsq(P, T_b, rcond=None)[0]
            except:
                pass
                
        # Update W
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        # Fix U, V.
        # T[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r].
        # Let Q_r = U[:, r] \otimes V[:, r]. Q is (n*m * m*p) x rank.
        # T[a,b,c] flattened over a,b is T_c.
        # T_c = sum_r Q_r W[c, r].
        # T_c = Q W[c, :].
        # W[c, :] = Q^+ T_c.
        
        Q = np.einsum('ar,br->ab r', U, V).reshape(n*m * m*p, rank)
        for c in range(p*n):
            T_c = T[:, :, c].reshape(-1)
            try:
                W[:, c] = np.linalg.lstsq(Q, T_c, rcond=None)[0]
            except:
                pass
                
        # Calculate residual
        T_dec = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_dec - T)
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if residual < 1e-8:
            break

    if best_residual < 1e-6:
        # Try to round
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        if _verify(U_round, V_round, W_round, n, m, p):
            return U_round, V_round, W_round
            
        # Try integer rounding
        U_round = np.round(best_U)
        V_round = np.round(best_V)
        W_round = np.round(best_W)
        
        if _verify(U_round, V_round, W_round, n, m, p):
            return U_round, V_round, W_round
            
    return None

def _try_rank_38_search(start_time, deadline):
    """
    Attempt to find a rank-38 decomposition.
    """
    n, m, p = DIMENSIONS
    # Try rank 38
    result = _try_als_search(38, start_time, deadline)
    if result is not None:
        return result
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
