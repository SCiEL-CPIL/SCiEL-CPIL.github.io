"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy:
1. Use Strassen rank-7 for 2x2x2 blocks to build a rank-56 baseline via Kronecker product
   with a rank-8 decomposition for 2x2x4 (constructed by solving).
2. Attempt rank reduction below 56 using vectorized ALS with randomized restarts.
3. Return the lowest rank solution found within the time limit.
"""

import time
import numpy as np

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    n = m = p = 2
    r = 7
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, atol=1e-8):
    """Verify tensor decomposition: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=atol))

def _build_rank56():
    """
    Build rank-56 decomposition for 4x4 * 4x8 using Kronecker product of:
    - 2x2x4 rank-8 (solved)
    - 2x2x2 rank-7 (Strassen)
    """
    n1, m1, p1 = 2, 2, 4
    n2, m2, p2 = 2, 2, 2
    n, m, p = 4, 4, 8
    
    # Solve for 2x2x4 rank 8
    r1 = 8
    U1 = np.random.randn(n1*m1, r1)
    V1 = np.random.randn(m1*p1, r1)
    W1 = np.random.randn(p1*n1, r1)
    
    # ALS for 2x2x4
    T1 = np.ones(n1*m1*p1)
    for _ in range(100):
        # Solve W1
        A_W = np.zeros((n1*m1*p1, p1*n1*r1))
        for c in range(r1):
            u_c = U1[:, c].reshape(n1, m1)
            v_c = V1[:, c].reshape(m1, p1)
            coeff = np.einsum('ij,jk->ijk', u_c, v_c)
            for i in range(n1):
                for j in range(m1):
                    for k in range(p1):
                        eq_idx = i*m1*p1 + j*p1 + k
                        var_idx = (k*n1 + i)*r1 + c
                        A_W[eq_idx, var_idx] = coeff[i,j,k]
        W1_flat, _, _, _ = np.linalg.lstsq(A_W, T1, rcond=None)
        W1 = W1_flat.reshape(p1*n1, r1)
        
        # Solve U1
        A_U = np.zeros((n1*m1*p1, n1*m1*r1))
        for c in range(r1):
            v_c = V1[:, c].reshape(m1, p1)
            w_c = W1[:, c].reshape(p1, n1)
            coeff = np.einsum('jk,ki->ijk', v_c, w_c)
            for i in range(n1):
                for j in range(m1):
                    for k in range(p1):
                        eq_idx = i*m1*p1 + j*p1 + k
                        var_idx = (i*m1 + j)*r1 + c
                        A_U[eq_idx, var_idx] = coeff[i,j,k]
        U1_flat, _, _, _ = np.linalg.lstsq(A_U, T1, rcond=None)
        U1 = U1_flat.reshape(n1*m1, r1)
        
        # Solve V1
        A_V = np.zeros((n1*m1*p1, m1*p1*r1))
        for c in range(r1):
            u_c = U1[:, c].reshape(n1, m1)
            w_c = W1[:, c].reshape(p1, n1)
            coeff = np.einsum('ij,ki->ijk', u_c, w_c)
            for i in range(n1):
                for j in range(m1):
                    for k in range(p1):
                        eq_idx = i*m1*p1 + j*p1 + k
                        var_idx = (j*p1 + k)*r1 + c
                        A_V[eq_idx, var_idx] = coeff[i,j,k]
        V1_flat, _, _, _ = np.linalg.lstsq(A_V, T1, rcond=None)
        V1 = V1_flat.reshape(m1*p1, r1)
        
        # Check convergence
        U3 = U1.reshape(n1, m1, r1)
        V3 = V1.reshape(m1, p1, r1)
        W3 = W1.reshape(p1, n1, r1)
        T_calc = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        if np.allclose(T_calc, 1.0, atol=1e-6):
            break
    
    # Round to half-integers
    U1 = np.round(U1 * 2) / 2.0
    V1 = np.round(V1 * 2) / 2.0
    W1 = np.round(W1 * 2) / 2.0
    
    if not _verify(U1, V1, W1, n1, m1, p1, atol=1e-5):
        # Try with more iterations or different seed
        for attempt in range(10):
            U1 = np.random.randn(n1*m1, r1)
            V1 = np.random.randn(m1*p1, r1)
            W1 = np.random.randn(p1*n1, r1)
            for _ in range(200):
                A_W = np.zeros((n1*m1*p1, p1*n1*r1))
                for c in range(r1):
                    u_c = U1[:, c].reshape(n1, m1)
                    v_c = V1[:, c].reshape(m1, p1)
                    coeff = np.einsum('ij,jk->ijk', u_c, v_c)
                    for i in range(n1):
                        for j in range(m1):
                            for k in range(p1):
                                eq_idx = i*m1*p1 + j*p1 + k
                                var_idx = (k*n1 + i)*r1 + c
                                A_W[eq_idx, var_idx] = coeff[i,j,k]
                W1_flat, _, _, _ = np.linalg.lstsq(A_W, T1, rcond=None)
                W1 = W1_flat.reshape(p1*n1, r1)
                
                A_U = np.zeros((n1*m1*p1, n1*m1*r1))
                for c in range(r1):
                    v_c = V1[:, c].reshape(m1, p1)
                    w_c = W1[:, c].reshape(p1, n1)
                    coeff = np.einsum('jk,ki->ijk', v_c, w_c)
                    for i in range(n1):
                        for j in range(m1):
                            for k in range(p1):
                                eq_idx = i*m1*p1 + j*p1 + k
                                var_idx = (i*m1 + j)*r1 + c
                                A_U[eq_idx, var_idx] = coeff[i,j,k]
                U1_flat, _, _, _ = np.linalg.lstsq(A_U, T1, rcond=None)
                U1 = U1_flat.reshape(n1*m1, r1)
                
                A_V = np.zeros((n1*m1*p1, m1*p1*r1))
                for c in range(r1):
                    u_c = U1[:, c].reshape(n1, m1)
                    w_c = W1[:, c].reshape(p1, n1)
                    coeff = np.einsum('ij,ki->ijk', u_c, w_c)
                    for i in range(n1):
                        for j in range(m1):
                            for k in range(p1):
                                eq_idx = i*m1*p1 + j*p1 + k
                                var_idx = (j*p1 + k)*r1 + c
                                A_V[eq_idx, var_idx] = coeff[i,j,k]
                V1_flat, _, _, _ = np.linalg.lstsq(A_V, T1, rcond=None)
                V1 = V1_flat.reshape(m1*p1, r1)
                
                U3 = U1.reshape(n1, m1, r1)
                V3 = V1.reshape(m1, p1, r1)
                W3 = W1.reshape(p1, n1, r1)
                T_calc = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                if np.allclose(T_calc, 1.0, atol=1e-6):
                    break
            U1 = np.round(U1 * 2) / 2.0
            V1 = np.round(V1 * 2) / 2.0
            W1 = np.round(W1 * 2) / 2.0
            if _verify(U1, V1, W1, n1, m1, p1, atol=1e-5):
                break
    
    # Get Strassen 2x2x2
    U2, V2, W2 = _strassen_2x2()
    r2 = 7
    
    # Kronecker product to build 4x4x8
    rank = r1 * r2
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # U: (i1*m1+j1, c1) * (i2*m2+j2, c2) -> (i*m+j, c)
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, c2]
                            if u2_val == 0:
                                continue
                            i = n1 * i1 + i2
                            j = m1 * j1 + j2
                            U[i * m + j, c] += u1_val * u2_val
            
            # V: (j1*p1+k1, c1) * (j2*p2+k2, c2) -> (j*p+k, c)
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, c2]
                            if v2_val == 0:
                                continue
                            j = m1 * j1 + j2
                            k = p1 * k1 + k2
                            V[j * p + k, c] += v1_val * v2_val
            
            # W: (k1*n1+i1, c1) * (k2*n2+i2, c2) -> (k*n+i, c)
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, c2]
                            if w2_val == 0:
                                continue
                            k = p1 * k1 + k2
                            i = n1 * i1 + i2
                            W[k * n + i, c] += w1_val * w2_val
    
    return U, V, W

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _als_step(U, V, W, n, m, p, r):
    """One ALS iteration: solve for W, then U, then V."""
    T_flat = np.ones(n * m * p)
    
    # Solve W
    A_W = np.zeros((n * m * p, p * n * r))
    for c in range(r):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        coeff = np.einsum('ij,jk->ijk', u_c, v_c)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    var_idx = (k * n + i) * r + c
                    A_W[eq_idx, var_idx] = coeff[i, j, k]
    W_flat, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
    W = W_flat.reshape(p * n, r)
    
    # Solve U
    A_U = np.zeros((n * m * p, n * m * r))
    for c in range(r):
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        coeff = np.einsum('jk,ki->ijk', v_c, w_c)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    var_idx = (i * m + j) * r + c
                    A_U[eq_idx, var_idx] = coeff[i, j, k]
    U_flat, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
    U = U_flat.reshape(n * m, r)
    
    # Solve V
    A_V = np.zeros((n * m * p, m * p * r))
    for c in range(r):
        u_c = U[:, c].reshape(n, m)
        w_c = W[:, c].reshape(p, n)
        coeff = np.einsum('ij,ki->ijk', u_c, w_c)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    var_idx = (j * p + k) * r + c
                    A_V[eq_idx, var_idx] = coeff[i, j, k]
    V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
    V = V_flat.reshape(m * p, r)
    
    return U, V, W

def _round_and_verify(U, V, W, n, m, p, r):
    """Round to half-integers and verify."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    if _verify(U_r, V_r, W_r, n, m, p, atol=1e-6):
        return U_r, V_r, W_r
    # Also try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, n, m, p, atol=1e-6):
        return U_i, V_i, W_i
    return None, None, None

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140
    
    n, m, p = 4, 4, 8
    
    # Build rank-56 baseline
    try:
        U_best, V_best, W_best = _build_rank56()
        best_rank = 56
        
        if not _verify(U_best, V_best, W_best, n, m, p):
            # Fallback to schoolbook
            U_best, V_best, W_best = _schoolbook(n, m, p)
            best_rank = n * m * p
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
        
    except Exception:
        U_best, V_best, W_best = _schoolbook(n, m, p)
        best_rank = n * m * p
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    # Try to reduce rank below 56
    for target_rank in range(48, 56):
        if time.time() > DEADLINE - 15:
            break
        
        # Multiple random restarts
        for restart in range(10):
            if time.time() > DEADLINE - 15:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Run ALS iterations
            converged = False
            for it in range(30):
                U, V, W = _als_step(U, V, W, n, m, p, target_rank)
                
                # Check if we're close to a valid solution
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                T_calc = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.max(np.abs(T_calc - 1.0))
                if err < 1e-6:
                    converged = True
                    break
            
            # Round and verify
            U_r, V_r, W_r = _round_and_verify(U, V, W, n, m, p, target_rank)
            if U_r is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found valid solution for this rank, try lower
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
