"""
Program to find a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication.
Combines block decomposition (Strassen 2x2x2 sub-blocks) with Newton-based rank reduction search.
"""

import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[n*m*p] as a flat list."""
    n, m, p = DIMENSIONS
    size = n * m * p
    T = [0.0] * size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T, n, m, p

def _apply_factors(U, V, W, n, m, p):
    """Apply factor matrices to produce the reconstructed tensor."""
    rank = len(U[0]) if U else 0
    nmp = n * m * p
    result = [0.0] * nmp
    for r in range(rank):
        u_col = [U[i][r] for i in range(n * m)]
        v_col = [V[j][r] for j in range(m * p)]
        w_col = [W[k][r] for k in range(p * n)]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    result[i * m * p + j * p + k] += u_val * v_val * w_val
    return result

def _verify(U, V, W, n, m, p, T=None):
    """Verify that the decomposition is exact."""
    if T is None:
        T, _, _, _ = _build_tensor()
    result = _apply_factors(U, V, W, n, m, p)
    for i in range(len(T)):
        if abs(result[i] - T[i]) > 1e-9:
            return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _verify_strassen():
    """Verify Strassen decomposition is correct for 2x2x2."""
    sU, sV, sW, srank = _strassen_2x2x2()
    n, m, p = 2, 2, 2
    T222 = [0.0] * 8
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T222[i * 4 + j * 2 + k] = 1.0
    result = _apply_factors(sU, sV, sW, n, m, p)
    for idx in range(8):
        if abs(result[idx] - T222[idx]) > 1e-9:
            return False
    return True

def _block_decomposition(n, m, p, T):
    """
    Block decomposition for 2x3x7 using Strassen for 2x2x2 sub-blocks.
    Returns (U, V, W, rank) or None if verification fails.
    """
    total_rank = 39  # 14 + 25
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, m_block=1, p=7. Schoolbook 2x1x7.
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2, m_block=2, p=7
    sU, sV, sW, srank = _strassen_2x2x2()
    
    # Sub-block 2a: j=1,2; k=0,1. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2b: j=1,2; k=2,3. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2c: j=1,2; k=4,5. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2d: j=1,2; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    if col != total_rank:
        return None
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    
    return U, V, W, total_rank

def _block_decomposition_alt(n, m, p, T):
    """
    Alternative block split: m=3 as {0,1}|{2}, p=7 as (0,1),(2,3),(4,5),(6)
    Same total rank 39, different arrangement.
    """
    total_rank = 39
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0,1, m_block=2, p=7
    sU, sV, sW, srank = _strassen_2x2x2()
    
    # Sub-block 1a: j=0,1; k=0,1. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1b: j=0,1; k=2,3. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1c: j=0,1; k=4,5. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1d: j=0,1; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    # Block 2: j=2, m_block=1, p=7. Schoolbook 2x1x7.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    if col != total_rank:
        return None
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    
    return U, V, W, total_rank

def _newton_search(rank, n, m, p, seed, max_iter=30, tol=1e-8):
    """
    Attempt to find a rank-R decomposition using Newton's method.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    rng = random.Random(seed)
    
    # Initialize with small random values
    U = np.random.uniform(-0.5, 0.5, (n*m, rank))
    V = np.random.uniform(-0.5, 0.5, (m*p, rank))
    W = np.random.uniform(-0.5, 0.5, (p*n, rank))
    
    for _ in range(max_iter):
        # Compute residuals
        res = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res[i*m*p + j*p + k] = 1.0 - s
        
        if np.linalg.norm(res) < tol:
            break
        
        # Compute Jacobian
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        total_vars = num_u + num_v + num_w
        
        J = np.zeros((n*m*p, total_vars))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    
                    for r in range(rank):
                        var_idx_u = (i*m+j)*rank + r
                        J[eq_idx, var_idx_u] = V[j*p+k, r] * W[k*n+i, r]
                        
                        var_idx_v = num_u + (j*p+k)*rank + r
                        J[eq_idx, var_idx_v] = U[i*m+j, r] * W[k*n+i, r]
                        
                        var_idx_w = num_u + num_v + (k*n+i)*rank + r
                        J[eq_idx, var_idx_w] = U[i*m+j, r] * V[j*p+k, r]
        
        try:
            delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        except Exception:
            break
        
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        U_flat += delta[:num_u]
        V_flat += delta[num_u:num_u+num_v]
        W_flat += delta[num_u+num_v:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
    
    # Check final residual
    res = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                res[i*m*p + j*p + k] = 1.0 - s
    
    if np.linalg.norm(res) < 1e-6:
        return U, V, W
    return None

def _round_and_verify(U, V, W, n, m, p):
    """
    Round the float solution to nearest half-integers and verify.
    """
    U_r = [[round(float(v) * 2) / 2.0 for v in row] for row in U]
    V_r = [[round(float(v) * 2) / 2.0 for v in row] for row in V]
    W_r = [[round(float(v) * 2) / 2.0 for v in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    
    U_i = [[round(float(v)) for v in row] for row in U]
    V_i = [[round(float(v)) for v in row] for row in V]
    W_i = [[round(float(v)) for v in row] for row in W]
    if _verify(U_i, V_i, W_i, n, m, p):
        return U_i, V_i, W_i
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # First verify Strassen is correct
    if not _verify_strassen():
        # Fallback to schoolbook if Strassen verification fails
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    T, _, _, _ = _build_tensor()
    
    # Try primary block decomposition
    result = _block_decomposition(n, m, p, T)
    if result is not None:
        U, V, W, rank = result
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    else:
        # Try alternative block decomposition
        result = _block_decomposition_alt(n, m, p, T)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            U, V, W, rank = _schoolbook(n, m, p)
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower ranks using Newton's method
    # Lower bound is max(n*m, m*p, p*n) = 21
    # Try ranks from 22 to best_rank-1
    for rank in range(22, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random seeds for Newton's method
        for seed in range(3):
            if time.time() > deadline:
                break
                
            result = _newton_search(rank, n, m, p, seed)
            if result:
                U_np, V_np, W_np = result
                verified = _round_and_verify(U_np.tolist(), V_np.tolist(), W_np.tolist(), n, m, p)
                if verified:
                    U_r, V_r, W_r = verified
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = rank
                    found = True
                    break # Found a valid decomposition for this rank, try lower rank
    
    return best_result
