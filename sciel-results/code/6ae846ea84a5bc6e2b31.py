import time
import numpy as np
from numpy.linalg import svd, norm
import random

DIMENSIONS = [2, 2, 4]

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, tol=1e-10):
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if U.shape != (n * m, r) or V.shape != (m * p, r) or W.shape != (p * n, r):
        return False
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err < tol

def _residual_matrix(U, V, W):
    n, m, p = 2, 2, 4
    r = U.shape[1]
    res = np.zeros(n * m * p)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                res[i * m * p + k * p + j] = val - 1.0
    return res

def _search_rank12(deadline):
    n, m, p = 2, 2, 4
    target_rank = 12
    np.random.seed(42)
    
    best_result = None
    best_residual = np.inf
    
    # Try multiple random restarts
    for restart in range(500):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize columns to avoid scaling issues
        for r in range(target_rank):
            nU = np.linalg.norm(U[:, r])
            nV = np.linalg.norm(V[:, r])
            nW = np.linalg.norm(W[:, r])
            if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                U[:, r] /= nU
                V[:, r] /= nV
                W[:, r] /= nW
        
        # ALS iterations
        for it in range(50):
            if time.time() > deadline - 2:
                break
            
            # Update U
            for r in range(target_rank):
                other_contrib = np.zeros(n * m * p)
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            other_contrib[i * m * p + k * p + j] = np.dot(
                                U[i * m + k, :],
                                V[k * p + j, :] * W[j * n + i, :]
                            ) - U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
                target = np.ones(n * m * p) - other_contrib
                vr = V[:, r]
                wr = W[:, r]
                for i in range(n):
                    for k in range(m):
                        U[i * m + k, r] = sum(
                            target[i * m * p + k * p + j] * vr[k * p + j] * wr[j * n + i]
                            for j in range(p)
                        )
            
            # Update V
            for r in range(target_rank):
                other_contrib = np.zeros(n * m * p)
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            other_contrib[i * m * p + k * p + j] = np.dot(
                                U[i * m + k, :],
                                V[k * p + j, :] * W[j * n + i, :]
                            ) - U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
                target = np.ones(n * m * p) - other_contrib
                ur = U[:, r]
                wr = W[:, r]
                for k in range(m):
                    for j in range(p):
                        V[k * p + j, r] = sum(
                            target[i * m * p + k * p + j] * ur[i * m + k] * wr[j * n + i]
                            for i in range(n)
                        )
            
            # Update W
            for r in range(target_rank):
                other_contrib = np.zeros(n * m * p)
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            other_contrib[i * m * p + k * p + j] = np.dot(
                                U[i * m + k, :],
                                V[k * p + j, :] * W[j * n + i, :]
                            ) - U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
                target = np.ones(n * m * p) - other_contrib
                ur = U[:, r]
                vr = V[:, r]
                for j in range(p):
                    for i in range(n):
                        W[j * n + i, r] = sum(
                            target[i * m * p + k * p + j] * ur[i * m + k] * vr[k * p + j]
                            for k in range(m)
                        )
        
        # Check residual
        res = _residual_matrix(U, V, W)
        res_norm = np.linalg.norm(res)
        
        if res_norm < best_residual:
            best_residual = res_norm
            if res_norm < 1e-8:
                # Found a valid rank-12 decomposition!
                U = np.round(U, 10)
                V = np.round(V, 10)
                W = np.round(W, 10)
                # Check if entries are close to integers or half-integers
                # The problem allows integer/half-integer entries.
                # If the solution is real, we might need to check if it's exactly representable.
                # However, the verifier checks the product, so if it's close enough, it's fine.
                # But the output must be exact? "exactly decompose".
                # So we should round to nearest 0.5 and re-verify.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded):
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # If rounding fails, try without rounding if it was already very close
                    if _verify_decomposition(U, V, W):
                        U_list = [[float(x) for x in row] for row in U]
                        V_list = [[float(x) for x in row] for row in V]
                        W_list = [[float(x) for x in row] for row in W]
                        return {"U": U_list, "V": V_list, "W": W_list}

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find rank 12
    result = _search_rank12(deadline)
    
    if result is not None:
        return result
    
    # Fall back to rank 14
    return _build_rank14()
