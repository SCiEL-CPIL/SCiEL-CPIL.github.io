import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify(U, V, W, n=2, m=2, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _als_step_U(U, V, W, T_flat):
    # T_flat is flattened target tensor of shape (n*m, m*p, p*n)
    # We solve for U
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Construct B matrix
    # Rows correspond to (j, k) pairs
    # B[j*p+k, r] = V[j*p+k, r] * W[k*n+i, r] ? No, W depends on i too.
    # The equation is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    # For a fixed i, we solve for U[i*m + j, :] for all j.
    # Actually, U has rows indexed by (i, j).
    # Let's solve for each row of U separately? Or all rows?
    # The system for U_row (which is U[i*m+j, :]) depends on i and j.
    # For fixed i, j:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    # This is a linear system in U[i*m+j, :].
    # Let x = U[i*m+j, :].
    # sum_r x_r (V[j*p+k, r] W[k*n+i, r]) = T_val
    # Let M[k] = V[j*p+k, :] * W[k*n+i, :]
    # Then x @ M^T = T_col
    # x = T_col @ (M^T M)^-1 M^T
    
    # We can do this for all i, j, k?
    # No, for fixed i, j, the system is over k.
    # There are p equations for R unknowns.
    # If p < R, it's underdetermined. We use least squares.
    
    U_new = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            # Construct M of shape (p, R)
            M = np.zeros((p, R))
            T_col = np.zeros(p)
            for k in range(p):
                M[k, :] = V[j*p + k, :] * W[k*n + i, :]
                T_col[k] = T_flat[i*m + j, j*p + k, k*n + i]
            
            # Solve x @ M^T = T_col
            # x = T_col @ (M^T M)^-1 M^T
            MTM = M.T @ M
            # Add regularization to avoid singularity
            MTM += 1e-8 * np.eye(R)
            MTT = M.T
            try:
                x = np.linalg.solve(MTM, MTT.T @ T_col)
                # x is shape (R,)
                U_new[i*m + j, :] = x
            except np.linalg.LinAlgError:
                U_new[i*m + j, :] = U[i*m + j, :]
                
    return U_new

def _als_step_V(U, V, W, T_flat):
    n, m, p = DIMENSIONS
    R = V.shape[1]
    V_new = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            # Solve for V[j*p+k, :]
            # sum_r V[j*p+k, r] U[i*m+j, r] W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
            # For fixed j, k, we have n equations (over i) for R unknowns.
            M = np.zeros((n, R))
            T_col = np.zeros(n)
            for i in range(n):
                M[i, :] = U[i*m + j, :] * W[k*n + i, :]
                T_col[i] = T_flat[i*m + j, j*p + k, k*n + i]
            
            MTM = M.T @ M
            MTM += 1e-8 * np.eye(R)
            MTT = M.T
            try:
                x = np.linalg.solve(MTM, MTT.T @ T_col)
                V_new[j*p + k, :] = x
            except np.linalg.LinAlgError:
                V_new[j*p + k, :] = V[j*p + k, :]
    return V_new

def _als_step_W(U, V, W, T_flat):
    n, m, p = DIMENSIONS
    R = W.shape[1]
    W_new = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            # Solve for W[k*n+i, :]
            # sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] = T[i*m+j, j*p+k, k*n+i]
            # For fixed k, i, we have m equations (over j) for R unknowns.
            M = np.zeros((m, R))
            T_col = np.zeros(m)
            for j in range(m):
                M[j, :] = U[i*m + j, :] * V[j*p + k, :]
                T_col[j] = T_flat[i*m + j, j*p + k, k*n + i]
            
            MTM = M.T @ M
            MTM += 1e-8 * np.eye(R)
            MTT = M.T
            try:
                x = np.linalg.solve(MTM, MTT.T @ T_col)
                W_new[k*n + i, :] = x
            except np.linalg.LinAlgError:
                W_new[k*n + i, :] = W[k*n + i, :]
    return W_new

def _try_als(rank, n=2, m=2, p=5, max_iters=50, restarts=5):
    T_flat = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_flat[i*m + j, j*p + k, k*n + i] = 1.0
                
    for r in range(restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            return None
            
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for it in range(max_iters):
            U = _als_step_U(U, V, W, T_flat)
            V = _als_step_V(U, V, W, T_flat)
            W = _als_step_W(U, V, W, T_flat)
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for rr in range(rank):
                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                        err += (val - 1.0)**2
            if err < 1e-6:
                # Snap to rational
                U_frac = [[Fraction(U[i, j]).limit_denominator(1000) for j in range(rank)] for i in range(n*m)]
                V_frac = [[Fraction(V[i, j]).limit_denominator(1000) for j in range(rank)] for i in range(m*p)]
                W_frac = [[Fraction(W[i, j]).limit_denominator(1000) for j in range(rank)] for i in range(p*n)]
                
                # Verify exactly
                if _verify(U_frac, V_frac, W_frac, n, m, p):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 16
    res = _try_als(16, n, m, p, max_iters=30, restarts=10)
    if res:
        return res
        
    # Try rank 17
    res = _try_als(17, n, m, p, max_iters=30, restarts=10)
    if res:
        return res
        
    # Fallback to rank 18 (Strassen + Schoolbook)
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen(U, V, W, col_offset, b_col_offset, c_col_offset):
        U[0][col_offset] = 1
        V[0 + b_col_offset][col_offset] = 1
        V[3 + b_col_offset][col_offset] = 1
        W[0 + c_col_offset][col_offset] = 1
        W[3 + c_col_offset][col_offset] = 1
        
        U[2][col_offset+1] = 1
        U[3][col_offset+1] = 1
        V[0 + b_col_offset][col_offset+1] = 1
        W[2 + c_col_offset][col_offset+1] = 1
        
        U[0][col_offset+2] = 1
        U[1][col_offset+2] = 1
        V[3 + b_col_offset][col_offset+2] = 1
        W[1 + c_col_offset][col_offset+2] = 1
        W[3 + c_col_offset][col_offset+2] = 1
        
        U[3][col_offset+3] = 1
        V[1 + b_col_offset][col_offset+3] = 1
        V[0 + b_col_offset][col_offset+3] = -1
        W[0 + c_col_offset][col_offset+3] = 1
        
        U[0][col_offset+4] = 1
        V[2 + b_col_offset][col_offset+4] = 1
        V[3 + b_col_offset][col_offset+4] = -1
        W[0 + c_col_offset][col_offset+4] = -1
        W[2 + c_col_offset][col_offset+4] = 1
        
        U[2][col_offset+5] = 1
        U[0][col_offset+5] = -1
        V[0 + b_col_offset][col_offset+5] = 1
        W[3 + c_col_offset][col_offset+5] = 1
        
        U[1][col_offset+6] = 1
        U[3][col_offset+6] = -1
        V[3 + b_col_offset][col_offset+6] = 1
        W[0 + c_col_offset][col_offset+6] = -1
        W[1 + c_col_offset][col_offset+6] = 1

    add_strassen(U, V, W, 0, 0, 0)
    add_strassen(U, V, W, 7, 2, 2)
    
    U[0][14] = 1
    V[0 + 4][14] = 1
    W[0 + 4][14] = 1
    
    U[1][15] = 1
    V[1 + 4][15] = 1
    W[0 + 4][15] = 1
    
    U[2][16] = 1
    V[0 + 4][16] = 1
    W[1 + 4][16] = 1
    
    U[3][17] = 1
    V[1 + 4][17] = 1
    W[1 + 4][17] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
        
    # Final fallback: Schoolbook
    rank_fb = n * m * p
    U = [[0] * rank_fb for _ in range(n * m)]
    V = [[0] * rank_fb for _ in range(m * p)]
    W = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
