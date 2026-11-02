import numpy as np
from fractions import Fraction
from itertools import product
import time

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = 145  # 145 seconds to be safe

    # Build the tensor for matrix multiplication C = A * B
    # A is n x m, B is m x p, C is n x p
    # T_{(i,j),k,l} = delta_{i,j} * delta_{k,l} for C_{i,l} = sum_j A_{i,j} B_{j,l}
    # Flatten indices: A has n*m=8 entries, B has m*p=24 entries, C has p*n=12 entries
    # Tensor T is 8x24x12
    
    # Wait, the interface says:
    # U: (n*m, rank) -> (8, rank)
    # V: (m*p, rank) -> (24, rank)
    # W: (p*n, rank) -> (12, rank)
    # And "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W is indexed by (k, i) where k is the column of C (0..p-1) and i is the row of C (0..n-1).
    # So W index = k * n + i.
    # The tensor T should be defined such that:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a is index in A (i*m + j), b is index in B (j*p + l), c is index in W (l*n + i).
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                # A index: i*m + j
                # B index: j*p + l
                # W index: l*n + i
                T[i*m+j, j*p+l, l*n+i] = 1

    # Helper to check if a decomposition is valid
    def check_decomposition(U, V, W, r):
        T_recon = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            outer_UV = np.outer(U[:, r_idx], V[:, r_idx])
            T_recon += outer_UV[:, :, None] * W[None, None, :]
        
        if not np.allclose(T_recon, T, atol=1e-6):
            return False
        
        # Convert to fractions
        U_frac = [[Fraction(U[i, j]).limit_denominator(10000) for j in range(r)] for i in range(n*m)]
        V_frac = [[Fraction(V[i, j]).limit_denominator(10000) for j in range(r)] for i in range(m*p)]
        W_frac = [[Fraction(W[i, j]).limit_denominator(10000) for j in range(r)] for i in range(p*n)]
        
        # Verify exactly
        T_exact = np.zeros((n*m, m*p, p*n), dtype=object)
        for r_idx in range(r):
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        T_exact[i, j, k] += U_frac[i][r_idx] * V_frac[j][r_idx] * W_frac[k][r_idx]
        
        T_target = np.zeros((n*m, m*p, p*n), dtype=object)
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    T_target[i*m+j, j*p+l, l*n+i] = 1
        
        if np.array_equal(T_exact, T_target):
            return (U_frac, V_frac, W_frac)
        return False

    def try_rank(r):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42 + r)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for iteration in range(100):
            if time.time() > deadline - 10:
                return None
            
            # Update W
            UV = np.zeros((r, n*m, m*p))
            for r_idx in range(r):
                UV[r_idx] = np.outer(U[:, r_idx], V[:, r_idx])
            UV_flat = UV.reshape(r, -1)
            
            W_new = np.zeros((p*n, r))
            for c in range(p*n):
                T_c = T[:, :, c].flatten()
                try:
                    W_c, _, _, _ = np.linalg.lstsq(UV_flat.T, T_c, rcond=None)
                    W_new[c, :] = W_c
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
            # Update U
            VW = np.zeros((r, m*p, p*n))
            for r_idx in range(r):
                VW[r_idx] = np.outer(V[:, r_idx], W[:, r_idx])
            VW_flat = VW.reshape(r, -1)
            
            U_new = np.zeros((n*m, r))
            for a in range(n*m):
                T_a = T[a, :, :].flatten()
                try:
                    U_a, _, _, _ = np.linalg.lstsq(VW_flat.T, T_a, rcond=None)
                    U_new[a, :] = U_a
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Update V
            UW = np.zeros((r, n*m, p*n))
            for r_idx in range(r):
                UW[r_idx] = np.outer(U[:, r_idx], W[:, r_idx])
            UW_flat = UW.reshape(r, -1)
            
            V_new = np.zeros((m*p, r))
            for b in range(m*p):
                T_b = T[:, b, :].flatten()
                try:
                    V_b, _, _, _ = np.linalg.lstsq(UW_flat.T, T_b, rcond=None)
                    V_new[b, :] = V_b
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Check convergence
            T_recon = np.zeros((n*m, m*p, p*n))
            for r_idx in range(r):
                outer_UV = np.outer(U[:, r_idx], V[:, r_idx])
                T_recon += outer_UV[:, :, None] * W[None, None, :]
            
            if np.allclose(T_recon, T, atol=1e-6):
                result = check_decomposition(U, V, W, r)
                if result is not False:
                    return result
        
        return None

    # Try ranks from 36 to 41
    for r in range(36, 42):
        if time.time() > deadline - 5:
            break
        result = try_rank(r)
        if result is not None:
            U_frac, V_frac, W_frac = result
            return {"U": U_frac, "V": V_frac, "W": W_frac}

    # Fall back to the Strassen construction (rank 42)
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M2
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        # M3
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M4
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        # M5
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        # M6
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col

    for k in range(3):
        A_L_idx = [0, 1, 4, 5]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        
        A_R_idx = [2, 3, 6, 7]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)

    return {"U": U, "V": V, "W": W}
