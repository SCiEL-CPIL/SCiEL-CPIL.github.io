import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Fallback: Schoolbook rank 36
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_schoolbook.tolist(),
        "V": V_schoolbook.tolist(),
        "W": W_schoolbook.tolist()
    }
    best_rank = rank_schoolbook
    
    # Function to verify decomposition
    def verify_decomp(U, V, W, n, m, p):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r U[:,r] V[:,r]^T W[:,r]^T == Tensor
        # Tensor T_{i,j,k} = 1 if consistent
        # Actually, we can check if the bilinear form matches.
        # Let's just check a few random matrices or all basis vectors.
        # Since R is small, we can check all basis vectors of A and B.
        # A has n*m dims, B has m*p dims.
        # For each basis vector of A (e_a) and B (e_b), compute C = sum_r U[a,r] V[b,r] W[:,r].
        # Check if C matches the expected multiplication.
        R = U.shape[1]
        # Expected C for A=e_a, B=e_b:
        # a = i*m + j, b = j'*p + k
        # If j == j', C[i, k] = 1. Else 0.
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                if j == j2:
                    # C should be 1 at (i, k) and 0 elsewhere
                    C_expected = np.zeros(p * n)
                    C_expected[k * n + i] = 1
                    # Compute C from decomposition
                    # C = sum_r U[a, r] * V[b, r] * W[:, r]
                    coeffs = U[a, :] * V[b, :]
                    C_computed = coeffs @ W
                    if not np.allclose(C_computed, C_expected, atol=1e-9):
                        return False
                else:
                    # C should be 0
                    coeffs = U[a, :] * V[b, :]
                    C_computed = coeffs @ W
                    if not np.allclose(C_computed, 0, atol=1e-9):
                        return False
        return True

    # Function to find rank R decomposition for 2x3x3
    def find_233_decomp(R, time_limit=10):
        # Tensor for 2x3x3
        n1, m1, p1 = 2, 3, 3
        dim_a = n1 * m1
        dim_b = m1 * p1
        dim_c = p1 * n1
        
        # Build target tensor T (dim_a, dim_b, dim_c)
        T = np.zeros((dim_a, dim_b, dim_c))
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    a = i * m1 + j
                    b = j * p1 + k
                    c = k * n1 + i
                    T[a, b, c] = 1
        
        # ALS for tensor decomposition
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(20):
            if time.time() > start_time + time_limit:
                break
            U = np.random.randn(dim_a, R)
            V = np.random.randn(dim_b, R)
            W = np.random.randn(dim_c, R)
            
            for it in range(200):
                # Update U
                # T = sum_r U[:,r] V[:,r]^T W[:,r]^T
                # For each a, T[a,:,:] = sum_r U[a,r] V[:,r]^T W[:,r]^T
                # Flatten T[a,:,:] to vector of length dim_b*dim_c
                # Let M_a be dim_b x dim_c matrix.
                # M_a = sum_r U[a,r] (V[:,r] W[:,r]^T)
                # This is a linear system for U[a,:].
                # Let's use a simpler approach: 
                # Residual R = T - approx
                # Update U[a,:] by solving least squares.
                # Actually, standard ALS:
                # Fix V, W. Compute M_r = V[:,r] W[:,r]^T.
                # T[a,b,c] = sum_r U[a,r] M_r[b,c].
                # For fixed a, T[a,:,:] = sum_r U[a,r] M_r.
                # This is a linear regression of T[a,:,:] on the vectors vec(M_r).
                
                # Precompute M_r vectors
                M_vecs = np.array([ (V[:,r] * W[:,r][None,:]).flatten() for r in range(R) ]) # (R, dim_b*dim_c)
                # T_flat_a = T[a,:,:].flatten()
                # Solve U[a,:] such that M_vecs.T @ U[a,:] ~= T_flat_a
                # U[a,:] = (M_vecs @ M_vecs.T)^-1 M_vecs @ T_flat_a
                # Actually, U[a,:] is the coefficient vector.
                # Let's use np.linalg.lstsq
                for a in range(dim_a):
                    T_flat = T[a, :, :].flatten()
                    # Solve M_vecs.T @ U[a,:] = T_flat
                    # M_vecs is (R, N). We want x (R,) such that M_vecs.T @ x = T_flat.
                    # This is underdetermined if R < N. 
                    # Least squares: x = (M_vecs.T @ M_vecs)^-1 M_vecs.T @ T_flat
                    # Or use lstsq on M_vecs.T (N x R) and T_flat (N,)
                    # np.linalg.lstsq(M_vecs.T, T_flat) gives x
                    U[a, :], residuals, rank, s = np.linalg.lstsq(M_vecs.T, T_flat, rcond=None)
                
                # Update V
                # Fix U, W.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b, T[:,b,:] = sum_r V[b,r] (U[:,r] W[:,r]^T)
                # Let N_r = U[:,r] W[:,r]^T.
                # T[:,b,:] = sum_r V[b,r] N_r.
                N_vecs = np.array([ (U[:,r] * W[:,r][None,:]).flatten() for r in range(R) ]) # (R, dim_a*dim_c)
                for b in range(dim_b):
                    T_flat = T[:, b, :].flatten()
                    V[b, :], residuals, rank, s = np.linalg.lstsq(N_vecs.T, T_flat, rcond=None)
                
                # Update W
                # Fix U, V.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed c, T[:,:,c] = sum_r W[c,r] (U[:,r] V[:,r]^T)
                # Let P_r = U[:,r] V[:,r]^T.
                # T[:,:,c] = sum_r W[c,r] P_r.
                P_vecs = np.array([ (U[:,r] * V[:,r][None,:]).flatten() for r in range(R) ]) # (R, dim_a*dim_b)
                for c in range(dim_c):
                    T_flat = T[:, :, c].flatten()
                    W[c, :], residuals, rank, s = np.linalg.lstsq(P_vecs.T, T_flat, rcond=None)
            
            # Calculate error
            approx = np.zeros((dim_a, dim_b, dim_c))
            for r in range(R):
                approx += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
            err = np.linalg.norm(T - approx)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-6:
                    break
        
        if best_err < 1e-4:
            # Try to round to nearest 0.5
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify rounded version
            if verify_decomp(U_rounded, V_rounded, W_rounded, n1, m1, p1):
                return U_rounded, V_rounded, W_rounded, R
            else:
                # Try to verify floating point version? 
                # The prompt requires integer/half-integer.
                # If rounded fails, we can't use it.
                return None, None, None, R
        else:
            return None, None, None, R

    # Try ranks 11, 12, 13 for 2x3x3
    for R in [11, 12, 13]:
        U1, V1, W1, r_found = find_233_decomp(R, time_limit=20)
        if U1 is not None:
            # Construct 2x3x6 decomposition
            # U_total: (6, 2R)
            # V_total: (18, 2R)
            # W_total: (12, 2R)
            
            U_total = np.zeros((6, 2 * R))
            V_total = np.zeros((18, 2 * R))
            W_total = np.zeros((12, 2 * R))
            
            # First block (B1)
            U_total[:, :R] = U1
            V_total[:9, :R] = V1
            W_total[:6, :R] = W1
            
            # Second block (B2)
            U_total[:, R:] = U1
            V_total[9:, R:] = V1
            W_total[6:, R:] = W1
            
            # Verify full decomposition
            if verify_decomp(U_total, V_total, W_total, n, m, p):
                best_rank = 2 * R
                best_result = {
                    "U": U_total.tolist(),
                    "V": V_total.tolist(),
                    "W": W_total.tolist()
                }
                break # Found a good one, stop searching for lower rank
            
    return best_result
