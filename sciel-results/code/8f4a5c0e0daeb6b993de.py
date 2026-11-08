"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses iterative randomized rank-revealing factorizations (ALS) to find a decomposition
with rank strictly less than the naive 72, aiming to beat the current 68.
"""

DIMENSIONS = [3, 3, 8]

import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Target Tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    T = np.ones((n, m, p))
    
    best_factors = None
    best_rank = 72 # Naive upper bound
    
    # Fallback: The known rank 68 solution
    # We will generate this if the search fails
    def get_fallback_68():
        U_terms = []
        V_terms = []
        W_terms = []
        
        def add_strassen(row_a_start, col_a_start, row_b_start, col_b_start):
            a = (row_a_start, col_a_start)
            b = (row_a_start, col_a_start + 1)
            c = (row_a_start + 1, col_a_start)
            d = (row_a_start + 1, col_a_start + 1)
            
            e = (row_b_start, col_b_start)
            f = (row_b_start, col_b_start + 1)
            g = (row_b_start + 1, col_b_start)
            h = (row_b_start + 1, col_b_start + 1)
            
            c00 = (row_a_start, col_b_start)
            c01 = (row_a_start, col_b_start + 1)
            c10 = (row_a_start + 1, col_b_start)
            c11 = (row_a_start + 1, col_b_start + 1)
            
            U_terms.append([(a, 1), (d, 1)])
            V_terms.append([(e, 1), (h, 1)])
            W_terms.append([(c00, 1), (c11, 1)])
            
            U_terms.append([(c, 1), (d, 1)])
            V_terms.append([(e, 1)])
            W_terms.append([(c10, 1), (c11, -1)])
            
            U_terms.append([(a, 1)])
            V_terms.append([(f, 1), (h, -1)])
            W_terms.append([(c01, 1), (c11, 1)])
            
            U_terms.append([(d, 1)])
            V_terms.append([(g, 1), (e, -1)])
            W_terms.append([(c00, 1), (c10, 1)])
            
            U_terms.append([(a, 1), (b, 1)])
            V_terms.append([(h, 1)])
            W_terms.append([(c00, -1), (c01, 1)])
            
            U_terms.append([(c, 1), (a, -1)])
            V_terms.append([(e, 1), (f, 1)])
            W_terms.append([(c11, 1)])
            
            U_terms.append([(b, 1), (d, -1)])
            V_terms.append([(g, 1), (h, 1)])
            W_terms.append([(c00, 1)])
            
        def add_direct(i, j, k):
            U_terms.append([((i, j), 1)])
            V_terms.append([((j, k), 1)])
            W_terms.append([((i, k), 1)])
            
        add_strassen(0, 0, 0, 0)
        add_strassen(0, 0, 0, 2)
        add_strassen(0, 0, 0, 4)
        add_strassen(0, 0, 0, 6)
        
        for j in [0, 1]:
            for k in range(8):
                add_direct(2, j, k)
                
        for i in [0, 1]:
            for k in range(8):
                add_direct(i, 2, k)
                
        for k in range(8):
            add_direct(2, 2, k)
            
        rank = len(W_terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            for (i, j), val in U_terms[r]:
                U[i * m + j, r] = val
            for (j, k), val in V_terms[r]:
                V[j * p + k, r] = val
            for (i, k), val in W_terms[r]:
                W[k * n + i, r] = val
                
        return U, V, W

    U_fallback, V_fallback, W_fallback = get_fallback_68()
    
    # Randomized Search
    # We try ranks from 67 down to 60
    for rank in range(67, 59, -1):
        found = False
        for restart in range(5): # 5 restarts per rank
            # Initialize U, V, W randomly
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS iterations
            for it in range(10):
                # Update U
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # For fixed i,j, we want U[i,j,:] to satisfy:
                # T[i,j,:] (vector over k) = sum_r U[i,j,r] (V[:,k,r] * W[k,i,r])?
                # Let's reshape for efficient solving.
                
                # We can solve for U by flattening V and W contributions.
                # For each i, j:
                # The contribution of U[i,j,r] is multiplied by (V[j,:,r] * W[:,i,r])?
                # V[j,k,r] is indexed by j,k,r. W[k,i,r] is indexed by k,i,r.
                # Let's precompute the "effective" vector for each r and (i,j).
                # This is expensive.
                
                # Alternative: Use numpy einsum or reshape.
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Update U:
                # For each r, U[:,r] is a vector of length nm.
                # It is coupled with V and W.
                # Let's fix V and W.
                # We want to minimize || T - sum_r U_r V_r W_r ||_F
                # This is not linear in U if V,W are fixed?
                # T[i,j,k] = sum_r U[i,j,r] (V[j,k,r] W[k,i,r])
                # Let M_{i,j, k, r} = V[j,k,r] W[k,i,r].
                # Then T[i,j,k] = sum_r U[i,j,r] M[i,j,k,r].
                # For each i,j, we have a linear system for U[i,j,:].
                # But M depends on i,j,k,r.
                
                # Simpler ALS approach:
                # 1. Fix V, W. Compute B_{i,j,k,r} = V[j,k,r] * W[k,i,r].
                #    Reshape B to shape (n*m, p, rank).
                #    T is (n, m, p). Reshape T to (n*m, p).
                #    We want U (n*m, rank) such that T[i,j,k] = sum_r U[i,j,r] B[i,j,k,r].
                #    This is a linear least squares problem for each (i,j)?
                #    No, it's a tensor regression.
                #    T_flat (nm * p) = U (nm, rank) * B_perm (rank, nm * p)?
                #    Let's check indices.
                #    T[i,j,k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]).
                #    Let A_{i,j, k, r} = V[j*p+k, r] * W[k*n+i, r].
                #    We can view this as: for each (i,j), we have a vector T[i,j,:] of length p.
                #    And we have rank vectors B_{i,j,:,r} of length p.
                #    We want U[i,j,:] to be the coefficients.
                #    This is underdetermined if rank > p, but we can solve via pinv or least squares.
                #    Actually, for each (i,j), we solve:
                #    T[i,j,:] (1 x p) = U[i,j,:] (1 x rank) * B_{i,j} (rank x p).
                #    U[i,j,:] = T[i,j,:] * pinv(B_{i,j}).
                
                # Let's implement this.
                
                # Prepare B for U update
                # B[i, j, k, r] = V[j, k, r] * W[k, i, r]
                # V is (m, p, rank) -> V[j, k, r]
                # W is (p, n, rank) -> W[k, i, r]
                
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                # B_4d[i, j, k, r]
                # We can use broadcasting.
                # V_3d[j, k, r] and W_3d[k, i, r]
                # We need to align j and i.
                # V_3d is (m, p, rank).
                # W_3d is (p, n, rank).
                # We want B[i, j, k, r].
                # B[i, j, k, r] = V_3d[j, k, r] * W_3d[k, i, r].
                
                # Reshape V_3d to (1, m, p, rank)
                # Reshape W_3d to (p, n, 1, rank)? No.
                # W_3d[k, i, r].
                # Let's transpose W_3d to (n, p, rank) -> W_T[i, k, r].
                W_T = W_3d.transpose(1, 0, 2) # (n, p, rank)
                
                # Now V_3d is (m, p, rank).
                # W_T is (n, p, rank).
                # We want B[i, j, k, r] = V_3d[j, k, r] * W_T[i, k, r].
                # Multiply along k? No, k is an index.
                # We need to compute this for all i, j, k, r.
                # B[i, j, k, r] = V_3d[j, k, r] * W_T[i, k, r].
                
                # V_3d: (m, p, rank) -> (1, m, p, rank)
                # W_T: (n, p, rank) -> (n, 1, p, rank)
                # Product: (n, m, p, rank)
                B_4d = V_3d[None, :, :, :] * W_T[:, None, :, :]
                
                # B_4d shape: (n, m, p, rank)
                # Reshape to (n*m, p, rank)
                B_3d = B_4d.reshape(n * m, p, rank)
                
                # T_2d: (n*m, p)
                T_2d = T.reshape(n * m, p)
                
                # Solve for U: U[i*m+j, :] = T_2d[i*m+j, :] @ pinv(B_3d[i*m+j, :, :])
                # B_3d is (nm, p, rank).
                # We can use np.linalg.lstsq or pinv.
                # Since p=8, rank ~ 60, p < rank, pinv is (rank, p).
                # U (nm, rank) = T_2d (nm, p) @ pinv(B_3d) (nm, rank, p) -> (nm, rank)
                
                # Efficiently:
                # pinv for each slice.
                # Or use einsum?
                # U = np.einsum('ip,ijp->ij', T_2d, np.linalg.pinv(B_3d, hermitian=False))
                # But pinv is expensive.
                
                # Since p is small (8), we can solve linear systems.
                # B_3d[i, :, :] is (p, rank).
                # We want U_i (rank,) such that B_3d[i] @ U_i = T_2d[i].
                # This is underdetermined. Minimum norm solution is U_i = B_3d[i].T @ pinv(B_3d[i] @ B_3d[i].T) @ T_2d[i]?
                # Or just use lstsq.
                
                U_new = np.zeros((n * m, rank))
                for idx in range(n * m):
                    B_slice = B_3d[idx] # (p, rank)
                    T_slice = T_2d[idx] # (p,)
                    # Solve B_slice @ U_slice = T_slice
                    # U_slice = pinv(B_slice) @ T_slice
                    U_new[idx] = np.linalg.pinv(B_slice) @ T_slice
                
                U = U_new
                
                # Update V
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Fix U, W.
                # For each j, k:
                # T[:, j, k] (vector over i) = sum_r V[j,k,r] (U[:,j,r] * W[k,:,r])
                # Let C_{j, k, i, r} = U[i,j,r] * W[k,i,r].
                # U is (n, m, rank) -> U[i, j, r].
                # W is (p, n, rank) -> W[k, i, r].
                
                U_3d = U.reshape(n, m, rank)
                W_3d = W.reshape(p, n, rank)
                
                # C[j, k, i, r] = U_3d[i, j, r] * W_3d[k, i, r]
                # U_3d: (n, m, rank) -> (1, m, n, rank)? No.
                # We need to align i.
                # U_3d[i, j, r].
                # W_3d[k, i, r].
                # C[j, k, i, r] = U_3d[i, j, r] * W_3d[k, i, r].
                
                # U_3d.T is (m, n, rank). Let U_T[j, i, r].
                U_T = U_3d.transpose(1, 0, 2) # (m, n, rank)
                
                # W_3d is (p, n, rank).
                # C[j, k, i, r] = U_T[j, i, r] * W_3d[k, i, r].
                # U_T: (m, n, rank) -> (m, 1, n, rank)
                # W_3d: (p, n, rank) -> (1, p, n, rank)
                C_4d = U_T[:, None, :, :] * W_3d[None, :, :, :] # (m, p, n, rank)
                
                # Reshape to (m*p, n, rank)
                C_3d = C_4d.reshape(m * p, n, rank)
                
                # T_2d_V: (m*p, n)
                # T[i, j, k]. We want to group by j, k.
                # T is (n, m, p).
                # Transpose to (m, p, n).
                T_V_3d = T.transpose(1, 2, 0) # (m, p, n)
                T_V_2d = T_V_3d.reshape(m * p, n)
                
                V_new = np.zeros((m * p, rank))
                for idx in range(m * p):
                    C_slice = C_3d[idx] # (n, rank)
                    T_slice = T_V_2d[idx] # (n,)
                    V_new[idx] = np.linalg.pinv(C_slice) @ T_slice
                
                V = V_new
                
                # Update W
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Fix U, V.
                # For each k, i:
                # T[i, :, k] (vector over j) = sum_r W[k,i,r] (U[i,:,r] * V[:,k,r])
                # Let D_{k, i, j, r} = U[i, j, r] * V[j, k, r].
                
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                
                # D[k, i, j, r] = U_3d[i, j, r] * V_3d[j, k, r]
                # U_3d: (n, m, rank) -> (n, m, 1, rank)?
                # V_3d: (m, p, rank) -> (1, m, p, rank)?
                # We need to align j.
                # U_3d[i, j, r].
                # V_3d[j, k, r].
                
                # U_3d: (n, m, rank) -> (n, m, 1, rank)
                # V_3d: (m, p, rank) -> (1, m, p, rank)
                # Product: (n, m, p, rank)
                D_4d = U_3d[:, :, None, :] * V_3d[None, :, :, :] # (n, m, p, rank)
                
                # We want D[k, i, j, r].
                # Current D_4d is [i, j, k, r].
                # Transpose to [k, i, j, r].
                D_4d_T = D_4d.transpose(2, 0, 1, 3) # (p, n, m, rank)
                
                # Reshape to (p*n, m, rank)
                D_3d = D_4d_T.reshape(p * n, m, rank)
                
                # T_2d_W: (p*n, m)
                # T[i, j, k]. Group by k, i.
                # T is (n, m, p).
                # Transpose to (p, n, m).
                T_W_3d = T.transpose(2, 0, 1) # (p, n, m)
                T_W_2d = T_W_3d.reshape(p * n, m)
                
                W_new = np.zeros((p * n, rank))
                for idx in range(p * n):
                    D_slice = D_3d[idx] # (m, rank)
                    T_slice = T_W_2d[idx] # (m,)
                    W_new[idx] = np.linalg.pinv(D_slice) @ T_slice
                
                W = W_new
                
            # Check residual
            # T_calc[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # Use einsum
            T_calc = np.einsum('ijr,jkr,ikir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
            residual = np.linalg.norm(T - T_calc)
            
            if residual < 1e-6:
                # Attempt rounding
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_calc_r = np.einsum('ijr,jkr,ikir->ijk', U_r.reshape(n,m,rank), V_r.reshape(m,p,rank), W_r.reshape(p,n,rank))
                residual_r = np.linalg.norm(T - T_calc_r)
                
                if residual_r < 1e-9:
                    # Success
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                else:
                    # Try integer rounding
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    T_calc_r = np.einsum('ijr,jkr,ikir->ijk', U_r.reshape(n,m,rank), V_r.reshape(m,p,rank), W_r.reshape(p,n,rank))
                    residual_r = np.linalg.norm(T - T_calc_r)
                    if residual_r < 1e-9:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        
    # If no better rank found, return fallback
    return {
        "U": U_fallback.tolist(),
        "V": V_fallback.tolist(),
        "W": W_fallback.tolist()
    }
