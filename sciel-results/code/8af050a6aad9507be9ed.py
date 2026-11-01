"""Matrix multiplication 2,5,5 - rank minimization via vectorized ALS search."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized operations."""
        # Vectorized check
        # U: (nm, r), V: (mp, r), W: (np, r)
        # We need T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1
        # Reshape U, V, W to 3D tensors for broadcasting
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute sum over rank
        # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(T - 1))
        return max_err < 1e-8

    def tensor_error(U, V, W, rank):
        """Compute max absolute error of the tensor reconstruction."""
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.max(np.abs(T - 1))

    def als_solve_vectorized(U, V, W, rank, max_iters=500, tol=1e-10):
        """Vectorized Alternating Least Squares."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute target tensor matricizations?
        # T is fixed.
        # M1 (nm x mp*np), M2 (mp x np*nm), M3 (np x nm*mp)
        # T[i,j,k] = 1.
        # M1[i*m+j, (j'*p+k')*(p*n) + (i''*n+k'')] = 1 if j=j', i=i'', k=k''
        
        # Construct M1, M2, M3
        # M1 shape: (n*m, m*p * p*n)
        # Index: a = i*m+j. b = (j'*p+k')*(p*n) + i''*n+k''
        # M1[a, b] = 1 if j=j', i=i'', k=k''
        
        # This is too large to construct explicitly if dimensions grow, but for 2,5,5 it's small.
        # nm=10, mp=25, np=10. M1 is 10 x 250.
        
        M1 = np.zeros((n*m, m*p * p*n))
        M2 = np.zeros((m*p, p*n * n*m))
        M3 = np.zeros((p*n, n*m * m*p))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Term (i,j,k)
                    a = i*m + j
                    # For M1 (unfold 1): row a. Col corresponds to (j', k') and (i'', k'')
                    # We need j'=j, i''=i, k''=k
                    # Col index in M1: (j'*p+k')*(p*n) + i''*n+k''
                    # Here j'=j, i''=i, k''=k
                    col1 = (j*p+k)*(p*n) + i*n+k
                    M1[a, col1] = 1
                    
                    # For M2 (unfold 2): row b = j*p+k. Col corresponds to (i'', k'') and (i, j)
                    # We need i''=i, k''=k, and (i,j) is the term index?
                    # M2[b, col2] = 1 if i''=i, k''=k, and term is (i,j,k).
                    # Col index in M2: (i''*n+k'')*(n*m) + i*m+j
                    # Here i''=i, k''=k
                    b = j*p+k
                    col2 = (i*n+k)*(n*m) + i*m+j
                    M2[b, col2] = 1
                    
                    # For M3 (unfold 3): row c = k*n+i. Col corresponds to (i, j) and (j', k')
                    # We need i=i, j=j, and term is (i,j,k).
                    # Col index in M3: (i*m+j)*(m*p) + j*p+k
                    # Here i=i, j=j
                    c = k*n+i
                    col3 = (i*m+j)*(m*p) + j*p+k
                    M3[c, col3] = 1

        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: U = M1 @ (K_VW)^+
            # K_VW is (rank x (mp*np))
            # K_VW[c, :] = vec(V[c,:] tensor W[c,:])?
            # V is (mp, rank). W is (np, rank).
            # For each c, we need the vector v_c (mp) and w_c (np).
            # The product in the tensor is v_c[j] * w_c[k].
            # The matricization M1 uses the vector (v_c tensor w_c) flattened.
            # Order: (j, k) flattened to j*p+k.
            # So K_VW[c, j*p+k] = V[j, c] * W[k, c].
            
            V_flat = V.T # (rank, mp)
            W_flat = W.T # (rank, np)
            # K_VW[c, j*p+k] = V_flat[c, j] * W_flat[c, k]
            # This is an outer product for each c.
            # We can compute this using broadcasting.
            # V_flat[:, :, None] * W_flat[:, None, :] -> (rank, mp, np)
            K_VW = (V_flat[:, :, None] * W_flat[:, None, :]).reshape(rank, m*p * p*n)
            
            try:
                U = np.linalg.lstsq(K_VW.T, M1, rcond=None)[0]
            except:
                return None
                
            # Update V: V = M2 @ (K_WU)^+
            # K_WU[c, (i'',k'')*(nm) + i*m+j] = W[i'',c] * U[i,c]?
            # Wait, M2 row is j*p+k. Col is (i'',k'') and (i,j).
            # Term is U[i*m+j, c] * W[k*n+i, c]?
            # No, the tensor term is U[i,j,c] V[j,k,c] W[k,i,c].
            # For M2 (unfold 2), the "other" indices are 1 and 3.
            # Index 1: (i,j). Index 3: (k,i).
            # M2[j*p+k, (k*n+i) * (nm) + i*m+j] = 1?
            # Let's re-verify M2 construction.
            # M2[b, col2] where b=j*p+k.
            # col2 = (k*n+i)*(n*m) + i*m+j.
            # This matches the term (i,j,k).
            # So K_WU[c, col2] = W[k*n+i, c] * U[i*m+j, c].
            
            W_flat = W.T # (rank, np)
            U_flat = U.T # (rank, nm)
            # K_WU[c, (k*n+i)*(nm) + i*m+j] = W_flat[c, k*n+i] * U_flat[c, i*m+j]
            # Reshape W_flat to (rank, p, n) and U_flat to (rank, n, m)
            # W_3d = W_flat[:, :, :].reshape(rank, p, n)
            # U_3d = U_flat[:, :, :].reshape(rank, n, m)
            # We need to combine them such that index is (k, i, j).
            # K_WU[c, k, i, j] = W_3d[c, k, i] * U_3d[c, i, j]
            # Then flatten (k, i, j) to (k*n+i)*nm + i*m+j.
            
            W_3d = W_flat.reshape(rank, p, n)
            U_3d = U_flat.reshape(rank, n, m)
            # Broadcast to (rank, p, n, m)
            K_WU_3d = W_3d[:, :, :, None] * U_3d[:, None, :, :]
            K_WU = K_WU_3d.reshape(rank, p*n * n*m)
            
            try:
                V = np.linalg.lstsq(K_WU.T, M2, rcond=None)[0]
            except:
                return None
                
            # Update W: W = M3 @ (K_UV)^+
            # M3[c, col3] where c=k*n+i.
            # col3 = (i*m+j)*(m*p) + j*p+k.
            # K_UV[c, col3] = U[i*m+j, c] * V[j*p+k, c].
            
            U_flat = U.T # (rank, nm)
            V_flat = V.T # (rank, mp)
            U_3d = U_flat.reshape(rank, n, m)
            V_3d = V_flat.reshape(rank, m, p)
            # K_UV_3d[c, i, j, k] = U_3d[c, i, j] * V_3d[c, j, k]
            # Flatten (i, j, k) to (i*m+j)*mp + j*p+k.
            
            K_UV_3d = U_3d[:, :, :, None] * V_3d[:, None, :, :]
            K_UV = K_UV_3d.reshape(rank, n*m * m*p)
            
            try:
                W = np.linalg.lstsq(K_UV.T, M3, rcond=None)[0]
            except:
                return None
                
            err = tensor_error(U, V, W, rank)
            if err < tol:
                return U, V, W
                
        return None

    def round_to_half_int(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2

    rng = np.random.default_rng(42)
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    for r in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            result = als_solve_vectorized(U, V, W, r)
            if result is None:
                continue
            
            U_sol, V_sol, W_sol = result
            err = tensor_error(U_sol, V_sol, W_sol, r)
            
            if err < 1e-8:
                U_r = round_to_half_int(U_sol)
                V_r = round_to_half_int(V_sol)
                W_r = round_to_half_int(W_sol)
                
                if verify(U_r, V_r, W_r, r):
                    best_rank = r
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
        
    return best_result

# EVOLVE-BLOCK-END
