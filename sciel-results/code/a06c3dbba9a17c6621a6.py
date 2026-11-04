import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if i,j,k are valid indices for the multiplication term?
    # The tensor for matrix multiplication is defined such that:
    # C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # So T_{ijk} is 1 if the indices are valid, but actually, the tensor T has T_{ijk} = 1 
    # for all i in [0,n), j in [0,m), k in [0,p)? 
    # No, the standard matrix multiplication tensor T has T_{i,j,k} = 1 
    # if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In the schoolbook, each term (i,j,k) is a separate rank-1 tensor.
    # So the target tensor is a 3D array of shape (n, m, p) where T[i,j,k] = 1.
    
    T = np.ones((n, m, p), dtype=float)
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try to find a better rank
    # Lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    for R in range(20, rank_school):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank R
        result = try_find_decomposition(T, n, m, p, R, time_limit=10)
        if result is not None:
            U, V, W = result
            # Verify and format
            best_result = {
                "U": U,
                "V": V,
                "W": W
            }
            best_rank = R
            # Since we are increasing R, the first one found is the minimal?
            # We start from 20, so yes, 20 is the best possible.
            # If we find 20, we can return immediately.
            if R == 20:
                return best_result
            # If we find 21, we can continue to see if we missed 20? 
            # But 20 is the lower bound. If we didn't find 20 in 10s, it might be hard.
            # Let's just return the first one found to save time for verification?
            # Actually, if we find 21, it's better than 60.
            # But maybe 20 exists. 
            # Given the time, finding 20 is the "holy grail".
            # Let's try a few restarts for 20 specifically.
            
    return best_result

def try_find_decomposition(T, n, m, p, R, time_limit=10):
    """
    Attempts to find a rank-R decomposition of T.
    Returns (U_list, V_list, W_list) if successful, else None.
    U, V, W are lists of lists of half-integers.
    """
    start_time = time.time()
    
    # Number of restarts
    max_restarts = 5
    max_als_iter = 100
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
        
        # Initialize randomly
        U = np.random.randn(n, R)
        V = np.random.randn(m, R)
        W = np.random.randn(p, R)
        
        # ALS Loop
        for it in range(max_als_iter):
            if time.time() > start_time + time_limit:
                break
                
            # Solve for U: T_{i,j,k} = sum_r U_{i,r} (V_{j,r} W_{k,r})
            # Reshape T to (n, m*p)
            # For each i, we want U[i, :] to minimize sum_{j,k} (T[i,j,k] - sum_r U[i,r] V[j,r] W[k,r])^2
            # This is a linear least squares problem for U[i, :].
            # Let M_{i, r} = sum_{j,k} T[i,j,k] V[j,r] W[k,r] ? No.
            # The equation is: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
            # For fixed V, W, this is linear in U.
            # We can vectorize.
            # Let T_flat be shape (n, m*p).
            # Let V_W be a matrix of shape (m*p, R) where (V_W)[j*p+k, r] = V[j,r] W[k,r].
            # Then T_flat[i, :] = U[i, :] @ V_W.T
            # So U = T_flat @ inv(V_W.T @ V_W) ? Or use lstsq.
            
            T_flat = T.reshape(n, m * p)
            V_W = np.zeros((m * p, R))
            for j in range(m):
                for k in range(p):
                    V_W[j * p + k, :] = V[j, :] * W[k, :]
            
            # Solve U @ V_W.T = T_flat
            # U = T_flat @ pinv(V_W.T)
            U_new, _, _, _ = np.linalg.lstsq(V_W.T, T_flat, rcond=None)
            U = U_new
            
            # Solve for V
            # T_{i,j,k} = sum_r V_{j,r} (U_{i,r} W_{k,r})
            # Reshape T to (m, n*p)
            T_flat_m = T.transpose(1, 0, 2).reshape(m, n * p)
            U_W = np.zeros((n * p, R))
            for i in range(n):
                for k in range(p):
                    U_W[i * p + k, :] = U[i, :] * W[k, :]
            
            V_new, _, _, _ = np.linalg.lstsq(U_W.T, T_flat_m, rcond=None)
            V = V_new
            
            # Solve for W
            # T_{i,j,k} = sum_r W_{k,r} (U_{i,r} V_{j,r})
            # Reshape T to (p, n*m)
            T_flat_p = T.transpose(2, 0, 1).reshape(p, n * m)
            U_V = np.zeros((n * m, R))
            for i in range(n):
                for j in range(m):
                    U_V[i * m + j, :] = U[i, :] * V[j, :]
            
            W_new, _, _, _ = np.linalg.lstsq(U_V.T, T_flat_p, rcond=None)
            W = W_new
            
        # Check error
        # Reconstruct T
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            
        err = np.linalg.norm(T - T_recon)
        
        if err < 1e-9:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness
            T_recon_rounded = np.zeros((n, m, p))
            for r in range(R):
                T_recon_rounded += np.outer(np.outer(U_rounded[:, r], V_rounded[:, r]), W_rounded[:, r])
                
            if np.allclose(T, T_recon_rounded, atol=1e-9):
                # Format output
                # U is (n, R), need to flatten to (n*m, R)? 
                # Wait, the interface requires U to be (n*m, rank).
                # In my derivation, U was (n, R).
                # The standard schoolbook U has size n*m.
                # My U[i, r] corresponds to the coefficient of A_{ij}?
                # Let's check the schoolbook mapping.
                # Schoolbook: U[i*m+j, c] = 1.
                # My U[i, r] is not directly compatible with the (n*m, R) shape unless I expand it.
                # The tensor decomposition is T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}.
                # This implies that the term for index i in U is associated with row i of A.
                # But A has n rows and m columns.
                # The standard decomposition is T_{ijk} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}?
                # No, the standard tensor is $T = \sum_{c} u_c \otimes v_c \otimes w_c$.
                # $u_c$ is a vector in $\mathbb{R}^{n \times m}$, $v_c$ in $\mathbb{R}^{m \times p}$, $w_c$ in $\mathbb{R}^{p \times n}$.
                # So $U$ should be $(n*m, R)$, $V$ $(m*p, R)$, $W$ $(p*n, R)$.
                # My ALS used $U$ of size $(n, R)$. This is incorrect for the general tensor.
                # I need to use the full matrices.
                
                # Let's restart the ALS with correct shapes.
                pass # This block is invalid, need to rewrite try_find_decomposition

    return None

# Rewrite try_find_decomposition with correct shapes
def try_find_decomposition(T, n, m, p, R, time_limit=10):
    start_time = time.time()
    max_restarts = 5
    max_als_iter = 50
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
        
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        for it in range(max_als_iter):
            if time.time() > start_time + time_limit:
                break
            
            # Solve for U
            # T_{ijk} = sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r}
            # For fixed V, W, linear in U.
            # Let's flatten T to (n*m, p*n)? No.
            # The tensor T is indexed by i, j, k.
            # U is indexed by (i, j).
            # V is indexed by (j, k).
            # W is indexed by (k, i).
            
            # We can view the equation as:
            # T_{ijk} = sum_r U_{ij, r} (V_{jk, r} W_{ki, r})
            # Let M_{ij, r} = V_{jk, r} W_{ki, r}. This depends on k.
            # This is not a simple matrix multiplication for U.
            # We need to sum over k.
            # T_{ijk} is a 3D array.
            # Let's reshape T to (n*m, p*n) ? No, the indices don't align simply.
            # T[i, j, k].
            # If we fix i, j, we have a vector over k.
            # The equation is: T[i, j, :] = sum_r U[i, j, r] (V[j, :, r] * W[:, i, r])
            # Let V_W[j, k, r] = V[j, k, r] * W[k, i, r].
            # Then T[i, j, k] = sum_r U[i, j, r] V_W[j, k, r].
            # This is a linear system for U[i, j, :].
            
            # We can solve for all U at once.
            # Let T_flat be shape (n*m, p) ? No, W depends on i.
            # So we must loop over i.
            
            U_new = np.zeros_like(U)
            for i in range(n):
                # For this i, W[:, i, :] is fixed (size p x R)
                # V is size (m*p, R). Reshape to (m, p, R).
                V_3d = V.reshape(m, p, R)
                # Construct M of size (m*p, R) such that M[j*p+k, r] = V[j,k,r] * W[k,i,r]
                M = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        M[j * p + k, :] = V_3d[j, k, :] * W[k * n + i, :]
                
                # T[i, j, k] is a vector of length m*p if we flatten j,k.
                # T_i = T[i, :, :].reshape(m * p)
                T_i = T[i, :, :].reshape(m * p)
                
                # T_i = U[i, :, :] @ M.T  (where U[i, :, :] is 1 x R)
                # Solve U[i, :, :] = T_i @ pinv(M.T)
                U_row, _, _, _ = np.linalg.lstsq(M.T, T_i, rcond=None)
                U_new[i * m : (i+1) * m, :] = U_row.reshape(m, R)
            
            U = U_new
            
            # Similarly for V (loop over j)
            V_new = np.zeros_like(V)
            for j in range(m):
                # Fix j. U[:, j, :] is fixed (n x R)
                # W is (p*n, R).
                # Equation: T[i, j, k] = sum_r V[j, k, r] (U[i, j, r] W[k, i, r])
                # Let M[k, r] = U[i, j, r] W[k, i, r] ? No, sum over i.
                # T[:, j, k] = sum_r V[j, k, r] (sum_i U[i, j, r] W[k, i, r])
                # Let S[k, r] = sum_i U[i, j, r] W[k, i, r]
                # S is (p, R).
                # T_j = T[:, j, :].reshape(n * p)
                # T_j[i*p+k] = sum_r V[j, k, r] S[k, r]
                # This is not linear in V[j, k, r] in a simple way because S depends on V? No, S depends on U, W.
                # Wait, S[k, r] = sum_i U[i, j, r] W[k, i, r].
                # T_j is a vector of length n*p.
                # We want to solve for V[j, :, :] (size p x R).
                # The equation is T_j[i*p+k] = sum_r V[j, k, r] S[k, r].
                # This couples i and k.
                # Actually, for a fixed k, T[:, j, k] (vector of length n) = sum_r V[j, k, r] (U[:, j, r] * W[k*n + :, :].T? No)
                # Let's check indices.
                # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # For fixed j, k:
                # T[:, j, k] (length n) = sum_r V[j*p+k, r] (U[:, j, r] * W[k*n + :, r])
                # Let M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then T[:, j, k] = M @ V[j*p+k, :]
                # So V[j*p+k, :] = T[:, j, k] @ pinv(M.T)
                
                for k in range(p):
                    M = np.zeros((n, R))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    T_vec = T[:, j, k]
                    V_row, _, _, _ = np.linalg.lstsq(M.T, T_vec, rcond=None)
                    V_new[j * p + k, :] = V_row
            
            V = V_new
            
            # Similarly for W (loop over k)
            W_new = np.zeros_like(W)
            for k in range(p):
                # Fix k.
                # T[i, j, k] = sum_r W[k*n+i, r] (U[i*m+j, r] V[j*p+k, r])
                # For fixed k, i:
                # T[i, :, k] (length m) = sum_r W[k*n+i, r] (U[i, :, r] * V[:, k, r])
                # Let M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then T[i, :, k] = M @ W[k*n+i, :]
                
                for i in range(n):
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    T_vec = T[i, :, k]
                    W_row, _, _, _ = np.linalg.lstsq(M.T, T_vec, rcond=None)
                    W_new[k * n + i, :] = W_row
            
            W = W_new
            
        # Check error
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            U_col = U[:, r].reshape(n, m)
            V_col = V[:, r].reshape(m, p)
            W_col = W[:, r].reshape(p, n)
            T_recon += np.einsum('ij,jk,ki->ijk', U_col, V_col, W_col)
            
        err = np.linalg.norm(T - T_recon)
        
        if err < 1e-9:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_recon_rounded = np.zeros((n, m, p))
            for r in range(R):
                U_col = U_rounded[:, r].reshape(n, m)
                V_col = V_rounded[:, r].reshape(m, p)
                W_col = W_rounded[:, r].reshape(p, n)
                T_recon_rounded += np.einsum('ij,jk,ki->ijk', U_col, V_col, W_col)
                
            if np.allclose(T, T_recon_rounded, atol=1e-9):
                return [U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()]

    return None
