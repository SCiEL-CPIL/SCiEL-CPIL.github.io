"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The decomposition is defined by three matrices U, V, W of shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Such that for all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_U, rank), dtype=int)
        V = np.zeros((dim_V, rank), dtype=int)
        W = np.zeros((dim_W, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Initialize with schoolbook
    U = np.zeros((dim_U, n * m * p), dtype=int)
    V = np.zeros((dim_V, n * m * p), dtype=int)
    W = np.zeros((dim_W, n * m * p), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    current_rank = n * m * p
    max_rank = current_rank
    
    # Helper to verify
    def verify(U, V, W, r):
        # Check half-integer
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        # Check tensor equation
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        # We can check this by reconstructing the tensor
        # T[i,j,k] = sum_c ...
        # Using vectorization:
        # U_sub = U[i*m+j, :]
        # V_sub = V[j*p+k, :]
        # W_sub = W[k*n+i, :]
        # Sum over c of U_sub * V_sub * W_sub
        # This is O(n*m*p*r)
        # For r=84, n*m*p=84, it's 84*84*84 ~ 590k ops. Fast.
        
        # Create index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        U_rows = i_idx * m + j_idx
        V_rows = j_idx * p + k_idx
        W_rows = k_idx * n + i_idx
        
        U_sub = U[U_rows, :]
        V_sub = V[V_rows, :]
        W_sub = W[W_rows, :]
        
        prod = U_sub * V_sub * W_sub
        sums = np.sum(prod, axis=1)
        
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        return True

    # Local Search: Compression
    # Try to find subsets of columns that sum to a rank-1 tensor
    # and replace them with a single column.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try subset sizes 2 and 3
        for k in [2, 3]:
            if time.time() > deadline - 5:
                break
            num_tries = 2000 if k == 2 else 500
            for _ in range(num_tries):
                if time.time() > deadline - 2:
                    break
                
                # Pick random subset of columns
                cols = np.random.choice(current_rank, size=k, replace=False)
                
                # Extract sub-tensor
                # T[i,j,k] = sum_{c in cols} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # We can compute this efficiently
                U_sub = U[:, cols]
                V_sub = V[:, cols]
                W_sub = W[:, cols]
                
                # T is (n, m, p)
                T = np.zeros((n, m, p))
                for c in range(k):
                    u_vec = U_sub[:, c].reshape(n, m)
                    v_vec = V_sub[:, c].reshape(m, p)
                    w_vec = W_sub[:, c].reshape(p, n)
                    # Term is u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
                    # This is an outer product of u_vec, v_vec, w_vec? No, indices are coupled.
                    # T[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
                    # This is a 3-way product with coupled indices.
                    # We can use einsum: 'ij,jk,ki->ijk'
                    T += np.einsum('ij,jk,ki->ijk', u_vec, v_vec, w_vec)
                
                # Check if T is rank 1
                # Unfold to matrix of shape (n, m*p)
                M = T.reshape(n, m * p)
                rank_T = np.linalg.matrix_rank(M)
                
                if rank_T == 0:
                    # T is zero. We can remove these columns.
                    # But wait, if T is zero, it means the sum of these terms is zero.
                    # So we can remove them.
                    mask = np.ones(current_rank, dtype=bool)
                    mask[cols] = False
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    current_rank -= k
                    improved = True
                    break # Restart search with new rank
                
                elif rank_T == 1:
                    # Check if T can be represented by half-integers
                    # Extract factors
                    # T[i,j,k] = u[i] v[j] w[k]
                    u = T[:, 0, 0]
                    v = T[0, :, 0]
                    w = T[0, 0, :]
                    
                    # Check consistency
                    # T[i,j,k] == u[i] * v[j] * w[k]
                    # Use einsum
                    T_recon = np.einsum('i,j,k->ijk', u, v, w)
                    if np.allclose(T, T_recon, atol=1e-9):
                        # Now check if u, v, w can be scaled to half-integers
                        # We need alpha, beta, gamma such that alpha*beta*gamma = 1
                        # and 2*alpha*u, 2*beta*v, 2*gamma*w are integers.
                        
                        # Simple heuristic: check if u, v, w are already half-integers
                        # or if they are integers.
                        # Since T is integer, u,v,w are rational.
                        
                        # Try to find a scaling.
                        # Let's just check if 2*u, 2*v, 2*w are close to integers.
                        if np.allclose(2*u, np.round(2*u), atol=1e-9) and \
                           np.allclose(2*v, np.round(2*v), atol=1e-9) and \
                           np.allclose(2*w, np.round(2*w), atol=1e-9):
                            
                            # Replace cols with new column
                            new_U = np.round(2*u) / 2.0
                            new_V = np.round(2*v) / 2.0
                            new_W = np.round(2*w) / 2.0
                            
                            # Insert new column
                            U = np.column_stack((U[:, ~np.isin(np.arange(current_rank), cols)], new_U))
                            V = np.column_stack((V[:, ~np.isin(np.arange(current_rank), cols)], new_V))
                            W = np.column_stack((W[:, ~np.isin(np.arange(current_rank), cols)], new_W))
                            current_rank = current_rank - k + 1
                            improved = True
                            break
                        
                        # If not half-integers, try scaling by 2, 1/2, etc.
                        # This is complex, so we skip for now.
                    
            if improved:
                break
        
        # If no improvement, try a random restart?
        # Or just stop.
        if not improved:
            # Maybe try a different strategy?
            # For now, we stop if no compression is found.
            break

    # Final verification
    if verify(U, V, W, current_rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return get_schoolbook()

# EVOLVE-BLOCK-END
