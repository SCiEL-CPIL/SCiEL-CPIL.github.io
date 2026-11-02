import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i'
    # Actually, the standard tensor definition for matmul C=AB:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T_{(i,j), (j',k), (k',i')} = delta_{j,j'} delta_{k,k'} delta_{i,i'}
    
    # We want to find U, V, W such that T = sum_r U_r \otimes V_r \otimes W_r
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    # Flatten indices
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    # Build the tensor T as a list of non-zero entries
    # T is sparse.
    # We can represent the decomposition constraint as:
    # For all (u_idx, v_idx, w_idx), sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r] = T[u_idx, v_idx, w_idx]
    
    # Non-zero entries of T:
    # For each i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    # T[u_idx, v_idx, w_idx] = 1
    
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                non_zeros.append((u_idx, v_idx, w_idx, 1))
                
    # Also, all other entries are 0.
    # We only need to satisfy the non-zero constraints AND the zero constraints.
    # To check validity, we can reconstruct the full tensor or just check the non-zeros and a sample of zeros.
    # Better: check all entries.
    
    def check_decomposition(U, V, W):
        # U, V, W are numpy arrays of shape (rows, R)
        R = U.shape[1]
        # Reconstruct T
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can compute this efficiently for all non-zero and zero entries.
        # But to be safe, let's check the full tensor if it's small enough.
        # n*m=4, m*p=10, p*n=10. Total entries 400.
        # R is small.
        # Compute full T_hat
        # T_hat[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is a tensor contraction.
        # We can do: T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Construct true T
        T_true = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_true[u_idx, v_idx, w_idx] = 1
                    
        # Check if T_hat matches T_true exactly (within integer tolerance)
        # Since we snap to integers, we should check integer equality.
        # But floating point errors might creep in.
        # We snap to nearest integer.
        # So T_hat should be very close to integers.
        
        # Check max diff
        if not np.allclose(T_hat, T_true, atol=1e-9):
            return False
        return True

    def snap_to_int(A):
        return np.rint(A)

    # Search for minimal rank
    # Try ranks from 10 to 20
    # We know 20 works.
    # We want to find lower.
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_result = None
    
    # Initial guess: Schoolbook (Rank 20)
    # We can try to improve it.
    
    # Try ranks 16, 17, 18, 19, 20
    for R in range(16, 21):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + trial + R)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS Optimization
            # Minimize || T - sum U_r \otimes V_r \otimes W_r ||_F
            # Update U: For each r, U[:, r] = argmin_U || T - sum U_r \otimes V_r \otimes W_r ||
            # This is a linear least squares problem for each r.
            # T = sum_r U_r \otimes V_r \otimes W_r
            # Unroll T into a matrix?
            # T is a 3-way tensor.
            # We can use the property:
            # Fix V, W. Find U.
            # T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
            # For a fixed r, the term is U_{u,r} * (V_{v,r} W_{w,r})
            # This is not a simple linear system for U_{u,r} because it's coupled with other r.
            # However, if we fix V and W, we can solve for U using least squares.
            # Let's reshape T into a matrix of size (n*m) x (m*p * p*n)?
            # T[u, v, w] = sum_r U[u, r] * (V[v, r] * W[w, r])
            # Let M_r be the outer product V[:, r] \otimes W[:, r].
            # Then T = sum_r U[:, r] \otimes M_r
            # We can stack M_r into a matrix M of size (m*p*p*n) x R?
            # No, M_r has size (m*p) x (p*n).
            # Let's flatten T into a vector t of size n*m * m*p * p*n.
            # t = sum_r u_r \otimes m_r
            # This is a linear system: A * coeffs = t
            # Where A is a block matrix?
            
            # Simpler: Use numpy lstsq.
            # For U update:
            # We want to minimize || T - sum_r U_r \otimes V_r \otimes W_r ||_F^2
            # This is equivalent to minimizing over U.
            # Let's create a matrix B of size (m*p * p*n) x R
            # B[:, r] = vec(V[:, r] \otimes W[:, r]) ? No, that's not right.
            # T is (n*m, m*p, p*n).
            # Reshape T to (n*m) x (m*p * p*n).
            # T_mat[u, (v, w)] = T[u, v, w]
            # T_mat[u, (v, w)] = sum_r U[u, r] * V[v, r] * W[w, r]
            # Let C_r be a vector of size m*p * p*n, where C_r[(v, w)] = V[v, r] * W[w, r].
            # Then T_mat = U * C^T ?
            # T_mat[u, z] = sum_r U[u, r] C_r[z]
            # So T_mat = U @ C_mat
            # Where C_mat is R x (m*p * p*n).
            # C_mat[r, z] = V[v, r] * W[w, r] for z corresponding to (v, w).
            
            # Let's build C_mat.
            # z index: v * (p*n) + w
            # v in 0..m*p-1, w in 0..p*n-1
            
            for step in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                # Build C_mat
                C_mat = np.zeros((R, m*p * p*n))
                for r in range(R):
                    # V[:, r] is (m*p,), W[:, r] is (p*n,)
                    # Outer product V[:, r] \otimes W[:, r] is (m*p) x (p*n)
                    # Flatten it
                    C_mat[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                    
                # T_mat is (n*m) x (m*p * p*n)
                T_mat = T_true.reshape(n*m, -1)
                # Solve U @ C_mat = T_mat
                # U = T_mat @ C_mat^+
                U_new, _, _, _ = np.linalg.lstsq(C_mat.T, T_mat.T, rcond=None)
                U = U_new.T
                
                # Update V
                # Similarly, T_mat2 is (m*p) x (n*m * p*n)
                # T_mat2[v, (u, w)] = T[u, v, w]
                # T_mat2[v, z] = sum_r V[v, r] * U[u, r] * W[w, r]
                # Let D_r be vector of size n*m * p*n, D_r[(u, w)] = U[u, r] * W[w, r]
                D_mat = np.zeros((R, n*m * p*n))
                for r in range(R):
                    D_mat[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                    
                T_mat2 = T_true.transpose(1, 0, 2).reshape(m*p, -1)
                V_new, _, _, _ = np.linalg.lstsq(D_mat.T, T_mat2.T, rcond=None)
                V = V_new.T
                
                # Update W
                # T_mat3 is (p*n) x (n*m * m*p)
                # T_mat3[w, (u, v)] = T[u, v, w]
                # T_mat3[w, z] = sum_r W[w, r] * U[u, r] * V[v, r]
                # Let E_r be vector of size n*m * m*p, E_r[(u, v)] = U[u, r] * V[v, r]
                E_mat = np.zeros((R, n*m * m*p))
                for r in range(R):
                    E_mat[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                    
                T_mat3 = T_true.transpose(2, 0, 1).reshape(p*n, -1)
                W_new, _, _, _ = np.linalg.lstsq(E_mat.T, T_mat3.T, rcond=None)
                W = W_new.T
                
            # After optimization, snap to integers
            U_int = snap_to_int(U)
            V_int = snap_to_int(V)
            W_int = snap_to_int(W)
            
            if check_decomposition(U_int, V_int, W_int):
                found = True
                # Convert to lists of lists for return
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
                
        if found:
            break
            
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

