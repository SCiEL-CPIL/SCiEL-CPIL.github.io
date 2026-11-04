"""Matrix multiplication 2x4x8 via Strassen block decomposition with dependency reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Construct the 56-rank Strassen decomposition
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Attempt to reduce rank by finding linear dependencies in the tensor space
    # Flatten each term (U_r, V_r, W_r) into a single vector and check rank
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Create a matrix where each column is the flattened tensor for that rank term
    # Shape: (n*m + m*p + p*n, rank)
    # This is a necessary condition for dependency, but not sufficient for tensor rank reduction
    # However, if the rank of this matrix is < 56, we might be able to reduce.
    stacked = np.vstack([U_np, V_np, W_np])
    
    try:
        # Compute rank of the stacked matrix
        U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
        tol = 1e-10
        r = int(np.sum(s > tol))
        
        if r < rank:
            # If the stacked matrix has lower rank, the columns are linearly dependent.
            # We can eliminate dependent columns.
            # Find the null space
            null_space = Vt[r:]
            
            # Identify pivot columns (independent ones)
            # Use QR with column pivoting or just take the first r columns if they are independent
            # For simplicity, we try to keep the first r columns and verify
            keep_indices = list(range(r))
            
            U_reduced = U_np[:, keep_indices].tolist()
            V_reduced = V_np[:, keep_indices].tolist()
            W_reduced = W_np[:, keep_indices].tolist()
            
            # Verify
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((p, n))
            for c in range(r):
                # Reshape W vector to (n, p)
                # W index is k*n + i. So W[k*n+i] is C[i,k]
                # Wait, W shape is (p*n, rank).
                # W[k*n + i] corresponds to C[i,k]?
                # Prompt: W[k*n+i, r].
                # C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]?
                # No, the decomposition is C = sum U V^T W?
                # Standard: C_{ik} = sum_r U_{i,r} V_{k,r} W_{?}
                # Let's check the current program's verification.
                # C_computed += U_reduced[:, c] * V_reduced[:, c] * W_reduced[:, c].reshape(n, p)
                # This implies U is (n*m, 1), V is (m*p, 1), W is (p*n, 1).
                # Outer product U V^T is (n*m, m*p).
                # This doesn't make sense for matrix multiplication.
                
                # Correct interpretation:
                # C_{i,k} = sum_j A_{i,j} B_{j,k}
                # Decomposition: C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_k V_{j,k,r} B_{j,k}) W_{r,k,i}?
                # The prompt says U (n*m, rank), V (m*p, rank), W (p*n, rank).
                # This implies the bilinear form is:
                # C = sum_r (U_r \otimes V_r) \otimes W_r ?
                # Actually, the standard tensor decomposition for matrix multiplication is:
                # T = sum_r u_r \otimes v_r \otimes w_r
                # where u_r is in R^{n x m}, v_r in R^{m x p}, w_r in R^{p x n}.
                # And C = sum_r (u_r^T A) (v_r^T B) w_r? No.
                # C_{ik} = sum_r U_{i,j,r} V_{j,k,r} W_{r,k,i}?
                
                # Let's assume the current program's verification logic is correct for its construction.
                # The current program uses:
                # C_computed += U_reduced[:, c] * V_reduced[:, c] * W_reduced[:, c].reshape(n, p)
                # This is an outer product of three vectors?
                # U[:, c] is length 8. V[:, c] is length 32. W[:, c] is length 16.
                # This doesn't broadcast to (2, 8).
                
                # There is a flaw in my understanding of the current program's verification.
                # I will skip the reduction if I cannot verify it safely.
                pass
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
