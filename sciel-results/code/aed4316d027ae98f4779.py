import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    
    # Target Rank 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    if col != 54:
        pass
        
    # Try to reduce rank by finding linearly dependent columns
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check if any column can be removed (is in span of others)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each pair of columns, check if they are linearly dependent
    # by checking if the corresponding rank-1 tensors are proportional
    # Actually, check if column r is in the span of others by solving a linear system
    
    # The tensor entries are T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # If we remove column r, we need to see if T can still be represented with rank-1 terms
    # from the remaining columns. This is equivalent to checking if the r-th rank-1 tensor
    # is in the span of the others.
    
    # Build matrix M where M[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Then check if any column of M is in the span of the others.
    
    # But this is a 60x54 matrix (2*5*6=60 entries, 54 rank terms)
    # Wait, the tensor has 2*5*6 = 60 entries.
    
    T_size = n * m * p  # 60
    M = np.zeros((T_size, col))
    for r in range(col):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    # Check rank of M
    rank_M = np.linalg.matrix_rank(M)
    
    # If rank_M < col, we can reduce
    if rank_M < col:
        # Find which columns are dependent
        # Use QR with column pivoting
        Q, R, piv = np.linalg.qr(M.T, mode='complete', pivoting=True)
        # piv tells us the column permutation
        # The first rank_M columns in the permuted order are independent
        # We can reconstruct using only those columns
        
        # Actually, let's just find a minimal subset
        # Use greedy: start with all, remove one at a time if rank doesn't change
        keep = list(range(col))
        changed = True
        while changed:
            changed = False
            for idx in list(keep):
                test_keep = [k for k in keep if k != idx]
                if len(test_keep) == 0:
                    break
                M_test = M[:, test_keep]
                if np.linalg.matrix_rank(M_test) == rank_M:
                    keep = test_keep
                    changed = True
                    break
        
        if len(keep) < col:
            new_rank = len(keep)
            U_new = U_arr[:, keep]
            V_new = V_arr[:, keep]
            W_new = W_arr[:, keep]
            
            # Convert back to lists
            U_out = U_new.tolist()
            V_out = V_new.tolist()
            W_out = W_new.tolist()
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        val = sum(U_out[i*m+j][r] * V_out[j*p+k][r] * W_out[k*n+i][r] for r in range(new_rank))
                        expected = i * 0 + j * 0 + k * 0  # This should be the actual tensor value
                        # The multiplication tensor has T[i,j,k] = 1 if j is the "middle" index
                        # Actually T[i,j,k] = delta (it's 1 for all valid (i,j,k) since it's the identity in the middle)
                        # Wait no. The matrix multiplication tensor T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
                        # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
                        # The tensor is defined such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
                        # So T[i,j,k] = 1 for all i,j,k. Yes!
                        if abs(val - 1.0) > 1e-9:
                            # Verification failed, keep original
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                # All verified
                return {"U": U_out, "V": V_out, "W": W_out}
    
    return {"U": U, "V": V, "W": W}
