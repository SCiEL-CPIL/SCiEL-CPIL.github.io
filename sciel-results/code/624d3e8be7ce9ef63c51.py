import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = 2, 4, 4
    
    # Build multiplication tensor: T[a,b,c]=1 where a=i*m+k, b=k*p+j, c=i*p+j
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, i*p+j] = 1
    
    def verify(U, V, W, r):
        """Verify that the decomposition is correct using vectorized operations."""
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        # T[a,b,c] = sum_s U[a,s] V[b,s] W[c,s]
        # Compute efficiently: for each s, add outer product
        result = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for s in range(r):
            result += np.einsum('a,b,c->abc', U_arr[:, s], V_arr[:, s], W_arr[:, s])
        return np.array_equal(result, T)
    
    # Known rank-28 Strassen-based solution as fallback
    def strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1

            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1

            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1

            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1

            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1

            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1

            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1

        X1 = [0, 1, 4, 5];  Y1 = [0, 1, 4, 5];   Z1 = [0, 2, 1, 3]
        X2 = [2, 3, 6, 7];  Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
        X3 = [0, 1, 4, 5];  Y3 = [2, 3, 6, 7];   Z3 = [4, 6, 5, 7]
        X4 = [2, 3, 6, 7];  Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]

        add_block(0,  X1, Y1, Z1)
        add_block(7,  X2, Y2, Z2)
        add_block(14, X3, Y3, Z3)
        add_block(21, X4, Y4, Z4)
        
        return U, V, W, rank
    
    # Compute F2 rank lower bound: the tensor rank over any field >= F2 rank
    # For each nonzero coordinate (a,b,c) of T, the coordinate vector is e_a ⊗ e_b ⊗ e_c
    # The span of these over F2 gives a lower bound on tensor rank over Q
    
    def compute_f2_lower_bound():
        """Compute rank of the set of coordinate vectors over F2."""
        # Collect all nonzero coordinates
        coords = []
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if T[a, b, c] != 0:
                        coords.append((a, b, c))
        
        # Each coordinate vector is a vector in F2^(n*m * m*p * p*n) = F2^1024
        # But we can work with the flattened representation
        # The rank of {e_a ⊗ e_b ⊗ e_c} over F2 is at most the number of distinct
        # "lines" through the origin in the projective space
        
        # Actually, for a 3-tensor, the F2 rank of the coordinate set gives a lower bound
        # Let's compute: represent each (a,b,c) as a vector in F2^(nm + mp + pn) 
        # No, that's not right either.
        
        # Better: the tensor rank over F2 is >= the maximum over all "slices"
        # A slice fixing one index gives a matrix, and its rank over F2 is a lower bound
        
        # Slice along a: for each a, M_a[b,c] = T[a,b,c], rank over F2
        max_slice_rank = 0
        for a in range(n*m):
            M_a = T[a, :, :].astype(int) % 2
            if M_a.size > 0:
                # Compute rank over F2 using Gaussian elimination
                mat = M_a.copy()
                rows, cols = mat.shape
                rank = 0
                for col in range(cols):
                    # Find pivot
                    pivot = -1
                    for row in range(rank, rows):
                        if mat[row, col] == 1:
                            pivot = row
                            break
                    if pivot >= 0:
                        mat[[rank, pivot]] = mat[[pivot, rank]]
                        for row in range(rows):
                            if row != rank and mat[row, col] == 1:
                                mat[row] = (mat[row] + mat[rank]) % 2
                        rank += 1
                max_slice_rank = max(max_slice_rank, rank)
        
        # Also try slices along b and c
        for b in range(m*p):
            M_b = T[:, b, :].astype(int) % 2
            if M_b.size > 0:
                mat = M_b.copy()
                rows, cols = mat.shape
                rank = 0
                for col in range(cols):
                    pivot = -1
                    for row in range(rank, rows):
                        if mat[row, col] == 1:
                            pivot = row
                            break
                    if pivot >= 0:
                        mat[[rank, pivot]] = mat[[pivot, rank]]
                        for row in range(rows):
                            if row != rank and mat[row, col] == 1:
                                mat[row] = (mat[row] + mat[rank]) % 2
                        rank += 1
                max_slice_rank = max(max_slice_rank, rank)
        
        for c in range(p*n):
            M_c = T[:, :, c].astype(int) % 2
            if M_c.size > 0:
                mat = M_c.copy()
                rows, cols = mat.shape
                rank = 0
                for col in range(cols):
                    pivot = -1
                    for row in range(rank, rows):
                        if mat[row, col] == 1:
                            pivot = row
                            break
                    if pivot >= 0:
                        mat[[rank, pivot]] = mat[[pivot, rank]]
                        for row in range(rows):
                            if row != rank and mat[row, col] == 1:
                                mat[row] = (mat[row] + mat[rank]) % 2
                        rank += 1
                max_slice_rank = max(max_slice_rank, rank)
        
        return max_slice_rank
    
    f2_lb = compute_f2_lower_bound()
    
    # Also compute the "Kronecker product" structure lower bound
    # The tensor T = sum_{i,k,j} e_{im+k} ⊗ e_{kp+j} ⊗ e_{ip+j}
    # This is a sum of n*m*p = 32 rank-1 tensors (the standard algorithm)
    # The F2 rank of the set of these 32 vectors gives a lower bound
    
    def search_rank(r, max_trials=5000):
        """Search for rank-r decomposition with small integer/half-integer entries."""
        if time.time() > deadline - 5:
            return None
        
        # For each trial, randomly initialize U and V with sparse +/-1 entries
        # Then solve for W using least squares, check if exact
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(12345 + trial)
            
            # Initialize U and V with sparse structure
            U_mat = np.zeros((n*m, r), dtype=np.int64)
            V_mat = np.zeros((m*p, r), dtype=np.int64)
            
            # Each column of U and V should have a few non-zero entries
            for s in range(r):
                nz_u = np.random.randint(2, min(4, n*m))
                idx_u = np.random.choice(n*m, nz_u, replace=False)
                U_mat[idx_u, s] = np.random.choice([-1, 1], nz_u)
                
                nz_v = np.random.randint(2, min(4, m*p))
                idx_v = np.random.choice(m*p, nz_v, replace=False)
                V_mat[idx_v, s] = np.random.choice([-1, 1], nz_v)
            
            # Now for each c, M_c[a,b] = sum_s U_mat[a,s] V_mat[b,s] W[c,s]
            # This is linear in W. Let's set up the system.
            # For each (a,b,c): sum_s U_mat[a,s] V_mat[b,s] W[c,s] = T[a,b,c]
            # Group by c: for fixed c, we have an 8x16 system in W[c,:]
            # But W[c,s] appears in all (a,b) equations for that c
            # So for each c, we need: sum_s W[c,s] * (U_mat[:,s] V_mat[:,s]^T) = M_c
            
            # This is a linear system: A_c @ W[c,:] = vec(M_c)
            # where A_c is 128 x r (for each (a,b) pair)
            
            W_mat = np.zeros((p*n, r), dtype=np.int64)
            valid = True
            
            for c in range(p*n):
                # Build the 128 x r matrix A_c
                # A_c[(a,b), s] = U_mat[a,s] * V_mat[b,s]
                A_c = np.zeros((n*m * m*p, r), dtype=np.int64)
                rhs = np.zeros(n*m * m*p, dtype=np.int64)
                
                for a in range(n*m):
                    for b in range(m*p):
                        idx = a * m*p + b
                        for s in range(r):
                            A_c[idx, s] = U_mat[a, s] * V_mat[b, s]
                        rhs[idx] = T[a, b, c]
                
                # Solve A_c @ W_c = rhs using least squares
                # Since entries are small integers, try to find exact solution
                W_c, residuals, rank_c, sv = np.linalg.lstsq(A_c.astype(float), rhs.astype(float), rcond=None)
                
                # Check if solution is close to integer/half-integer
                W_c_rounded = np.round(W_c * 2) / 2
                if not np.allclose(A_c @ W_c_rounded, rhs, atol=1e-9):
                    valid = False
                    break
                
                W_mat[c, :] = W_c_rounded
            
            if not valid:
                continue
            
            # Verify the full decomposition
            U_list = [list(U_mat[a, :]) for a in range(n*m)]
            V_list = [list(V_mat[b, :]) for b in range(m*p)]
            W_list = [list(W_mat[c, :]) for c in range(p*n)]
            
            if verify(U_list, V_list, W_list, r):
                return U_list, V_list, W_list, r
        
        return None
    
    # Try ranks from the F2 lower bound to 28
    best_result = None
    start_rank = max(16, f2_lb)
    for r in range(start_rank, 29):
        if time.time() > deadline - 10:
            break
        result = search_rank(r, max_trials=3000)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-28 Strassen solution
    U, V, W, rank = strassen_28()
    return {"U": U, "V": V, "W": W}
