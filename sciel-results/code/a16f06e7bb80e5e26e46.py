import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T
    # T[i, j, k] = 1 if this is a schoolbook term? No, T is the multiplication tensor.
    # T[i, j, k] is the coefficient of A[i, j] * B[j, k] in C[i, k].
    # In the decomposition T = sum U_c V_c W_c, the entries are:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The product sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should equal 1 if (i,j,k) is a valid term?
    # Wait. The tensor for matrix multiplication is defined such that
    # C[i, k] = sum_j A[i, j] B[j, k].
    # This corresponds to the tensor T where T[i, j, k] = 1 if the indices match the structure?
    # No, the tensor is T = sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i} (with appropriate W index).
    # Let's check the indices in the prompt: W[k*n+i, r].
    # So the tensor entry is T[i, j, k] = 1 for all i, j, k?
    # No, the multiplication tensor has non-zero entries at positions corresponding to the sum.
    # Actually, the tensor T is defined by T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # In the standard tensor definition for matmul, T[i, j, k] = 1 for all i, j, k in the ranges?
    # Let's verify.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # This is a trilinear form.
    # The tensor is T = sum_{i,j,k} A[i,j] B[j,k] C^*[i,k]?
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    # Entry T[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # For the multiplication tensor, T[i, j, k] = 1 if the indices are compatible?
    # Yes, T[i, j, k] = 1 for all valid i, j, k.
    # Because A[i,j] and B[j,k] always multiply to contribute to C[i,k].
    # So the target tensor is all 1s.
    
    # Let's double check.
    # If T[i,j,k] = 1, then sum_c ... = 1.
    # This means we are decomposing the tensor of all ones.
    # Is that correct?
    # The "matrix multiplication tensor" usually refers to the tensor that maps (A, B) to C.
    # The tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} (in the dual space).
    # So yes, the entries are 1.
    
    T_flat = np.ones(n * m * p)
    
    # Indices
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)
    
    def get_error(U, V, W):
        # U, V, W are numpy arrays
        # Compute sum_c U[:,c] * V[:,c] * W[:,c]
        # This is a tensor contraction.
        # T_calc[i*m+j, j*p+k, k*n+i] ? No.
        # We need to compute the scalar for each (i,j,k).
        # T_calc[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient computation:
        # Reshape U to (n, m, r)
        # Reshape V to (m, p, r)
        # Reshape W to (p, n, r)
        
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Sum over r
        # We want T[i, j, k]
        # This is element-wise product and sum.
        # Can use np.einsum
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        T_calc = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Flatten T_calc to match T_flat order
        # T_flat index l corresponds to (i, j, k)
        # How is T_flat ordered?
        # The prompt doesn't specify the order of T_flat, but we can define it.
        # Let's use row-major (C-order) for (i, j, k).
        # l = (i * m + j) * p + k
        T_calc_flat = T_calc.flatten()
        
        # Compute error
        err = np.linalg.norm(T_calc_flat - T_flat)
        return err

    def build_A_U(V, W, fixed_mask=None, fixed_vals=None):
        # Build matrix A_U such that A_U @ vec(U_free) = T_flat - A_U_fixed @ vec(U_fixed)
        # A_U has shape (n*m*p, n*m*r)
        # But we only build the part for free variables if mask is provided.
        
        r = V.shape[1]
        size_U = n * m
        size_T = n * m * p
        
        # Initialize A_U
        # We can build it column by column or block by block.
        # For each c, the block is (n*m*p, n*m)
        
        A_U = np.zeros((size_T, size_U * r))
        
        for c in range(r):
            V_c = V[:, c] # length m*p
            W_c = W[:, c] # length p*n
            
            # For each i, j, the column in A_U corresponding to U[i*m+j, c]
            # has non-zeros at T indices (i, j, k) for all k.
            # Value is V[j*p+k, c] * W[k*n+i, c]
            
            for i in range(n):
                for j in range(m):
                    col_idx = c * size_U + (i * m + j)
                    
                    # Vector of values for all k
                    vals = np.zeros(p)
                    for k in range(p):
                        v_val = V_c[j * p + k]
                        w_val = W_c[k * n + i]
                        vals[k] = v_val * w_val
                        
                    # Fill rows in A_U
                    # Row index for (i, j, k) is (i*m + j)*p + k
                    start_row = (i * m + j) * p
                    A_U[start_row:start_row+p, col_idx] = vals
        
        return A_U

    # Similar for V and W
    
    def build_A_V(U, W, fixed_mask=None, fixed_vals=None):
        r = U.shape[1]
        size_V = m * p
        size_T = n * m * p
        
        A_V = np.zeros((size_T, size_V * r))
        
        for c in range(r):
            U_c = U[:, c] # length n*m
            W_c = W[:, c] # length p*n
            
            for j in range(m):
                for k in range(p):
                    col_idx = c * size_V + (j * p + k)
                    
                    vals = np.zeros(n)
                    for i in range(n):
                        u_val = U_c[i * m + j]
                        w_val = W_c[k * n + i]
                        vals[i] = u_val * w_val
                        
                    start_row = (i * m + j) * p + k # Wait, row index depends on i, j, k
                    # Row index is (i*m + j)*p + k
                    # So for fixed j, k, and varying i, the rows are not contiguous?
                    # i=0: (0*m+j)*p+k = j*p+k
                    # i=1: (1*m+j)*p+k = m*p + j*p+k
                    # They are separated by m*p.
                    
                    for i in range(n):
                        row_idx = (i * m + j) * p + k
                        A_V[row_idx, col_idx] = vals[i]
                        
        return A_V

    def build_A_W(U, V, fixed_mask=None, fixed_vals=None):
        r = U.shape[1]
        size_W = p * n
        size_T = n * m * p
        
        A_W = np.zeros((size_T, size_W * r))
        
        for c in range(r):
            U_c = U[:, c] # length n*m
            V_c = V[:, c] # length m*p
            
            for k in range(p):
                for i in range(n):
                    col_idx = c * size_W + (k * n + i)
                    
                    vals = np.zeros(m)
                    for j in range(m):
                        u_val = U_c[i * m + j]
                        v_val = V_c[j * p + k]
                        vals[j] = u_val * v_val
                        
                    for j in range(m):
                        row_idx = (i * m + j) * p + k
                        A_W[row_idx, col_idx] = vals[j]
                        
        return A_W

    def solve_als(r, init_U=None, init_V=None, init_W=None, max_iter=50):
        if init_U is None:
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
        else:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
            
        for _ in range(max_iter):
            # Solve for U
            A_U = build_A_U(V, W)
            U_flat, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
            U = U_flat.reshape(n * m, r)
            
            # Solve for V
            A_V = build_A_V(U, W)
            V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            # Solve for W
            A_W = build_A_W(U, V)
            W_flat, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
            W = W_flat.reshape(p * n, r)
            
            # Check error
            err = get_error(U, V, W)
            if err < 1e-8:
                return U, V, W, True
                
        return U, V, W, False

    # Search for minimal rank
    best_rank = 60
    best_U, best_V, best_W = None, None, None
    
    # Initialize best with schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    deadline = time.time() + 110
    
    # Try ranks from 55 down to 50
    # If we find a solution, update best
    # We can also try 40-50 if time permits
    
    for r in range(55, 49, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random starts
        for _ in range(100):
            if time.time() > deadline:
                break
                
            U, V, W, success = solve_als(r)
            if success:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break # Found a solution for this rank, try lower rank
                
        if found:
            # We found a solution for rank r.
            # Since we are iterating downwards, this is the best so far.
            # We can continue to lower ranks if time permits.
            continue
            
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
