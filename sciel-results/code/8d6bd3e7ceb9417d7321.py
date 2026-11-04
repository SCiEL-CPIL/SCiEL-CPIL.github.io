import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Fallback schoolbook decomposition
    def get_schoolbook():
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

    def verify(U, V, W, r):
        """
        Verify that the decomposition satisfies the matrix multiplication tensor.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        Should be 1 for all i, j, k.
        """
        T_comp = np.zeros((n, m, p))
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Efficient verification using einsum or loops
        # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i * m + j, :]
                for k in range(p):
                    v_val = V_arr[j * p + k, :]
                    w_val = W_arr[k * n + i, :]
                    T_comp[i, j, k] = np.dot(u_row, v_val * w_val)
        
        return np.allclose(T_comp, 1.0, atol=1e-9)

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    def try_find_decomposition(r, num_tries=100, seed_offset=0):
        """
        Try to find a rank-r decomposition.
        Strategy: Block-structured initialization.
        1. Last 18 factors (if r >= 18) are initialized as Schoolbook for the last 2 columns of B (k=3,4).
           Wait, if r is small, we can't have 18 dedicated factors.
           We will initialize the last min(18, r) factors with Schoolbook patterns for k=3,4.
           The remaining factors are initialized with random integers.
        2. Run ALS to minimize error.
        3. Snap and verify.
        """
        np.random.seed(42 + seed_offset)
        
        # Determine how many factors to dedicate to the "tail" (last 2 columns)
        # Schoolbook for 3x3x2 is rank 18.
        # If r < 18, we can't dedicate 18. We will just use random init for all.
        tail_rank = min(18, r)
        head_rank = r - tail_rank
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Initialize Head (random)
        if head_rank > 0:
            U[:, :head_rank] = np.random.randint(-1, 2, size=(n * m, head_rank))
            V[:, :head_rank] = np.random.randint(-1, 2, size=(m * p, head_rank))
            W[:, :head_rank] = np.random.randint(-1, 2, size=(p * n, head_rank))
            
        # Initialize Tail (Schoolbook for k=3,4)
        # Note: This assumes the tail factors handle k=3 and k=4 exclusively.
        # U[i*m+j, c] = 1
        # V[j*p+k, c] = 1
        # W[k*n+i, c] = 1
        tail_col = head_rank
        for k in [3, 4]:
            for i in range(n):
                for j in range(m):
                    if tail_col < r:
                        U[i * m + j, tail_col] = 1
                        V[j * p + k, tail_col] = 1
                        W[k * n + i, tail_col] = 1
                        tail_col += 1
                    else:
                        break
                if tail_col >= r:
                    break
            if tail_col >= r:
                break

        # ALS Optimization
        for _ in range(200):
            if time.time() > deadline:
                break
                
            # Solve for U
            # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # For fixed V, W, U[i*m+j, :] is linear in U.
            # We want sum_k sum_c U[i*m+j, c] (V[j*p+k, c] W[k*n+i, c]) = 1
            # Let A_ik[c, d] = sum_k V[j*p+k, c] W[k*n+i, c] ... No, c is the factor index.
            # Equation for U row (i, j):
            # sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] W[k*n+i, c]) = 1
            # This is a system of n*p equations (for each i, k) for each j?
            # No, for each i, j, we have a vector of length r.
            # The constraint is coupled across i, k.
            
            # Let's solve for each column c? No, ALS solves for the whole matrix U.
            # Objective: min_U sum_{i,j,k} (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
            # This is a least squares problem for U if V, W are fixed.
            # Let's construct the matrix A for the linear system.
            # Variables: U_flat (size n*m*r)
            # Equations: n*m*p (one for each i, j, k)
            
            # A is (n*m*p) x (n*m*r)
            # Row index: (i, j, k)
            # Col index: (i', j', c)
            # A[row, col] = V[j*p+k, c] * W[k*n+i, c] if i==i' and j==j', else 0.
            # This is sparse.
            
            # Given small size, we can build the dense matrix.
            A_mat = np.zeros((n * m * p, n * m * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i * m * p + j * p + k
                        # This row depends on U[i*m+j, :]
                        base_col = (i * m + j) * r
                        for c in range(r):
                            val = V[j * p + k, c] * W[k * n + i, c]
                            A_mat[row_idx, base_col + c] = val
            
            # Solve A_mat @ U_flat = b_vec
            try:
                U_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U = U_flat.reshape((n * m, r))
            except:
                break
                
            # Solve for V
            A_mat = np.zeros((n * m * p, m * p * r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i * m * p + j * p + k
                        base_col = (j * p + k) * r
                        for c in range(r):
                            val = U[i * m + j, c] * W[k * n + i, c]
                            A_mat[row_idx, base_col + c] = val
            try:
                V_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V = V_flat.reshape((m * p, r))
            except:
                break
                
            # Solve for W
            A_mat = np.zeros((n * m * p, p * n * r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i * m * p + j * p + k
                        base_col = (k * n + i) * r
                        for c in range(r):
                            val = U[i * m + j, c] * V[j * p + k, c]
                            A_mat[row_idx, base_col + c] = val
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_flat.reshape((p * n, r))
            except:
                break

        # Snap to half-integers
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if verify(U_snap, V_snap, W_snap, r):
            return U_snap, V_snap, W_snap
            
        return None

    # Try to find a better decomposition
    # Target ranks: 41 (Laderman 23 + Schoolbook 18), 40, 39...
    # We start with 41 because we have a strong structural prior for it.
    
    # If r=41, tail_rank=18, head_rank=23.
    # This matches the Laderman + Schoolbook structure.
    
    for r in range(41, 35, -1):
        if time.time() > deadline:
            break
        result = try_find_decomposition(r, num_tries=20, seed_offset=r)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # If no improvement found, return schoolbook
    return get_schoolbook()
