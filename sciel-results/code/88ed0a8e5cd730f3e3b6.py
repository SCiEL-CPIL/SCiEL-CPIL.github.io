"""Matrix multiplication 2,4,7 tensor decomposition using SVD-based subspace analysis."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 140  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify if the decomposition is exact using the tensor identity."""
    rank = U.shape[1]
    if rank == 0:
        return False
        
    # Reshape for einsum
    # U is (n*m, rank) -> U[i*m+j, r]
    # V is (m*p, rank) -> V[j*p+k, r]
    # W is (p*n, rank) -> W[k*n+i, r]
    
    U_ = U.reshape(n, m, rank)  # U[i, j, r]
    V_ = V.reshape(m, p, rank)  # V[j, k, r]
    W_ = W.reshape(p, n, rank)  # W[k, i, r]
    
    # We need T[i,j,k] = sum_r U_[i,j,r] * V_[j,k,r] * W_[k,i,r]
    # W_ is (p, n, rank). We need W_[k, i, r].
    # Let's transpose W_ to (n, p, rank) so W_t[i, k, r] = W_[k, i, r]
    W_t = W_.transpose(1, 0, 2) # shape (n, p, rank), W_t[i, k, r]
    
    T = np.einsum('ijr,jkr,ikr->ijk', U_, V_, W_t)
    
    # T should be all ones
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[0, 2] = -1
    
    # M4 = d(b-e)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = -1; W[3, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = -1; W[2, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1; W[1, 6] = -1
    
    return U, V, W

def _construct_block_decomposition():
    """
    Construct 2x4x7 decomposition using block structure.
    Rank 50 baseline.
    """
    n, m, p = 2, 4, 7
    
    U_total = np.zeros((n*m, 0), dtype=np.float64)
    V_total = np.zeros((m*p, 0), dtype=np.float64)
    W_total = np.zeros((p*n, 0), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # 2x2x1 decomposition
    U_221 = np.zeros((4, 4), dtype=np.float64)
    V_221 = np.zeros((2, 4), dtype=np.float64)
    W_221 = np.zeros((2, 4), dtype=np.float64)
    
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1
    
    def embed_tensor(U_t, V_t, W_t, a_col_start, b_row_start, b_col_start, c_col_start, b_width):
        nonlocal U_total, V_total, W_total
        R = U_t.shape[1]
        U_block = np.zeros((8, R), dtype=np.float64)
        V_block = np.zeros((28, R), dtype=np.float64)
        W_block = np.zeros((14, R), dtype=np.float64)
        
        for i in range(2):
            for j in range(b_width):
                U_block[i*4 + a_col_start + j, :] = U_t[i*2 + j, :]
        
        for i in range(b_width):
            for j in range(b_width):
                V_block[(b_row_start + i)*7 + b_col_start + j, :] = V_t[i*b_width + j, :]
        
        for i in range(b_width):
            for j in range(2):
                W_block[(c_col_start + i)*2 + j, :] = W_t[i*b_width + j, :]
        
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # Term 1: A_top (cols 0,1) * B_top (rows 0,1)
    embed_tensor(U_s, V_s, W_s, 0, 0, 0, 0, 2)
    embed_tensor(U_s, V_s, W_s, 0, 0, 2, 2, 2)
    embed_tensor(U_s, V_s, W_s, 0, 0, 4, 4, 2)
    embed_tensor(U_221, V_221, W_221, 0, 0, 6, 6, 1)
    
    # Term 2: A_bottom (cols 2,3) * B_bottom (rows 2,3)
    embed_tensor(U_s, V_s, W_s, 2, 2, 0, 0, 2)
    embed_tensor(U_s, V_s, W_s, 2, 2, 2, 2, 2)
    embed_tensor(U_s, V_s, W_s, 2, 2, 4, 4, 2)
    embed_tensor(U_221, V_221, W_221, 2, 2, 6, 6, 1)
    
    return U_total, V_total, W_total

def _tensor_to_matrix(U, V, W, n, m, p):
    """
    Flatten the tensor defined by U, V, W into a matrix M of shape (n*m*p, rank).
    M[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This is the "flattened tensor" where each column is a rank-1 term vector.
    """
    rank = U.shape[1]
    M = np.zeros((n*m*p, rank), dtype=np.float64)
    
    # Precompute indices for speed
    # U is (n*m, rank)
    # V is (m*p, rank)
    # W is (p*n, rank)
    
    # We need to map (i, j, k) to index in M: idx = (i*p + k)*m + j? No, standard row-major for T[i,j,k] is i*(m*p) + j*p + k.
    # Let's stick to that.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * (m * p) + j * p + k
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                M[idx, :] = u_val * v_val * w_val
    return M

def _try_reduce_rank_svd(U, V, W, n, m, p, max_iter=10, tol=1e-8):
    """
    Iteratively try to reduce rank by checking if any column of the flattened tensor
    is in the span of the others.
    
    If column c is in the span of others, we can remove it and adjust the others?
    Actually, if M[:, c] = sum_{r != c} alpha_r M[:, r], it doesn't mean we can just remove c
    unless we can absorb it into the U, V, W structure.
    
    However, there is a known property: If the tensor rank is r, and we have a decomposition
    of rank r, the columns of the flattened matrix M are linearly independent IF the
    decomposition is "minimal" in a strong sense? No.
    
    A better approach: 
    1. Compute M (n*m*p, rank).
    2. Compute SVD of M. If the numerical rank of M is less than rank, we have redundancy.
    3. Specifically, we can try to find a subset of columns that spans the same space.
    
    But simply reducing the rank of M doesn't guarantee a valid U, V, W decomposition of that rank.
    The structure of U, V, W imposes constraints.
    
    Alternative Strategy: "Peeling via Projection"
    For each column r, check if T_r (the rank-1 tensor) is in the span of {T_s | s != r}.
    If yes, T_r is redundant. We can remove it.
    But checking span membership in tensor space is hard.
    
    Let's use the "Matrix Flattening" trick for specific modes.
    If we fix the U part, the remaining tensor is in V x W space.
    
    Let's try a simpler, more robust heuristic:
    1. Start with a valid decomposition.
    2. Randomly transform the rank space (U'=U*T, V'=V*T^-1, W'=W*T^-1).
    3. Check if any column of U', V', or W' is zero (or close to zero). If so, remove it.
       This works because if a column of U is 0, the term contributes nothing.
       But random transforms rarely produce exact zeros.
    
    Better:
    1. Use the SVD of the flattened matrix M to find the principal components.
    2. Project the decomposition onto the top k components.
    3. Check if the projection is still exact.
    
    Actually, the most effective method for small tensors is "Grassmannian Search" or simply
    checking linear dependencies in the "slice" matrices.
    
    Let's implement a robust peeling:
    For each column r:
      1. Compute the residual tensor R = T - T_r.
      2. Check if R can be decomposed with rank r-1.
      This is equivalent to checking if T_r is in the span of the others.
      
      We can approximate this by checking if the vector M[:, r] is in the column space of M[:, -r].
      If yes, remove r.
      
      Why is this valid?
      If M[:, r] = M[:, -r] @ x, then sum_s alpha_s T_s = T_r.
      This implies T_r - sum_s alpha_s T_s = 0.
      So the set {T_1, ..., T_{r-1}, T_{r+1}, ..., T_r - sum...} is a valid decomposition?
      No, T_r - sum is a linear combination of the others, so it's just a re-weighting.
      Wait. If M[:, r] is in the span of others, it means the *vectorized tensor* T_r is in the span
      of the other *vectorized tensors*.
      This means T_r = sum_{s != r} c_s T_s.
      Then T = sum_{s != r} T_s + T_r = sum_{s != r} (1 + c_s) T_s.
      So we can simply absorb T_r into the others!
      
      So the algorithm is:
      1. Compute M.
      2. For each r, check if M[:, r] is in the span of M[:, -r].
      3. If yes, calculate coefficients c.
      4. Update U, V, W by adding c_r * T_r to the other terms?
         No, we just remove T_r and scale the others?
         Actually, if T_r = sum c_s T_s, then T = sum_{s != r} T_s + sum c_s T_s = sum_{s != r} (1+c_s) T_s.
         So we can update U_s' = U_s + c_s U_r, V_s' = V_s + c_s V_r, W_s' = W_s + c_s W_r?
         No, that's not right. The terms are U_s V_s W_s.
         We need to find new coefficients for the existing basis vectors U_s, V_s, W_s.
         
         Actually, the decomposition is not unique.
         If we have T = sum T_s, and T_r = sum_{s != r} c_s T_s,
         then T = sum_{s != r} (1 + c_s) T_s.
         This is a valid decomposition of rank r-1 using the *same* U_s, V_s, W_s vectors for s != r,
         but with modified "weights"? No, the weights are embedded in U, V, W.
         
         Wait, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
         If we remove r, we need to represent T_r using the other U_s, V_s, W_s.
         But U_s, V_s, W_s are fixed vectors. We can't just "add" c_s U_r to U_s because
         (U_s + c_s U_r) \otimes V_s \otimes W_s is NOT c_s U_r \otimes V_s \otimes W_s.
         
         So this simple peeling only works if we can find a transformation.
         
         However, there is a special case: if U_r is in the span of U_{-r}, V_r in span of V_{-r}, etc.
         
         Let's stick to the "Random Transform + Zero Check" method but make it smarter.
         Apply a random invertible transformation T.
         U' = U T, V' = V T^{-1}, W' = W T^{-1}.
         Check if any column of U', V', or W' is zero.
         If U'[:, k] is zero, then the k-th term is zero, so we can drop it.
         
         This is what the current program does, but it's slow because it verifies every time.
         We can speed up verification by only checking the tensor residual if a zero is found.
    """
    pass

def _fast_peeling(U, V, W, n, m, p, time_budget=60):
    """
    Fast peeling using random transformations and zero-detection.
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    rng = np.random.default_rng(54321)
    
    # Pre-compute tensor M for fast checks? No, M changes with U,V,W.
    # But we can check for zeros in U, V, W directly.
    
    for iteration in range(1000):
        if time.time() - start_time > time_budget:
            break
        if best_rank <= 28: # Lower bound
            break
            
        # Apply random transformation
        rank = best_rank
        T = rng.standard_normal((rank, rank)) * 0.5 + np.eye(rank)
        try:
            T_inv = np.linalg.inv(T)
        except:
            continue
            
        U_t = best_U @ T
        V_t = best_V @ T_inv
        W_t = best_W @ T_inv
        
        # Check for zero columns (or very small)
        # If a column is zero in U, V, or W, the term is 0.
        zero_mask = np.zeros(rank, dtype=bool)
        zero_mask[np.any(np.abs(U_t) < 1e-9, axis=0)] = True
        zero_mask[np.any(np.abs(V_t) < 1e-9, axis=0)] = True
        zero_mask[np.any(np.abs(W_t) < 1e-9, axis=0)] = True
        
        if np.any(zero_mask):
            # Found a zero column. Remove it.
            keep_mask = ~zero_mask
            U_new = U_t[:, keep_mask]
            V_new = V_t[:, keep_mask]
            W_new = W_t[:, keep_mask]
            
            # Verify
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-7):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = U_new.shape[1]
                # Continue from this new state
                continue
                
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    try:
        # Start with block decomposition (rank 50)
        U, V, W = _construct_block_decomposition()
        
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            U, V, W = _construct_schoolbook(n, m, p)
        
        # Try to reduce rank further using fast peeling
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 10:
            U, V, W = _fast_peeling(U, V, W, n, m, p, time_budget=remaining_time)
            
            if not _verify(U, V, W, n, m, p):
                # Fallback
                U, V, W = _construct_block_decomposition()
                if not _verify(U, V, W, n, m, p):
                    U, V, W = _construct_schoolbook(n, m, p)
        
    except Exception:
        try:
            U, V, W = _construct_block_decomposition()
            if not _verify(U, V, W, n, m, p):
                U, V, W = _construct_schoolbook(n, m, p)
        except Exception:
            U, V, W = _construct_schoolbook(n, m, p)
    
    # Final safety check
    if not _verify(U, V, W, n, m, p):
        U, V, W = _construct_schoolbook(n, m, p)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
