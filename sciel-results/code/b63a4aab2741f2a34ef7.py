import time
import numpy as np
from itertools import product, combinations_with_replacement

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Schoolbook fallback (rank 28)
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)

    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)

    def to_half_int_list(mat):
        doubled = np.round(mat * 2)
        return doubled.astype(int).tolist()

    # Build the linear system for a given set of column patterns
    # Each column c has:
    #   U: one non-zero at (i_c, k_c)
    #   V: one non-zero at (k_c, j_c)
    #   W: one non-zero at (j_c, i_c)
    # The tensor equation is:
    #   sum_c u_c * v_c * w_c * e_{i_c,k_c} ⊗ e_{k_c,j_c} ⊗ e_{j_c,i_c} = T
    # where T[i,k,j] = 1 if (i,k,j) is a valid multiplication index, else 0.
    # Actually, T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # because the multiplication tensor has 1s at all positions where the
    # corresponding output element is computed. Wait, the multiplication tensor
    # T[i,k,j] = 1 if the (i,k) element of A and (k,j) element of B contribute
    # to the (i,j) element of C. Since C[i,j] = sum_k A[i,k] * B[k,j], the
    # tensor has T[i,k,j] = 1 for all i,k,j.
    
    # So we need: sum_c u_c * v_c * w_c * delta(i,i_c) * delta(k,k_c) * delta(j,j_c) = 1
    # for all (i,k,j) in [0,n) x [0,m) x [0,p).
    
    # This is a linear system in the variables x_c = u_c * v_c * w_c.
    # For each (i,k,j), the equation is:
    #   sum_{c: i_c=i, k_c=k, j_c=j} x_c = 1
    # This means that for each (i,k,j), the number of columns that exactly
    # match (i,k,j) must be such that the sum of their x_c is 1.
    # But if multiple columns match the same (i,k,j), we can distribute.
    # However, if we restrict to binary patterns where each column has a unique
    # (i_c, k_c, j_c) triple, then each (i,k,j) can be matched by at most one column,
    # and we need exactly one column for each (i,k,j) with x_c = 1.
    # This gives rank = n*m*p = 28, which is schoolbook.
    
    # To get lower rank, we need columns that do NOT have a unique (i,k,j) triple.
    # Wait, the pattern is: U has one 1 at (i_c, k_c), V has one 1 at (k_c, j_c),
    # W has one 1 at (j_c, i_c). The triple is (i_c, k_c, j_c).
    # The tensor equation at (i,k,j) is:
    #   sum_{c: i_c=i, k_c=k, j_c=j} x_c = 1
    # So if no column has (i_c,k_c,j_c) = (i,k,j), the equation is 0=1, which fails.
    # Therefore, for every (i,k,j), there must be at least one column with that triple.
    # This implies rank >= n*m*p = 28. This is a contradiction with known lower ranks.
    
    # I made a mistake. The tensor decomposition is:
    # T[i,k,j] = sum_c U[i,k,c] * V[k,j,c] * W[j,i,c]
    # If U[i,k,c] is non-zero only when (i,k) = (i_c, k_c), etc., then
    # U[i,k,c] = u_c * delta(i,i_c) * delta(k,k_c)
    # V[k,j,c] = v_c * delta(k,k_c) * delta(j,j_c)
    # W[j,i,c] = w_c * delta(j,j_c) * delta(i,i_c)
    # Then T[i,k,j] = sum_c u_c v_c w_c * delta(i,i_c) delta(k,k_c) delta(j,j_c)
    #                 * delta(k,k_c) delta(j,j_c) delta(i,i_c)
    #                 * delta(j,j_c) delta(i,i_c)
    # = sum_c u_c v_c w_c * delta(i,i_c) delta(k,k_c) delta(j,j_c)
    # So indeed, for T[i,k,j] to be 1, we need sum_{c: (i_c,k_c,j_c)=(i,k,j)} u_c v_c w_c = 1.
    # This requires that for every (i,k,j), there is at least one column with that triple.
    # So rank >= 28. This means the sparse binary pattern approach with one non-zero per factor
    # per column cannot achieve rank < 28.
    
    # Therefore, the columns must have MORE than one non-zero in each factor.
    # Let's relax the pattern: each column c has a support in U, V, W.
    # U[:,c] is a vector in R^{n*m}, V[:,c] in R^{m*p}, W[:,c] in R^{p*n}.
    # The tensor equation is:
    # T[i,k,j] = sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c]
    # This is a bilinear (actually trilinear) system.
    # To solve for half-integer entries, we can fix the supports and solve for the values.
    # But the supports are also variables.
    
    # Alternative: Use the known fact that for 2x2x2, rank 7 is achievable.
    # For 2x2x7, we can use the block structure: 2x2x7 = 2x2x(3*2 + 1).
    # Use rank 7 for each 2x2x2 block and rank 2 for the last 2x2x1 block.
    # Total rank = 3*7 + 2 = 23. This is what the current program does.
    
    # To improve, we need a better decomposition for 2x2x7.
    # Known lower bound for 2x2x7 is at least 20 (from asymptotic bounds).
    # Let's try to find a rank 20, 21, or 22 decomposition.
    
    # Strategy: Use a structured search where U, V, W have a block-diagonal or
    # block-circulant structure. For 2x2x7, we can view it as 2x2x(2+2+2+1).
    # Let's try to find a decomposition where the first 6 columns of the 7-column
    # dimension are handled by a 2x2x6 decomposition, and the last by 2x2x1.
    # 2x2x6 = 2x2x(3*2). If we can find a rank 18 decomposition for 2x2x6,
    # then total rank = 18 + 2 = 20.
    
    # How to find a rank 18 decomposition for 2x2x6?
    # 2x2x6 can be seen as 3 copies of 2x2x2. If we use rank 7 for each, total is 21.
    # To get 18, we need to merge columns across the 3 blocks.
    
    # Let's try a direct search for rank 20, 21, 22 with a smarter initialization.
    # Instead of random, use a structured initialization based on the Laderman decomposition
    # for 2x2x2, and try to merge columns.
    
    def laderman_block():
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        U_b[0,0]=1; U_b[1,0]=-1; U_b[2,0]=-1; U_b[3,0]=1
        V_b[0,0]=1; V_b[1,0]=-1; V_b[2,0]=-1; V_b[3,0]=1
        W_b[0,0]=1; W_b[1,0]=1; W_b[2,0]=1; W_b[3,0]=1
        U_b[0,1]=1; U_b[1,1]=-1
        V_b[1,1]=1; V_b[3,1]=-1
        W_b[1,1]=1; W_b[3,1]=1
        U_b[0,2]=1; U_b[2,2]=-1
        V_b[2,2]=1; V_b[0,2]=-1
        W_b[2,2]=1; W_b[3,2]=1
        U_b[0,3]=1; V_b[0,3]=1; W_b[0,3]=1
        U_b[1,4]=1; V_b[1,4]=1; W_b[1,4]=1
        U_b[2,5]=1; V_b[2,5]=1; W_b[2,5]=1
        U_b[3,6]=1; V_b[3,6]=1; W_b[3,6]=1
        return U_b, V_b, W_b

    def schoolbook_2x2x1():
        U_b = np.zeros((4, 2))
        V_b = np.zeros((2, 2))
        W_b = np.zeros((2, 2))
        U_b[0,0]=1; V_b[0,0]=1; W_b[0,0]=1
        U_b[1,1]=1; V_b[1,1]=1; W_b[1,1]=1
        return U_b, V_b, W_b

    def build_laderman_decomposition():
        rank = 23
        U_full = np.zeros((n*m, rank))
        V_full = np.zeros((m*p, rank))
        W_full = np.zeros((p*n, rank))
        U_l, V_l, W_l = laderman_block()
        U_s, V_s, W_s = schoolbook_2x2x1()
        for block_idx in range(3):
            col_offset = block_idx * 7
            j_start = block_idx * 2
            for c_local in range(7):
                c_global = col_offset + c_local
                for idx in range(4):
                    U_full[idx, c_global] = U_l[idx, c_local]
                for k in range(2):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        V_full[k*p + j_global, c_global] = V_l[k*2 + j_local, c_local]
                for j_local in range(2):
                    for i in range(2):
                        j_global = j_start + j_local
                        W_full[j_global*n + i, c_global] = W_l[j_local*2 + i, c_local]
        col_offset = 21
        for c_local in range(2):
            c_global = col_offset + c_local
            for idx in range(4):
                U_full[idx, c_global] = U_s[idx, c_local]
            for k in range(2):
                V_full[k*p + 6, c_global] = V_s[k, c_local]
            for i in range(2):
                W_full[6*n + i, c_global] = W_s[i, c_local]
        return U_full, V_full, W_full

    U0, V0, W0 = build_laderman_decomposition()
    best_rank = 23
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    if not verify(U0, V0, W0):
        return fallback

    # New strategy: Use a structured search for rank 20, 21, 22.
    # Initialize with the rank-23 decomposition and try to merge columns.
    # For each target rank, try to find a linear combination of the 23 columns
    # that reduces the rank.
    
    def compute_error(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.max(np.abs(T - 1.0))

    rng = np.random.default_rng(12345)
    
    # Try to find a rank-20, 21, 22 decomposition by merging columns.
    # Start from the rank-23 decomposition and try to eliminate columns.
    
    for target_rank in [20, 21, 22]:
        if time.time() > deadline - 10:
            break
        
        # Try to find a subset of target_rank columns from the 23 that still span
        # the tensor. This is a combinatorial problem.
        # Instead, use a local search: start from the rank-23 decomposition,
        # and try to perturb the entries to reduce the rank.
        
        best_err = float('inf')
        best_U_t = None
        best_V_t = None
        best_W_t = None
        
        n_tries = 100
        for attempt in range(n_tries):
            if time.time() > deadline - 5:
                break
            
            # Initialize with the rank-23 decomposition and set (23-target_rank) columns to zero.
            U_t = U0.copy()
            V_t = V0.copy()
            W_t = W0.copy()
            
            # Randomly zero out (23-target_rank) columns.
            cols_to_zero = rng.choice(23, size=(23-target_rank,), replace=False)
            for c in cols_to_zero:
                U_t[:, c] = 0
                V_t[:, c] = 0
                W_t[:, c] = 0
            
            # Now try to adjust the remaining columns to compensate.
            # Use a local search to minimize the error.
            err = compute_error(U_t, V_t, W_t)
            
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                # Pick a random non-zero column and perturb its entries.
                non_zero_cols = [c for c in range(23) if np.any(U_t[:, c] != 0)]
                if not non_zero_cols:
                    break
                c = non_zero_cols[rng.randint(len(non_zero_cols))]
                
                # Perturb a random entry in U, V, or W for this column.
                factor = rng.randint(3)
                if factor == 0:
                    row = rng.randint(U_t.shape[0])
                    old = U_t[row, c]
                    U_t[row, c] = old + rng.choice([-1, -0.5, 0.5, 1])
                    new_err = compute_error(U_t, V_t, W_t)
                    if new_err < err:
                        err = new_err
                        if err < 1e-8:
                            break
                    else:
                        U_t[row, c] = old
                elif factor == 1:
                    row = rng.randint(V_t.shape[0])
                    old = V_t[row, c]
                    V_t[row, c] = old + rng.choice([-1, -0.5, 0.5, 1])
                    new_err = compute_error(U_t, V_t, W_t)
                    if new_err < err:
                        err = new_err
                        if err < 1e-8:
                            break
                    else:
                        V_t[row, c] = old
                else:
                    row = rng.randint(W_t.shape[0])
                    old = W_t[row, c]
                    W_t[row, c] = old + rng.choice([-1, -0.5, 0.5, 1])
                    new_err = compute_error(U_t, V_t, W_t)
                    if new_err < err:
                        err = new_err
                        if err < 1e-8:
                            break
                    else:
                        W_t[row, c] = old
            
            if err < best_err:
                best_err = err
                best_U_t = U_t.copy()
                best_V_t = V_t.copy()
                best_W_t = W_t.copy()
            
            if err < 1e-8:
                break
        
        if best_U_t is not None and best_err < 1e-8:
            if verify(best_U_t, best_V_t, best_W_t) and is_half_integer(best_U_t) and is_half_integer(best_V_t) and is_half_integer(best_W_t):
                best_rank = target_rank
                best_U, best_V, best_W = best_U_t.copy(), best_V_t.copy(), best_W_t.copy()
                print(f"Found rank {target_rank}")
                break

    # Final check
    if verify(best_U, best_V, best_W) and is_half_integer(best_U) and is_half_integer(best_V) and is_half_integer(best_W):
        return {
            "U": to_half_int_list(best_U),
            "V": to_half_int_list(best_V),
            "W": to_half_int_list(best_W)
        }
    
    return fallback

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
