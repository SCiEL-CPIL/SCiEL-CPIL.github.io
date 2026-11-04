import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _build_tensor():
    n, m, p = 4, 4, 8
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)

def _to_half_integer(mat):
    mat2 = mat * 2
    rounded = np.round(mat2)
    if np.allclose(mat2, rounded, atol=1e-6):
        return (rounded / 2).tolist()
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    return mat.tolist()

def _schoolbook():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=-1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=-1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=-1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _build_4x4x4_49():
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)

    def offset(b):
        if b == 0: return 0
        if b == 1: return 2
        if b == 2: return 8
        if b == 3: return 10
        return 0

    A_blocks = [
        [(0,1),(3,1)],[(2,1),(3,1)],[(0,1)],[(3,1)],
        [(0,1),(1,1)],[(2,1),(0,-1)],[(1,1),(3,-1)]
    ]
    B_blocks = [
        [(0,1),(3,1)],[(0,1)],[(2,1),(3,-1)],[(1,1),(0,-1)],
        [(3,1)],[(0,1),(1,1)],[(2,1),(3,1)]
    ]
    C_blocks = [
        [(0,1),(3,1),(1,-1),(2,1)],[(2,1),(3,-1)],[(0,-1),(1,1)],
        [(1,-1),(3,1)],[(0,1),(1,-1),(3,1)],[(2,-1),(3,1)],[(0,1),(2,-1)]
    ]

    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            u_vals = U2[:, s]
            v_vals = V2[:, s]
            w_vals = W2[:, s]
            for i_local in range(4):
                val = u_vals[i_local]
                if val != 0:
                    for bA, sA in A_blocks[t]:
                        U4[offset(bA) + i_local, col] += val * sA
            for j_local in range(4):
                val = v_vals[j_local]
                if val != 0:
                    for bB, sB in B_blocks[t]:
                        V4[offset(bB) + j_local, col] += val * sB
            for k_local in range(4):
                val = w_vals[k_local]
                if val != 0:
                    for bC, sC in C_blocks[t]:
                        W4[offset(bC) + k_local, col] += val * sC
    return U4, V4, W4

def _build_4x4x8_98():
    U4, V4, W4 = _build_4x4x4_49()
    U = np.zeros((16, 98), dtype=np.float64)
    V = np.zeros((32, 98), dtype=np.float64)
    W = np.zeros((32, 98), dtype=np.float64)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local + 4, 49:98] = V4[j*4 + k_local, 0:49]
    for k_local in range(4):
        for i in range(4):
            W[(k_local+4)*4 + i, 49:98] = W4[k_local*4 + i, 0:49]
    return U, V, W

def _build_4x4x8_56():
    """
    Try a 56-rank construction: use 7 Strassen terms for the A-side,
    and for each term, the B-side and C-side handle all 8 columns.
    
    The idea: C[i,k] = sum_j A[i,j] B[j,k]
    Using Strassen on the 4x4 A matrix gives 7 "intermediate" values M_t.
    For each M_t, we need to multiply by a 4x8 matrix N_t(B) and add
    to C via P_t. But P_t is 4x4, so we need to extend it to handle 8 columns.
    
    Actually, the correct approach: think of B as two 4x4 blocks B1, B2.
    C = [A*B1 | A*B2].
    Using Strassen: A*B1 = sum_t M_t(A) * N_t(B1) * P_t  (7 terms, each rank-1 in the 4x4x4 sense)
    A*B2 = sum_t M_t(A) * N_t(B2) * P_t  (7 terms)
    
    In the full 4x4x8 tensor, each "term" t for B1 contributes to the first 16
    entries of C (k=0..3) and each term t for B2 contributes to the last 16 (k=4..7).
    
    The rank-1 term for (t, B1) is:
      u = M_t(A) expanded to 16-dim (A-side)
      v = N_t(B1) expanded to 32-dim (B-side, only first 16 entries nonzero)
      w = P_t expanded to 32-dim (C-side, only first 16 entries nonzero)
    
    But this gives 49+49=98 again. To get 56, we'd need each Strassen term to
    handle both B1 and B2 simultaneously, which requires the V and W vectors
    to have support on both halves.
    
    Let me try: for each of the 7 Strassen terms, create ONE rank-1 term where:
      U[:,r] = the A-side vector for term r (same as in 4x4x4)
      V[:,r] = [N_t(B1); N_t(B2)]  (32-dim, both halves)
      W[:,r] = [P_t; P_t]  (32-dim, both halves)
    
    But this won't work because the C-side needs different P_t for different k positions.
    In 4x4x4, P_t is a 4x4 matrix, meaning for each output position (i,k), there's
    a coefficient. In the rank-1 decomposition, W is a vector, not a matrix.
    
    The 49 rank for 4x4x4 comes from 7x7 = 49 rank-1 terms, where the extra 7
    dimension comes from the fact that each Strassen "product" M_t * N_t produces
    a 4x4 matrix, which requires 16 rank-1 terms, but Strassen reduces this to 7
    per product, and there are 7 products, giving 49.
    
    Wait, I'm confusing myself. Let me re-examine.
    
    In the 4x4x4 Strassen decomposition with rank 49:
    - There are 49 rank-1 terms
    - These come from 7 "groups" (one per Strassen product) x 7 "sub-terms" per group
    - The 7 sub-terms per group correspond to the 7 independent rank-1 components
      of the 4x4 output matrix produced by that Strassen product
    
    For 4x4x8: if we could make each group produce only 4 sub-terms (one per output
    row, since the 8 output columns are handled by the B-side), we'd get 7*4=28.
    But that's not how it works.
    
    Let me just try the straightforward 56 = 7 * 8 construction:
    For each of 7 Strassen terms t, and each of 8 output columns k:
      One rank-1 term.
    
    U[:, (t,k)] = A-side vector for Strassen term t (16-dim)
    V[:, (t,k)] = B-side vector for Strassen term t, column k (32-dim)  
    W[:, (t,k)] = C-side vector for Strassen term t, column k (32-dim)
    
    But this doesn't make sense either because V and W are coupled.
    
    Let me abandon the 56 idea and focus on the null-space approach.
    """
    return None

def _null_space_reduce(U, V, W, T, max_iterations=5000, deadline=None):
    """
    Iteratively try to remove columns from the decomposition.
    For each candidate column r, check if the tensor contribution of column r
    can be exactly represented by linear combinations of other columns.
    
    The tensor contribution of column r is:
      T_r = U[:,r] * V[:,r] * W[:,r]  (outer product, 16x32x32)
    
    We want to find coefficients alpha_s (for s != r) such that:
      T_r = sum_{s != r} alpha_s * U[:,s] * V[:,s] * W[:,s]
    
    This is a linear system in the alpha_s. The number of equations is 16*32*32 = 16384,
    and the number of unknowns is (rank-1). For rank=98, that's 97 unknowns and 16384 equations.
    The system is overdetermined, so a solution exists only if T_r is in the span of the other terms.
    
    We can check this by:
    1. Building the 16384 x 97 matrix M where M[entry, s] = (U[:,s]*V[:,s]*W[:,s])[entry]
    2. Checking if T_r is in the column space of M
    
    This is expensive, so we use a smarter approach:
    - Only check a subset of entries (the 128 nonzero entries of T)
    - Use SVD to check if T_r is in the span
    
    Actually, even simpler: if the full set of 98 columns is linearly independent
    in the tensor space, we can't remove any. But they might not be.
    
    Let's use a different approach: Gaussian elimination on the constraint matrix.
    """
    if deadline is not None and time.time() > deadline - 5:
        return U, V, W, U.shape[1]
    
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    # Get the 128 nonzero entries of T
    nonzero_entries = np.argwhere(T != 0)
    n_nonzero = len(nonzero_entries)
    
    # Build the constraint matrix: for each nonzero entry (a,b,c) and each column r,
    # the coefficient is U[a,r] * V[b,r] * W[c,r]
    # We want to check if removing column r still allows exact reconstruction.
    
    # For efficiency, work with the 128 nonzero entries only.
    # constraint_matrix[n_nonzero, r] = U[a_r, r] * V[b_r, r] * W[c_r, r]
    # where (a_r, b_r, c_r) = nonzero_entries[r]
    
    # Actually, the constraint is:
    # For each (a,b,c) in nonzero_entries: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # For each (a,b,c) not in nonzero_entries: sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # The second set of constraints (16384 - 128 = 16256 equations) is what makes it hard.
    # If we only enforce the 128 nonzero constraints, we get a much less constrained system.
    
    # Strategy: try to find a lower-rank solution that satisfies ALL constraints.
    # Use the following iterative approach:
    # 1. Start with rank R
    # 2. For each column r, try to express it as a combination of others
    # 3. If successful, remove it and reduce R
    
    # Due to the complexity, let's try a simpler heuristic:
    # Check if any two columns are "similar" in a way that allows merging.
    
    # For now, just return the input unchanged. The real work is in the
    # constructive approach below.
    return U, V, W, rank

def _build_4x4x8_64():
    """
    Try a 64-rank construction using a different block structure.
    
    Idea: instead of splitting B into two 4x4 blocks and doing 49+49,
    use a 2x2 block structure for A and handle the 8 columns of B
    more efficiently.
    
    A is 4x4 = 2x2 blocks of 2x2.
    B is 4x8.
    C is 4x8.
    
    C_top (2x8) = A00 * B_top + A01 * B_bottom
    C_bottom (2x8) = A10 * B_top + A11 * B_bottom
    
    Where B_top is 2x8, B_bottom is 2x8.
    
    Now, 2x8 = 4 blocks of 2x2. Let's call them B_top = [Bt0 Bt1 Bt2 Bt3]
    and B_bottom = [Bb0 Bb1 Bb2 Bb3].
    
    C_top = [A00*Bt0 + A01*Bb0 | A00*Bt1 + A01*Bb1 | A00*Bt2 + A01*Bb2 | A00*Bt3 + A01*Bb3]
    
    Each 2x2*2x2 product uses 7 Strassen terms.
    For C_top, we have 4 such "sums" of 2x2*2x2 products.
    Each sum uses 14 terms (7 for A00*Bti, 7 for A01*Bbi).
    Total for C_top: 4 * 14 = 56 terms.
    Similarly for C_bottom: 56 terms.
    Total: 112. Worse than 98.
    
    What if we share the Strassen products between adjacent blocks?
    The issue is that Bt0 and Bt1 are different matrices, so N_t(Bt0) != N_t(Bt1).
    
    Alternative: use the fact that 2x8 can be handled with fewer terms.
    The rank of 2x2x8 matrix multiplication is... let's think.
    C[i,k] = sum_j A[i,j] B[j,k] where A is 2x2, B is 2x8, C is 2x8.
    This is 2*2*8 = 32 schoolbook terms.
    Can we do better? The rank of 2x2xn is known to be 2n-2 for n>=2 (I think).
    For n=8: rank = 14.
    
    So 2x2x8 has rank 14! That means:
    C_top = A00 * B_top + A01 * B_bottom
    This is a sum of two 2x2x8 products, each of rank 14.
    But they share the A-side structure...
    
    Actually, C_top[i,k] = sum_j (A00[i,j]*B_top[j,k] + A01[i,j]*B_bottom[j,k])
    = sum_j A_combined[i,j] * B_combined[j,k]
    where A_combined is 2x4 (columns: A00 col0, A00 col1, A01 col0, A01 col1)
    and B_combined is 4x8 (rows: B_top row0, B_top row1, B_bottom row0, B_bottom row1)
    
    So C_top is a 2x4 * 4x8 = 2x8 product, which has schoolbook rank 2*4*8 = 64.
    The tensor rank of 2x4x8 is at most 64, and likely less.
    
    This is getting complicated. Let me just try to verify the 98-rank solution
    and return it, with a fallback to schoolbook.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Fallback: Schoolbook (rank 128)
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 128
    
    # Try 98-rank constructive solution
    if time.time() < deadline - 10:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            if _verify(U98, V98, W98, T):
                best_U, best_V, best_W = U98, V98, W98
                best_rank = 98
        except Exception:
            pass
    
    # Try to reduce rank using null-space / column elimination
    if time.time() < deadline - 10 and best_rank <= 98:
        try:
            U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
            rank_cur = U_cur.shape[1]
            
            # Try removing columns one at a time
            # For each column r, check if T - T_r is in the span of other columns
            # This is expensive, so use a randomized approach:
            # Try random subsets of columns to remove
            
            # Simpler: check if the current solution has any linearly dependent columns
            # in the tensor space. Build the 16384 x rank matrix and check rank.
            
            # For efficiency, only use the 128 nonzero entries of T.
            nonzero = np.argwhere(T != 0)
            n_nz = len(nonzero)
            
            # Build constraint matrix: n_nz x rank
            # constraint[i, r] = U[a_i, r] * V[b_i, r] * W[c_i, r]
            a_idx = nonzero[:, 0]
            b_idx = nonzero[:, 1]
            c_idx = nonzero[:, 2]
            
            # constraint[i, r] = U[a_i, r] * V[b_i, r] * W[c_i, r]
            # This is a 3-way product, so we can't easily build it as a matrix product.
            # Instead, for each r, the column is U[:,r] * V[:,r] * W[:,r] evaluated at (a_i, b_i, c_i)
            
            # Build it element by element (n_nz is only 128)
            constraint = np.zeros((n_nz, rank_cur), dtype=np.float64)
            for i in range(n_nz):
                a, b, c = a_idx[i], b_idx[i], c_idx[i]
                constraint[i, :] = U_cur[a, :] * V_cur[b, :] * W_cur[c, :]
            
            # Check the rank of this matrix
            # If rank(constraint) < rank_cur, then some columns are dependent
            # and we can potentially reduce.
            
            # Use SVD to find the rank
            try:
                s, _, _ = np.linalg.svd(constraint, full_matrices=False)
                tol = 1e-10
                eff_rank = np.sum(s > tol)
                if eff_rank < rank_cur:
                    # There are dependent columns. Find which ones to remove.
                    # Use the null space to identify redundant columns.
                    # For now, just note that reduction is possible.
                    pass
            except Exception:
                pass
            
        except Exception:
            pass
    
    # Final attempt: try a direct 56-rank construction by solving a linear system
    if time.time() < deadline - 10 and best_rank > 56:
        try:
            # Try to find a 56-rank solution by exploiting the structure.
            # The 4x4x8 tensor has 128 nonzero entries.
            # A rank-56 solution has 56*16 + 56*32 + 56*32 = 896 + 1792 + 1792 = 4480 variables.
            # The constraints are 16384 equations (128 = 1, 16256 = 0).
            # This is a nonlinear system (products of variables), so it's hard.
            
            # Instead, use the following structured approach:
            # Assume the solution has the form where U is 16x56, and
            # the columns come in 7 groups of 8.
            # Group t (t=0..6) corresponds to Strassen term t.
            # Within each group, the 8 columns correspond to the 8 output columns.
            
            # For group t:
            #   U[:, t*8 + k] = u_t for all k (same A-side vector)
            #   V[:, t*8 + k] = v_tk (B-side, depends on k)
            #   W[:, t*8 + k] = w_tk (C-side, depends on k)
            
            # The tensor equation for entry (a, b, c):
            # sum_t sum_k U[a, t*8+k] * V[b, t*8+k] * W[c, t*8+k] = T[a,b,c]
            # = sum_t U[a, t*8] * sum_k V[b, t*8+k] * W[c, t*8+k] = T[a,b,c]
            
            # Let M_t = sum_k V[b, t*8+k] * W[c, t*8+k] (a 32x32 matrix for each t)
            # Then: sum_t U[a, t*8] * M_t[b, c] = T[a,b,c]
            
            # This is a linear system in U and M_t.
            # U is 16x7 (one vector per group), and each M_t is 32x32.
            # Total variables: 16*7 + 7*32*32 = 112 + 7168 = 7280.
            # Constraints: 16384.
            # Still overdetermined, but the structure might allow a solution.
            
            # This is getting too complex for the time budget. Skip.
            pass
        except Exception:
            pass
    
    U_out = _to_half_integer(best_U)
    V_out = _to_half_integer(best_V)
    W_out = _to_half_integer(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
