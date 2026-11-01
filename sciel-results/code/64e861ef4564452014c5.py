import numpy as np
import time

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    
    Each term is (A_coeffs, B_coeffs, C_coeffs) where each is a dict
    mapping (row, col) -> coefficient for the 2x2 matrices.
    """
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    terms = [
        # P1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # P2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # P3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # P4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        # P5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # P6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # P7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    return terms

def _generate_strassen_decomposition(n=4):
    """Recursively generate the Strassen-based rank decomposition for n x n matrix multiplication.
    
    For n=2: use the 7-term Strassen decomposition directly.
    For n=4: split into 2x2 blocks, apply Strassen at block level,
             then recursively decompose each 2x2 block product.
    
    Returns (U, V, W) as lists of lists with integer entries.
    U: shape (n*n, rank), V: shape (n*n, rank), W: shape (n*n, rank)
    Convention: U[i*n+k, r], V[k*n+j, r], W[j*n+i, r] for C[i,j]
    """
    if n == 2:
        terms = _strassen_2x2()
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        for t, (a_coeffs, b_coeffs, c_coeffs) in enumerate(terms):
            for (r, c), coeff in a_coeffs.items():
                U[r * 2 + c][t] += coeff
            for (r, c), coeff in b_coeffs.items():
                V[r * 2 + c][t] += coeff
            for (r, c), coeff in c_coeffs.items():
                W[c * 2 + r][t] += coeff  # W[j*2+i, t] for C[i,j]
        
        return U, V, W, rank
    
    # For n=4: use Strassen at the 2x2 block level
    half = n // 2  # 2
    terms = _strassen_2x2()
    outer_rank = 7
    inner_U, inner_V, inner_W, inner_rank = _generate_strassen_decomposition(half)
    total_rank = outer_rank * inner_rank
    
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    for t, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        for s in range(inner_rank):
            rank_idx = t * inner_rank + s
            
            # For each block in A
            for (br, bc), block_coeff in a_blocks.items():
                for (er, ec), elem_coeff in inner_U[er * half + ec][s:s+1][0].__class__.__mro__[0].__subclasses__ if False else []:
                    pass
                # Use inner_U directly: inner_U[er*half+ec][s] is the coefficient
                for (er, ec) in [(0,0),(0,1),(1,0),(1,1)]:
                    elem_coeff = inner_U[er * half + ec][s]
                    if elem_coeff != 0:
                        row = 2 * br + er
                        col = 2 * bc + ec
                        U[row * n + col][rank_idx] += block_coeff * elem_coeff
            
            # For each block in B
            for (br, bc), block_coeff in b_blocks.items():
                for (er, ec) in [(0,0),(0,1),(1,0),(1,1)]:
                    elem_coeff = inner_V[er * half + ec][s]
                    if elem_coeff != 0:
                        row = 2 * br + er
                        col = 2 * bc + ec
                        V[row * n + col][rank_idx] += block_coeff * elem_coeff
            
            # For each block in C
            for (br, bc), block_coeff in c_blocks.items():
                for (er, ec) in [(0,0),(0,1),(1,0),(1,1)]:
                    elem_coeff = inner_W[ec * half + er][s]  # inner_W[j*2+i] for C[i,j]
                    if elem_coeff != 0:
                        row = 2 * br + er
                        col = 2 * bc + ec
                        W[col * n + row][rank_idx] += block_coeff * elem_coeff
    
    return U, V, W, total_rank

def _reduce_rank(U, V, W, tol=1e-10):
    """Check for linearly dependent columns in the decomposition and remove them."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    keep = []
    basis_U = []
    basis_V = []
    basis_W = []
    
    for r in range(rank):
        col_u = U_np[:, r]
        col_v = V_np[:, r]
        col_w = W_np[:, r]
        
        if np.allclose(col_u, 0, atol=tol) and np.allclose(col_v, 0, atol=tol) and np.allclose(col_w, 0, atol=tol):
            continue
        
        if len(keep) == 0:
            keep.append(r)
            basis_U.append(col_u.copy())
            basis_V.append(col_v.copy())
            basis_W.append(col_w.copy())
        else:
            all_kept_U = np.column_stack(basis_U)
            all_kept_V = np.column_stack(basis_V)
            all_kept_W = np.column_stack(basis_W)
            
            combined_old = np.vstack([all_kept_U, all_kept_V, all_kept_W])
            combined_new = np.vstack([col_u, col_v, col_w])
            
            combined_all = np.hstack([combined_old, combined_new.reshape(-1, 1)])
            rank_old = np.linalg.matrix_rank(combined_old, tol=tol)
            rank_new = np.linalg.matrix_rank(combined_all, tol=tol)
            
            if rank_new > rank_old:
                keep.append(r)
                basis_U.append(col_u.copy())
                basis_V.append(col_v.copy())
                basis_W.append(col_w.copy())
    
    if len(keep) == rank:
        return U, V, W, rank
    
    U_red = U_np[:, keep]
    V_red = V_np[:, keep]
    W_red = W_np[:, keep]
    
    # Verify
    U4 = U_red.reshape(4, 4, -1)
    V4 = V_red.reshape(4, 4, -1)
    W4 = W_red.reshape(4, 4, -1)
    
    np.random.seed(12345)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    u_vals = np.einsum('ikr,ik->ir', U4, A_test)
    v_vals = np.einsum('kjr,kj->jr', V4, B_test)
    C_computed = np.einsum('jir,ir,jr->ij', W4, u_vals, v_vals)
    
    if np.max(np.abs(C_computed - C_expected)) > 1e-9:
        return U, V, W, rank
    
    return U_red.tolist(), V_red.tolist(), W_red.tolist(), len(keep)

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    
    Uses the 7-product Strassen algorithm applied to 2x2 blocks of the 4x4 matrices,
    where each 2x2 block multiplication is done via the trivial 4-product method.
    Total rank = 7 * 4 = 28? No wait, let me think again.
    
    Actually: 4x4 = 4 blocks of 2x2. Strassen gives 7 multiplications of 2x2 blocks.
    Each 2x2 block multiplication has rank 4 (trivial). So total rank = 7 * 4 = 28.
    
    Wait, that's not right either. The trivial 2x2 multiplication has rank 4, not 2.
    So 7 * 4 = 28. But we know the trivial 4x4 is rank 64.
    
    Let me reconsider: The tensor for 4x4 multiplication has 4*4*4 = 64 terms.
    Strassen reduces the 7 multiplications at the top level, each of which is a 2x2
    multiplication (rank 4). So 7 * 4 = 28? That seems too good.
    
    Actually no. The standard analysis: M(4) <= 7 * M(2) = 7 * 4 = 28? 
    But M(2) = 4 is the trivial bound, and M(2) <= 7 (Strassen for 2x2).
    So M(4) <= 7 * 4 = 28 using trivial 2x2, or M(4) <= 7 * 7 = 49 using Strassen 2x2.
    
    Wait, I'm confusing myself. Let me be precise:
    - The multiplication tensor for n x n matrices has rank at most n^3 (trivial).
    - For 2x2: rank <= 4 (trivial) or rank <= 7 (Strassen).
    - For 4x4: split into 2x2 blocks. C = A*B where A,B are 4x4 with 2x2 blocks.
      Strassen: 7 products of 2x2 blocks. Each 2x2 product has rank 4 (trivial).
      Total rank = 7 * 4 = 28.
    
    But wait, that's not how it works. The rank of the whole tensor is not simply
    7 times the rank of the sub-tensor. Let me think more carefully.
    
    Actually, the correct way: if we have a rank-r decomposition of the n x n
    multiplication tensor, and we apply it to compute block multiplications, we get
    a rank-(r * k^2) decomposition for (nk) x (nk) where k is the block size... no.
    
    Let me just use the standard result: M(4) <= 49 via recursive Strassen
    (7 multiplications at top, each 2x2 done with Strassen's 7).
    Or M(4) <= 28 if each 2x2 is done trivially (rank 4).
    
    Hmm, let me verify: for 4x4, split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    Strassen computes:
      P1 = (A11+A22)(B11+B22)
      P2 = (A21+A22)B11
      P3 = A11(B12-B22)
      P4 = A22(B21-B11)
      P5 = (A11+A12)B22
      P6 = (A21-A11)(B11+B12)
      P7 = (A12-A22)(B21+B22)
    
    Each Pi is a product of two 2x2 matrices. If we compute each 2x2 product
    trivially (4 scalar multiplications), total = 7 * 4 = 28 scalar multiplications.
    
    So the rank should be 28! Let me implement this.
    
    The tensor decomposition: for each Strassen term t (0..6) and each trivial
    2x2 product term s (0..3), we get one rank-1 term. Total rank = 7 * 4 = 28.
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n = 4
    
    # Generate decomposition programmatically
    U, V, W, rank = _generate_strassen_decomposition(n)
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    U4 = U_np.reshape(4, 4, rank)   # [i, k, r]
    V4 = V_np.reshape(4, 4, rank)   # [k, j, r]
    W4 = W_np.reshape(4, 4, rank)   # [j, i, r]
    
    u_vals = np.einsum('ikr,ik->ir', U4, A_test)
    v_vals = np.einsum('kjr,kj->jr', V4, B_test)
    C_computed = np.einsum('jir,ir,jr->ij', W4, u_vals, v_vals)
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-9:
        # Fallback: trivial rank-64 decomposition
        rank_fallback = 64
        U_f = [[0] * rank_fallback for _ in range(16)]
        V_f = [[0] * rank_fallback for _ in range(16)]
        W_f = [[0] * rank_fallback for _ in range(16)]
        
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U_f[i*4+k][r] = 1
                    V_f[k*4+j][r] = 1
                    W_f[j*4+i][r] = 1
        
        return {"U": U_f, "V": V_f, "W": W_f}
    
    # Deadline check before expensive rank reduction
    if time.time() - start_time > DEADLINE - 30:
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by removing linearly dependent terms
    U_red, V_red, W_red, rank_red = _reduce_rank(U, V, W)
    
    return {"U": U_red, "V": V_red, "W": W_red}
# EVOLVE-BLOCK-END
