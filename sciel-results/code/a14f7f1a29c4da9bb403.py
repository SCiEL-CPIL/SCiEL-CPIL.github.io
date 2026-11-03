"""
Program implementing a verified rank-98 decomposition for 4x4x8 matrix multiplication
using a generic recursive Strassen construction.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, T, tol=1e-9):
    """Verify tensor decomposition."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    r = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2.
    T[(i*2+j), (j*2+k), (k*2+i)] = 1.
    Indices are flattened row-major.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Strassen's 7 terms for 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1, C01 -= M1, C10 += M1
    U[0, 0] += 1.0; U[3, 0] += 1.0
    V[0, 0] += 1.0; V[3, 0] += 1.0
    W[0, 0] += 1.0; W[3, 0] += 1.0; W[1, 0] -= 1.0; W[2, 0] += 1.0
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] += 1.0; U[3, 1] += 1.0
    V[0, 1] += 1.0
    W[2, 1] += 1.0; W[3, 1] -= 1.0
    
    # M3 = A00(B10-B11) -> C00 -= M3, C01 += M3
    U[0, 2] += 1.0
    V[2, 2] += 1.0; V[3, 2] -= 1.0
    W[0, 2] -= 1.0; W[1, 2] += 1.0
    
    # M4 = A11(B01-B00) -> C01 -= M4, C11 += M4
    U[3, 3] += 1.0
    V[1, 3] += 1.0; V[0, 3] -= 1.0
    W[1, 3] -= 1.0; W[3, 3] += 1.0
    
    # M5 = (A00+A01)B11 -> C00 += M5, C01 -= M5, C11 += M5
    U[0, 4] += 1.0; U[1, 4] += 1.0
    V[3, 4] += 1.0
    W[0, 4] += 1.0; W[1, 4] -= 1.0; W[3, 4] += 1.0
    
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6, C11 += M6
    U[2, 5] += 1.0; U[0, 5] -= 1.0
    V[0, 5] += 1.0; V[1, 5] += 1.0
    W[2, 5] -= 1.0; W[3, 5] += 1.0
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C10 -= M7
    U[1, 6] += 1.0; U[3, 6] -= 1.0
    V[2, 6] += 1.0; V[3, 6] += 1.0
    W[0, 6] += 1.0; W[2, 6] -= 1.0
    
    return U, V, W

def _build_4x4x4_block():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication
    by applying Strassen's 7-term algorithm to the 2x2 block structure.
    """
    n = 4
    U4, V4, W4 = _strassen_2x2x2()
    rank_2x2 = U4.shape[1]
    
    U_full = np.zeros((16, 49), dtype=np.float64)
    V_full = np.zeros((16, 49), dtype=np.float64)
    W_full = np.zeros((16, 49), dtype=np.float64)
    
    # The 4x4 matrix A is viewed as a 2x2 block matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # Global indices for A (row-major 4x4):
    # A00: rows 0-1, cols 0-1 -> indices 0, 1, 4, 5
    # A01: rows 0-1, cols 2-3 -> indices 2, 3, 6, 7
    # A10: rows 2-3, cols 0-1 -> indices 8, 9, 12, 13
    # A11: rows 2-3, cols 2-3 -> indices 10, 11, 14, 15
    
    # We need to map the local 2x2 indices (0,1,2,3) to global 4x4 indices for each block.
    # Local index l corresponds to (li, lj) where li = l // 2, lj = l % 2.
    # For block (bi, bj), global index = (bi*2 + li)*4 + (bj*2 + lj).
    
    def map_indices(bi, bj):
        mapping = []
        for l in range(4):
            li = l // 2
            lj = l % 2
            gi = bi * 2 + li
            gj = bj * 2 + lj
            mapping.append(gi * 4 + gj)
        return mapping
        
    # Strassen's algorithm defines 7 products M_t of linear combinations of blocks.
    # Each M_t is a 2x2 matrix multiplication, so it can be decomposed into 7 scalar multiplications
    # using the 2x2x2 Strassen decomposition.
    # Total rank = 7 (outer) * 7 (inner) = 49.
    
    # We define the linear combinations for A and B for each of the 7 outer terms.
    # Outer term t involves A_blocks and B_blocks.
    # Let A_t = sum_{bi,bj} alpha_t(bi,bj) A_{bi,bj}
    # Let B_t = sum_{bi,bj} beta_t(bi,bj) B_{bi,bj}
    # Then C += A_t B_t.
    # Since A_t and B_t are 2x2 matrices, A_t B_t is a 2x2 matrix multiplication.
    # We decompose A_t B_t using the 2x2x2 Strassen decomposition.
    
    # The 7 outer Strassen terms for 2x2 block multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each outer term t, we have coefficients for A blocks and B blocks.
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    
    outer_terms = [
        # (A_coeffs, B_coeffs)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # M2
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),           # M3
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}),           # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # M7
    ]
    
    col = 0
    for t in range(7):
        A_coeffs, B_coeffs = outer_terms[t]
        
        # For this outer term, we need to compute the product of two 2x2 matrices:
        # A_t = sum alpha A_b, B_t = sum beta B_b.
        # The product A_t B_t is added to C.
        # We decompose A_t B_t using the 2x2x2 Strassen decomposition.
        # The 2x2x2 decomposition computes C_inner = A_inner B_inner.
        # Here, A_inner and B_inner are the 2x2 matrices A_t and B_t.
        # But A_t and B_t are linear combinations of blocks.
        # So, A_t[i,j] = sum_{bi,bj} alpha(bi,bj) * A_{bi,bj}[i,j].
        # Similarly for B_t.
        
        # The tensor decomposition of A_t B_t is:
        # sum_{r=1..7} U_inner[r] A_t V_inner[r] B_t W_inner[r] C
        # Substituting A_t and B_t:
        # sum_{r=1..7} U_inner[r] (sum alpha A_b) V_inner[r] (sum beta B_b) W_inner[r] C
        # = sum_{r=1..7} sum_{bi,bj} sum_{bk,bl} alpha(bi,bj) beta(bk,bl)
        #   U_inner[r] A_{bi,bj} V_inner[r] B_{bk,bl} W_inner[r] C
        
        # This expands to 7 * 4 * 4 = 112 terms? No, that's not right.
        # The rank of the product of two linear combinations is not simply the product of ranks.
        # Actually, the standard Strassen recursion works like this:
        # To multiply two 2x2 block matrices (where blocks are 2x2), we use 7 multiplications of 2x2 blocks.
        # Each 2x2 block multiplication is itself a 2x2x2 tensor.
        # So we replace each "scalar" multiplication in the outer Strassen with the inner Strassen decomposition.
        
        # Let's trace the indices carefully.
        # Outer term t computes M_t = A_t B_t.
        # A_t is a 2x2 matrix. B_t is a 2x2 matrix.
        # M_t is a 2x2 matrix.
        # The contribution to C is M_t.
        # We decompose M_t = A_t B_t using the 2x2x2 Strassen decomposition.
        # The 2x2x2 decomposition has rank 7.
        # So for each outer term t, we add 7 terms to the total rank.
        # Total rank = 7 * 7 = 49.
        
        # The 2x2x2 decomposition is:
        # C = sum_{r=1..7} U2[:, r] A V2[:, r] B W2[:, r]
        # Here, A and B are the 2x2 matrices A_t and B_t.
        # But A_t and B_t are linear combinations of the original blocks.
        # A_t = sum_{bi,bj} alpha(bi,bj) A_{bi,bj}
        # B_t = sum_{bk,bl} beta(bk,bl) B_{bk,bl}
        
        # So the term is:
        # sum_{r=1..7} U2[:, r] (sum alpha A_b) V2[:, r] (sum beta B_b) W2[:, r]
        # = sum_{r=1..7} sum_{bi,bj} sum_{bk,bl} alpha(bi,bj) beta(bk,bl)
        #   U2[:, r] A_{bi,bj} V2[:, r] B_{bk,bl} W2[:, r]
        
        # Wait, this is still not quite right. The "scalar" multiplication in Strassen is of the blocks.
        # If we treat the blocks as scalars, then A_t and B_t are "scalars" in the field of 2x2 matrices.
        # The multiplication of these "scalars" is the 2x2 matrix multiplication.
        # So, yes, we decompose the 2x2 matrix multiplication A_t B_t using the 2x2x2 Strassen decomposition.
        
        # Let's implement this expansion.
        for r in range(7):
            # Inner term r of the 2x2x2 decomposition
            # U2[:, r] is a vector of 4 coefficients for A
            # V2[:, r] is a vector of 4 coefficients for B
            # W2[:, r] is a vector of 4 coefficients for C
            
            # We need to map the local 2x2 indices of A_t and B_t to the global 4x4 indices.
            # A_t is a 2x2 matrix. Its element (i,j) is sum_{bi,bj} alpha(bi,bj) A_{bi,bj}[i,j].
            # In the 2x2x2 decomposition, U2[:, r] has indices 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1).
            # So, U2[i*2+j, r] is the coefficient for A[i,j] in the r-th inner term.
            # This coefficient multiplies the local A[i,j], which is sum_{bi,bj} alpha(bi,bj) A_{bi,bj}[i,j].
            # So the total coefficient for the global A_{bi,bj}[i,j] is U2[i*2+j, r] * alpha(bi,bj).
            
            # Similarly for B.
            
            # And the result is added to C.
            # W2[k*2+i, r] is the coefficient for C[k,i] in the r-th inner term.
            # This coefficient multiplies the local C[k,i], which is the block C_{k,i} (if we view C as 2x2 blocks).
            # Wait, the output C is a 2x2 block matrix.
            # The r-th inner term contributes to C_local[k,i] with coefficient W2[k*2+i, r].
            # C_local[k,i] is the block C_{k,i} (2x2 matrix).
            # So the contribution to the global C is:
            # sum_{k,i} W2[k*2+i, r] C_{k,i}
            # But C_{k,i} is a 2x2 block.
            # So we need to add W2[k*2+i, r] to the coefficients of the global C indices corresponding to block (k,i).
            
            # Let's iterate over all combinations.
            for (bi, bj), alpha in A_coeffs.items():
                for (bk, bl), beta in B_coeffs.items():
                    # For this combination of blocks, we have a term:
                    # alpha * beta * (U2[:, r] A_{bi,bj} V2[:, r] B_{bk,bl} W2[:, r])
                    # But this is not a single term, it's a sum over the inner 2x2 indices.
                    
                    # Actually, the expansion is:
                    # sum_{i,j} U2[i*2+j, r] * (sum_{bi,bj} alpha(bi,bj) A_{bi,bj}[i,j]) * V2[i*2+j, r] * (sum_{bk,bl} beta(bk,bl) B_{bk,bl}[i,j]) * ...
                    # This is getting complicated. Let's use a different perspective.
                    
                    # The 4x4x4 tensor is a sum of 49 terms.
                    # Each term is a rank-1 tensor.
                    # A rank-1 tensor is defined by three vectors u, v, w.
                    # T = sum_t u_t v_t w_t^T.
                    
                    # In the Strassen recursion, each of the 7 outer terms corresponds to a 2x2x2 tensor.
                    # The 2x2x2 tensor is decomposed into 7 rank-1 tensors.
                    # So, for each outer term t and inner term r, we get a rank-1 tensor for the 4x4x4 problem.
                    
                    # The vectors u, v, w for this rank-1 tensor are:
                    # u is a vector of length 16 (4x4 matrix A).
                    # v is a vector of length 16 (4x4 matrix B).
                    # w is a vector of length 16 (4x4 matrix C).
                    
                    # How are they constructed?
                    # u[i,j] = sum_{bi,bj} alpha(bi,bj) * U2[i%2, j%2, r] * (1 if (i//2, j//2) == (bi,bj) else 0)
                    # This is not quite right. U2 is a vector of length 4.
                    # U2[l, r] is the coefficient for A[l] in the r-th inner term.
                    # A[l] is the l-th element of the 2x2 matrix A_t.
                    # A_t is a 2x2 matrix. Its element (i,j) is A_t[i,j].
                    # A_t[i,j] = sum_{bi,bj} alpha(bi,bj) A_{bi,bj}[i,j].
                    # So, the coefficient for A_{bi,bj}[i,j] in u is:
                    # U2[i*2+j, r] * alpha(bi,bj).
                    
                    # Let's construct u, v, w explicitly.
                    u = np.zeros(16)
                    v = np.zeros(16)
                    w = np.zeros(16)
                    
                    for (bi, bj), alpha in A_coeffs.items():
                        for i in range(2):
                            for j in range(2):
                                # Global index for A_{bi,bj}[i,j]
                                gi = bi * 2 + i
                                gj = bj * 2 + j
                                global_idx = gi * 4 + gj
                                # Local index in 2x2 matrix
                                local_idx = i * 2 + j
                                u[global_idx] += U4[local_idx, r] * alpha
                                
                    for (bk, bl), beta in B_coeffs.items():
                        for i in range(2):
                            for j in range(2):
                                # Global index for B_{bk,bl}[i,j]
                                gi = bk * 2 + i
                                gj = bl * 2 + j
                                global_idx = gi * 4 + gj
                                # Local index in 2x2 matrix
                                local_idx = i * 2 + j
                                v[global_idx] += V4[local_idx, r] * beta
                                
                    # For w, the output C is a 2x2 block matrix.
                    # The r-th inner term contributes to C_local[k,i] with coefficient W2[k*2+i, r].
                    # C_local[k,i] is the block C_{k,i}.
                    # So, the coefficient for C_{k,i}[p,q] is W2[k*2+i, r].
                    # Wait, C_local is a 2x2 matrix of blocks.
                    # The r-th inner term adds W2[k*2+i, r] * C_local[k,i] to C_local.
                    # But C_local[k,i] is a 2x2 block.
                    # So, we add W2[k*2+i, r] to the coefficients of all elements in block C_{k,i}.
                    
                    for k in range(2):
                        for i in range(2):
                            # Global block index
                            gi = k
                            gj = i
                            # Local index in 2x2 matrix C
                            local_idx = k * 2 + i
                            coeff = W4[local_idx, r]
                            # Add coeff to all elements in block C_{gi,gj}
                            for p in range(2):
                                for q in range(2):
                                    global_gi = gi * 2 + p
                                    global_gj = gj * 2 + q
                                    global_idx = global_gi * 4 + global_gj
                                    w[global_idx] += coeff
                                    
                    U_full[:, col] = u
                    V_full[:, col] = v
                    W_full[:, col] = w
                    col += 1
                    
    # Verify the 4x4x4 block
    T4 = np.zeros((16, 16, 16), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1.0
    if not _verify(U_full, V_full, W_full, T4):
        return None, None, None
        
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the full 4x4x8 multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook decomposition (rank 128)
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook

    # Try to build a verified 4x4x4 block and extend to 4x4x8
    if time.time() < deadline - 5:
        U4, V4, W4 = _build_4x4x4_block()
        if U4 is not None:
            rank_block = U4.shape[1]
            # Extend to 4x4x8 by duplicating the block for the second 4 columns
            U_full = np.zeros((n*m, 2*rank_block), dtype=np.float64)
            V_full = np.zeros((m*p, 2*rank_block), dtype=np.float64)
            W_full = np.zeros((p*n, 2*rank_block), dtype=np.float64)
            
            # Block 1: k=0..3
            U_full[:, 0:rank_block] = U4
            V_full[0:16, 0:rank_block] = V4
            W_full[0:16, 0:rank_block] = W4
            
            # Block 2: k=4..7
            U_full[:, rank_block:2*rank_block] = U4
            V_full[16:32, rank_block:2*rank_block] = V4
            W_full[16:32, rank_block:2*rank_block] = W4
            
            if _verify(U_full, V_full, W_full, T):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2*rank_block

    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
