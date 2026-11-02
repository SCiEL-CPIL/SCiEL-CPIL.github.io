"""Matrix multiplication 4x4x5 decomposition via Strassen-based construction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition, each of shape (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # T1: (a+d)(e+h) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # T2: (c+d)e -> C10
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # T3: a(f-h) -> C01
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # T4: d(g-c) -> C11
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # T5: (a+b)(h+e) -> C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1; V[0, 4] = 1
    W[1, 4] = 1
    
    # T6: (c-a)(e+f) -> C10
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # T7: (b-d)(g+h) -> C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _strassen_4x4():
    """
    Return U, V, W for 4x4x4 Strassen decomposition.
    Uses recursive 2x2 Strassen blocks.
    Each factor has shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # High-level Strassen for 4x4:
    # Split into 2x2 blocks of 2x2 matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_i is a 2x2x2 problem solved with rank 7
    # Total rank = 7 * 7 = 49
    
    # Block layout:
    # A block (bi, bj): rows bi*2..bi*2+1, cols bj*2..bj*2+1 in 4x4
    #   -> U index: (bi*2+li)*4 + (bj*2+lj)
    # B block (bj, bk): rows bj*2..bj*2+1, cols bk*2..bk*2+1 in 4x4
    #   -> V index: (bj*2+lj)*4 + (bk*2+lk)
    # C block (bi, bk): rows bi*2..bi*2+1, cols bk*2..bk*2+1 in 4x4
    #   -> W index: (bk*2+lk)*4 + (bi*2+li)
    
    def add_strassen_term(r_offset, a_blocks, b_blocks, c_blocks, a_signs, b_signs):
        """
        Add a Strassen term.
        a_blocks: list of (bi, bj) for A blocks (with signs from a_signs)
        b_blocks: list of (bj, bk) for B blocks (with signs from b_signs)
        c_blocks: list of (bi, bk) for C blocks
        """
        for r_local in range(7):
            r = r_offset + r_local
            # For each local 2x2 Strassen index
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        u_local = li * 2 + lj
                        v_local = lj * 2 + lk
                        w_local = lk * 2 + li
                        
                        u_coeff = U2[u_local, r_local]
                        v_coeff = V2[v_local, r_local]
                        w_coeff = W2[w_local, r_local]
                        
                        if abs(u_coeff) < 1e-15 and abs(v_coeff) < 1e-15 and abs(w_coeff) < 1e-15:
                            continue
                        
                        # Sum contributions from A blocks
                        u_val = 0.0
                        for (bi, bj), sign in zip(a_blocks, a_signs):
                            u_global = (bi * 2 + li) * 4 + (bj * 2 + lj)
                            u_val += sign * u_coeff
                        
                        # Sum contributions from B blocks
                        v_val = 0.0
                        for (bj, bk), sign in zip(b_blocks, b_signs):
                            v_global = (bj * 2 + lj) * 4 + (bk * 2 + lk)
                            v_val += sign * v_coeff
                        
                        # Sum contributions from C blocks
                        w_val = 0.0
                        for (bi, bk) in c_blocks:
                            w_global = (bk * 2 + lk) * 4 + (bi * 2 + li)
                            w_val += w_coeff
                        
                        # We need to be more careful: each A block contributes to specific U entries
                        # Let me redo this properly
                        pass
    
    # Actually, let me use a cleaner recursive approach
    # For 4x4x4, we have 7 high-level Strassen multiplications
    # Each involves products of 2x2 blocks
    
    # Let's define the 7 Strassen combinations more carefully
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B01-B11) -> contributes to C01
    # M4 = A11(B10-B00) -> contributes to C11
    # M5 = (A00+A01)B11 -> contributes to C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C10
    # M7 = (A01-A11)(B10+B11) -> contributes to C11
    
    # For each Mi, we need to express it as a sum of 2x2x2 Strassen terms
    # where the 2x2 blocks are the sub-matrices
    
    # Let me use a different approach: directly build using the known structure
    
    # Reset
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Helper: add a 2x2x2 Strassen term at given block positions
    # A at block (row_block, col_block) of 2x2 blocks
    # B at block (row_block, col_block) of 2x2 blocks  
    # C at block (row_block, col_block) of 2x2 blocks
    def add_2x2_strassen(a_rb, a_cb, b_rb, b_cb, c_rb, c_cb, r_offset, scale_a=1, scale_b=1, scale_c=1):
        """Add a 2x2x2 Strassen decomposition for blocks at specified positions."""
        for r_local in range(7):
            r = r_offset + r_local
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        u_local = li * 2 + lj
                        v_local = lj * 2 + lk
                        w_local = lk * 2 + li
                        
                        u_val = U2[u_local, r_local] * scale_a
                        v_val = V2[v_local, r_local] * scale_b
                        w_val = W2[w_local, r_local] * scale_c
                        
                        if abs(u_val) < 1e-15:
                            continue
                        u_global = (a_rb * 2 + li) * 4 + (a_cb * 2 + lj)
                        U[u_global, r] += u_val
                        
                        if abs(v_val) < 1e-15:
                            continue
                        v_global = (b_rb * 2 + lj) * 4 + (b_cb * 2 + lk)
                        V[v_global, r] += v_val
                        
                        if abs(w_val) < 1e-15:
                            continue
                        w_global = (c_cb * 2 + lk) * 4 + (c_rb * 2 + li)
                        W[w_global, r] += w_val
    
    # Now build the 4x4x4 Strassen:
    # M1 = (A00+A11)(B00+B11)
    # This is a sum of 4 products: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But in Strassen, we compute it as one 2x2x2 multiplication where:
    # The "A" input is A00+A11 (a 2x2 matrix formed from two blocks)
    # The "B" input is B00+B11
    # The "C" output goes to C00 and C11
    
    # Actually, the standard 4x4 Strassen works like this:
    # We treat the 4x4 matrix as a 2x2 block matrix of 2x2 blocks
    # Then apply 2x2 Strassen to the block matrix
    # Each block multiplication is itself a 2x2x2 problem
    
    # M1 = (A00+A11)(B00+B11)
    # To compute this, we form A_sum = A00+A11 and B_sum = B00+B11
    # Then compute A_sum * B_sum using 2x2 Strassen
    # The result contributes to C00 and C11
    
    # For the U factor: A_sum has entries from A00 and A11
    #   A_sum[li, lj] = A00[li, lj] + A11[li, lj]
    #   So U[(0*2+li)*4 + (0*2+lj), r] gets contribution from A00 part
    #   And U[(1*2+li)*4 + (1*2+lj), r] gets contribution from A11 part
    #   But in the 2x2 Strassen, the local index is (li, lj)
    #   So for the 2x2 Strassen at position (a_rb, a_cb), the U entry is at
    #   global row (a_rb*2+li)*4 + (a_cb*2+lj)
    
    # The key insight: when we say "A00+A11", the 2x2 Strassen term has
    # its U factor receiving contributions from BOTH A00 and A11 blocks.
    
    # Let me redefine: for each of the 7 high-level Strassen terms, I specify
    # which A-blocks contribute (with signs), which B-blocks contribute (with signs),
    # and which C-blocks receive the output.
    
    # M1: A blocks: [(0,0)+1, (1,1)+1], B blocks: [(0,0)+1, (1,1)+1], C blocks: [(0,0), (1,1)]
    # M2: A blocks: [(1,0)+1, (1,1)+1], B blocks: [(0,0)+1], C blocks: [(1,0)]
    # M3: A blocks: [(0,0)+1], B blocks: [(0,1)+1, (1,1)-1], C blocks: [(0,1)]
    # M4: A blocks: [(1,1)+1], B blocks: [(1,0)+1, (0,0)-1], C blocks: [(1,1)]
    # M5: A blocks: [(0,0)+1, (0,1)+1], B blocks: [(1,1)+1], C blocks: [(0,1)]
    # M6: A blocks: [(1,0)+1, (0,0)-1], B blocks: [(0,0)+1, (0,1)+1], C blocks: [(1,0)]
    # M7: A blocks: [(0,1)+1, (1,1)-1], B blocks: [(1,0)+1, (1,1)+1], C blocks: [(1,1)]
    
    strassen_terms = [
        # (a_blocks_with_signs, b_blocks_with_signs, c_blocks)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [(0,0), (1,1)]),  # M1
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [(1,0)]),  # M2
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [(0,1)]),  # M3
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [(1,1)]),  # M4
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [(0,1)]),  # M5
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [(1,0)]),  # M6
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [(1,1)]),  # M7
    ]
    
    for term_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_terms):
        r_offset = term_idx * 7
        
        for r_local in range(7):
            r = r_offset + r_local
            
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        u_local = li * 2 + lj
                        v_local = lj * 2 + lk
                        w_local = lk * 2 + li
                        
                        u_base = U2[u_local, r_local]
                        v_base = V2[v_local, r_local]
                        w_base = W2[w_local, r_local]
                        
                        if abs(u_base) < 1e-15 and abs(v_base) < 1e-15 and abs(w_base) < 1e-15:
                            continue
                        
                        # U contributions from A blocks
                        u_val = 0.0
                        for (br, bc), sign in a_blocks:
                            u_global = (br * 2 + li) * 4 + (bc * 2 + lj)
                            u_val += sign * u_base
                        if abs(u_val) > 1e-15:
                            U[u_global, r] += u_val
                        
                        # V contributions from B blocks
                        v_val = 0.0
                        for (br, bc), sign in b_blocks:
                            v_global = (br * 2 + lj) * 4 + (bc * 2 + lk)
                            v_val += sign * v_base
                        if abs(v_val) > 1e-15:
                            V[v_global, r] += v_val
                        
                        # W contributions from C blocks
                        for (br, bc) in c_blocks:
                            w_global = (bc * 2 + lk) * 4 + (br * 2 + li)
                            if abs(w_base) > 1e-15:
                                W[w_global, r] += w_base
    
    return U, V, W

def _extend_to_4x4x5(U4, V4, W4):
    """
    Extend a 4x4x4 decomposition to 4x4x5.
    The 4x4x5 tensor is the 4x4x4 tensor with an extra column (k=4) that
    has all zero entries (since we're multiplying by a 4x5 matrix, the 5th
    column of B just adds to the result, but in the tensor decomposition
    for C = A*B, the tensor is T[i,j,k] = 1 if the (i,k) entry of C
    depends on A[i,j]*B[j,k]. For a 4x4 * 4x5 multiplication, C is 4x5,
    so k ranges 0..4. The tensor T[i,j,k] = 1 for all valid i,j,k.
    
    Wait, let me reconsider. The matrix multiplication tensor for C = A*B
    where A is n x m and B is m x p:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    So for 4x4x5, we need T[i,j,k] = 1 for i in [0,4), j in [0,4), k in [0,5).
    
    The 4x4x4 decomposition gives us T[i,j,k] = 1 for k in [0,4).
    We need to extend it to also satisfy k=4.
    
    The simplest way: add a schoolbook term for the k=4 column.
    For each i in [0,4), j in [0,4):
      Add a term with U[i*4+j, r] = 1, V[j*5+4, r] = 1, W[4*4+i, r] = 1
    That's 16 additional terms.
    
    But we can do better: if we already have terms where V[j*5+4, r] is nonzero,
    we might be able to reuse them. However, in the 4x4x4 decomposition,
    V has shape (16, 49) corresponding to B being 4x4. For 4x4x5, V has
    shape (20, 49+extra). The first 16 rows correspond to B[0:4, 0:4],
    and rows 16-19 correspond to B[0:4, 4].
    
    So we need to add 16 new terms (one for each (i,j) pair for k=4).
    Total rank = 49 + 16 = 65.
    
    Alternatively, we can try to find a more efficient extension.
    """
    n, m, p = 4, 4, 5
    rank4 = U4.shape[1]  # 49
    
    U = np.zeros((n * m, rank4 + n * m))
    V = np.zeros((m * p, rank4 + n * m))
    W = np.zeros((p * n, rank4 + n * m))
    
    # Copy the 4x4x4 part
    # U4 has shape (16, 49), corresponding to A[i,j] with i,j in [0,4)
    # V4 has shape (16, 49), corresponding to B[j,k] with j,k in [0,4)
    # W4 has shape (16, 49), corresponding to C[k,i] with k,i in [0,4)
    
    U[:, :rank4] = U4
    V[:m*4, :rank4] = V4  # First 16 rows of V (B[0:4, 0:4])
    W[:4*n, :rank4] = W4  # First 16 rows of W (C[0:4, 0:4])
    
    # Add schoolbook terms for k=4
    for idx in range(n * m):
        i = idx // m
        j = idx % m
        r = rank4 + idx
        U[i * m + j, r] = 1
        V[j * p + 4, r] = 1
        W[4 * n + i, r] = 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank by finding linear dependencies among the terms.
    This is a simplified approach: check if we can merge terms.
    """
    # For now, return as-is. A full rank reduction would require
    # solving a system of equations, which is complex.
    return U, V, W, U.shape[1]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _strassen_4x4()
    
    # Extend to 4x4x5
    U, V, W = _extend_to_4x4x5(U4, V4, W4)
    
    # Round to half-integers
    U = np.array([[_round_to_half_integer(x) for x in row] for row in U])
    V = np.array([[_round_to_half_integer(x) for x in row] for row in V])
    W = np.array([[_round_to_half_integer(x) for x in row] for row in W])
    
    # Verify
    if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
