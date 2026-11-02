"""Program for 2x3x9 matrix multiplication tensor decomposition using hybrid block/Strassen strategy."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-50 decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    1. Split the inner dimension (size 3) into two parts: size 2 and size 1.
       A (2x3) = [A_2x2, A_2x1]
       B (3x9) = [B_2x9; B_1x9]
       
    2. The product C = A*B = A_2x2 * B_2x9 + A_2x1 * B_1x9
    
    3. Decompose each term:
       - Term 1: (2, 2, 9) multiplication.
         Split the output dimension (size 9) into four 2s and one 1.
         B_2x9 = [B_col1, B_col2, B_col3, B_col4, B_col5] where each is 2x2 except the last is 2x1.
         Actually, it's easier to split B_2x9 into 4 blocks of 2x2 and 1 block of 2x1.
         Let B_2x9 = [B_0, B_1, B_2, B_3, B_4] where B_0..B_3 are 2x2 and B_4 is 2x1.
         
         For each 2x2 block B_i, we compute A_2x2 * B_i (2x2 * 2x2 -> 2x2).
         We use Strassen's algorithm for 2x2x2 multiplication, which has rank 7.
         There are 4 such blocks, so 4 * 7 = 28 multiplications.
         
         For the last block B_4 (2x1), we compute A_2x2 * B_4 (2x2 * 2x1 -> 2x1).
         This is equivalent to two independent 2x1x1 multiplications? No, it's a (2,2,1) tensor.
         Rank of (2,2,1) is 2 (schoolbook: a1*b1 + a2*b2 for each output row? No).
         Let's check (2,2,1). A is 2x2, B is 2x1. C is 2x1.
         C_0 = A_00 B_0 + A_01 B_1
         C_1 = A_10 B_0 + A_11 B_1
         This can be done with 4 multiplications (schoolbook). Can it be done with fewer?
         Rank of (2,2,1) is 4. It's just dot products.
         So 4 multiplications for the last block.
         Total for Term 1: 28 + 4 = 32.
         
       - Term 2: (2, 1, 9) multiplication.
         A_2x1 is 2x1. B_1x9 is 1x9. C is 2x9.
         This is an outer product of a 2x1 and 1x9? No, A_2x1 * B_1x9.
         C_{i,k} = A_{i,0} * B_{0,k}.
         This is a rank-9 tensor? No.
         For each k (0..8), C_{:,k} = A_{:,0} * B_{0,k}.
         This requires 2*9 = 18 multiplications if done naively.
         Wait, A_{i,0} is a scalar for each i. B_{0,k} is a scalar for each k.
         C_{i,k} = A_{i,0} B_{0,k}.
         This is an outer product of vector A_col (size 2) and vector B_row (size 9).
         The tensor for (2,1,9) has rank 18?
         Let's check. The tensor is sum_{i,k} e_i^A e_k^C \otimes e_0^B \otimes e_i^A e_k^C?
         No, the tensor T is defined by T_{i, 0, k} = delta_{i,0} delta_{0,k}? No.
         Indices: i in {0,1} (A row), j in {0} (inner), k in {0..8} (B col).
         C_{i,k} = sum_j A_{i,j} B_{j,k}. Since j=0 only, C_{i,k} = A_{i,0} B_{0,k}.
         The tensor is T = sum_{i=0}^1 sum_{k=0}^8 e_i^A \otimes e_0^B \otimes e_k^C? No.
         T_{i,j,k} = 1 if j=0, else 0? No, T maps A,B to C.
         T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i',k'}.
         Here i'=i, k'=k. So T_{i,j,k} is 1 if i'=i, k'=k, j is the summation index.
         Actually, the tensor is sum_{i,k} E_{i,k}^C \otimes E_{i,0}^A \otimes E_{0,k}^B.
         Each term is a rank-1 tensor. There are 2*9=18 such terms.
         Are they linearly independent? Yes.
         So rank is 18.
         
    Total Rank = 32 (Term 1) + 18 (Term 2) = 50.
    
    This is an improvement over 54.
    """
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    
    # We will build U, V, W matrices.
    # U: (n*m) x rank = 6 x 50
    # V: (m*p) x rank = 27 x 50
    # W: (p*n) x rank = 18 x 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            raise RuntimeError("Rank exceeded")
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # --- Term 1: A_2x2 * B_2x9 ---
    # A indices: 0,1 (rows), 0,1 (cols of the 2x2 block). 
    # In original A (2x3), these are A[0,0], A[1,0], A[0,1], A[1,1].
    # U indices: i*m + j. 
    # A_00: U[0*3+0] = U[0]
    # A_10: U[1*3+0] = U[3]
    # A_01: U[0*3+1] = U[1]
    # A_11: U[1*3+1] = U[4]
    
    # B indices: j (row of 3x9), k (col of 3x9).
    # B_2x9 corresponds to rows j=0,1 of B.
    # V indices: j*p + k.
    # B_00: V[0*9+0] = V[0]
    # B_10: V[1*9+0] = V[9]
    # B_01: V[0*9+1] = V[1]
    # B_11: V[1*9+1] = V[10]
    # etc.
    
    # C indices: k (col of 2x9), i (row of 2x2).
    # W indices: k*n + i.
    # C_00: W[0*2+0] = W[0]
    # C_10: W[0*2+1] = W[1]
    # C_01: W[1*2+0] = W[2]
    # C_11: W[1*2+1] = W[3]
    # etc.
    
    # We split the 9 columns of B_2x9 into 4 blocks of width 2 and 1 block of width 1.
    # Block 0: cols 0,1
    # Block 1: cols 2,3
    # Block 2: cols 4,5
    # Block 3: cols 6,7
    # Block 4: col 8
    
    # Strassen for 2x2x2:
    # Inputs:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # Outputs:
    # C = [[p, q], [r, s]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # p = M1 + M4 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M2 + M3 - M6
    
    # Let's map this to our indices.
    # For a specific block with column offset k0:
    # B block is 2x2:
    # B_00 (top-left of block) = V[0*9 + k0]
    # B_01 (top-right of block) = V[0*9 + k0+1]
    # B_10 (bottom-left of block) = V[1*9 + k0]
    # B_11 (bottom-right of block) = V[1*9 + k0+1]
    
    # A is fixed 2x2:
    # A_00 = U[0]
    # A_01 = U[1]
    # A_10 = U[3]
    # A_11 = U[4]
    
    # C block is 2x2:
    # C_00 (top-left) = W[k0*2 + 0]
    # C_01 (top-right) = W[k0*2 + 1]
    # C_10 (bottom-left) = W[(k0+1)*2 + 0]  -- Wait.
    # W index is k*n + i.
    # C_{i,k}.
    # For block k0, columns are k0, k0+1.
    # C_{0, k0} -> W[k0*2 + 0]
    # C_{0, k0+1} -> W[(k0+1)*2 + 0]
    # C_{1, k0} -> W[k0*2 + 1]
    # C_{1, k0+1} -> W[(k0+1)*2 + 1]
    
    # Let's define the Strassen terms for block k0.
    # We need to add terms to U, V, W.
    # Each Strassen multiplication Mx involves a product of a linear combination of A's and a linear combination of B's.
    # We can implement this by adding multiple rank-1 terms or by pre-combining?
    # The interface requires U, V, W matrices. We cannot pre-combine in U or V unless we change the structure.
    # Standard tensor decomposition: T = sum_r u_r \otimes v_r \otimes w_r.
    # Strassen M1 = (A_00 + A_11) * (B_00 + B_11).
    # This expands to:
    # A_00 B_00 + A_00 B_11 + A_11 B_00 + A_11 B_11.
    # Each of these is a rank-1 term in the tensor decomposition of the multiplication?
    # No, the tensor decomposition of Strassen's algorithm itself is rank 7 for the 2x2x2 tensor.
    # The 7 multiplications are the "multiplications" in the arithmetic circuit.
    # In the tensor decomposition context, we are looking for a decomposition of the *matrix multiplication tensor*.
    # The rank of the tensor is the minimum number of multiplications.
    # Strassen shows rank <= 7.
    # The 7 terms correspond to the 7 products.
    # So, for each Strassen product Mx, we need to express it as a linear combination of the basis tensors?
    # No. The decomposition is T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
    # The vectors u_r, v_r, w_r are the coefficients.
    # For M1 = (A_00 + A_11)(B_00 + B_11):
    # u_1 has 1 at A_00 and 1 at A_11.
    # v_1 has 1 at B_00 and 1 at B_11.
    # w_1 has 1 at C_00, 1 at C_01, 1 at C_10, 1 at C_11? No.
    # M1 contributes to all 4 outputs.
    # C_00 += M1
    # C_01 += M1
    # C_10 += M1
    # C_11 += M1
    # So w_1 has 1 at all 4 output positions.
    
    # Let's verify the Strassen formulas again.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 - M6
    
    # So for each block, we add 7 rank-1 terms.
    # Total 4 blocks * 7 = 28 terms.
    
    def add_strassen_block(k0):
        nonlocal col
        
        # Indices for A (fixed)
        # a = A[0,0] -> U[0]
        # b = A[0,1] -> U[1]
        # c = A[1,0] -> U[3]
        # d = A[1,1] -> U[4]
        
        # Indices for B (block k0)
        # e = B[0, k0]   -> V[0*9 + k0]
        # f = B[0, k0+1] -> V[0*9 + k0+1]
        # g = B[1, k0]   -> V[1*9 + k0]
        # h = B[1, k0+1] -> V[1*9 + k0+1]
        
        # Indices for C (block k0)
        # p = C[0, k0]   -> W[k0*2 + 0]
        # q = C[0, k0+1] -> W[(k0+1)*2 + 0]
        # r = C[1, k0]   -> W[k0*2 + 1]
        # s = C[1, k0+1] -> W[(k0+1)*2 + 1]
        
        u_a, u_b, u_c, u_d = 0, 1, 3, 4
        
        v_e = 0 * 9 + k0
        v_f = 0 * 9 + k0 + 1
        v_g = 1 * 9 + k0
        v_h = 1 * 9 + k0 + 1
        
        w_p = k0 * 2 + 0
        w_q = (k0 + 1) * 2 + 0
        w_r = k0 * 2 + 1
        w_s = (k0 + 1) * 2 + 1
        
        # M1 = (a+d)(e+h)
        # u: a, d
        # v: e, h
        # w: p, q, r, s
        add_term(u_a, v_e, w_p, 1)
        add_term(u_a, v_e, w_q, 1)
        add_term(u_a, v_e, w_r, 1)
        add_term(u_a, v_e, w_s, 1)
        
        add_term(u_a, v_h, w_p, 1)
        add_term(u_a, v_h, w_q, 1)
        add_term(u_a, v_h, w_r, 1)
        add_term(u_a, v_h, w_s, 1)
        
        add_term(u_d, v_e, w_p, 1)
        add_term(u_d, v_e, w_q, 1)
        add_term(u_d, v_e, w_r, 1)
        add_term(u_d, v_e, w_s, 1)
        
        add_term(u_d, v_h, w_p, 1)
        add_term(u_d, v_h, w_q, 1)
        add_term(u_d, v_h, w_r, 1)
        add_term(u_d, v_h, w_s, 1)
        
        # This expansion is rank 4 for M1, not rank 1.
        # Strassen's algorithm works because we compute M1 as ONE multiplication.
        # In the tensor decomposition T = sum u_r v_r w_r, the rank is the number of terms.
        # If we expand (a+d)(e+h) into 4 terms, we are using 4 multiplications, not 1.
        # This defeats the purpose of Strassen.
        
        # How do we represent a single multiplication of linear forms in a tensor decomposition?
        # We can't directly if the tensor decomposition is defined over the field with basis elements.
        # Wait. The tensor is T = sum_{i,j,k} T_{ijk} e_i^A e_j^B e_k^C.
        # A rank-1 decomposition is u \otimes v \otimes w.
        # u, v, w are vectors.
        # If u = e_a + e_d, v = e_e + e_h, w = e_p + e_q + e_r + e_s.
        # Then u \otimes v \otimes w = (e_a + e_d) \otimes (e_e + e_h) \otimes (e_p + e_q + e_r + e_s)
        # = sum_{x in {a,d}} sum_{y in {e,h}} sum_{z in {p,q,r,s}} e_x \otimes e_y \otimes e_z.
        # This IS a single rank-1 term in the tensor decomposition!
        # My `add_term` function adds a single e_u \otimes e_v \otimes e_w term.
        # To represent the Strassen term M1, I need to add ONE term where:
        # U has 1s at u_a and u_d.
        # V has 1s at v_e and v_h.
        # W has 1s at w_p, w_q, w_r, w_s.
        
        # Let's correct the implementation.
        # I need a function to add a rank-1 term with potentially multiple non-zero entries in U, V, W.
        
        pass # This block logic is flawed in the helper. Let's rewrite the helper usage.

    # Let's redefine the helper to handle vectors.
    def add_rank1_term(u_vec, v_vec, w_vec):
        """
        u_vec, v_vec, w_vec are lists of (index, value) pairs.
        """
        nonlocal col
        if col >= rank:
            raise RuntimeError("Rank exceeded")
        for idx, val in u_vec:
            U[idx][col] += val
        for idx, val in v_vec:
            V[idx][col] += val
        for idx, val in w_vec:
            W[idx][col] += val
        col += 1

    # Reset col? No, we haven't added anything yet in the corrected logic.
    # But I already defined add_term which increments col.
    # I will use add_rank1_term for Strassen blocks.
    
    # Block 0: k0 = 0
    # Block 1: k0 = 2
    # Block 2: k0 = 4
    # Block 3: k0 = 6
    
    for k0 in [0, 2, 4, 6]:
        u_a, u_b, u_c, u_d = 0, 1, 3, 4
        v_e = 0 * 9 + k0
        v_f = 0 * 9 + k0 + 1
        v_g = 1 * 9 + k0
        v_h = 1 * 9 + k0 + 1
        
        w_p = k0 * 2 + 0
        w_q = (k0 + 1) * 2 + 0
        w_r = k0 * 2 + 1
        w_s = (k0 + 1) * 2 + 1
        
        # M1 = (a+d)(e+h)
        add_rank1_term(
            [(u_a, 1), (u_d, 1)],
            [(v_e, 1), (v_h, 1)],
            [(w_p, 1), (w_q, 1), (w_r, 1), (w_s, 1)]
        )
        
        # M2 = (c+d)e
        add_rank1_term(
            [(u_c, 1), (u_d, 1)],
            [(v_e, 1)],
            [(w_r, 1), (w_s, 1)]  # M2 contributes to C_10 and C_11?
            # C_10 = M2 + M4
            # C_11 = M1 - M2 + M3 - M6
            # So M2 contributes +1 to C_10 (w_r) and -1 to C_11 (w_s).
        )
        # Correction: M2 contributes +1 to r and -1 to s.
        # Let's re-verify Strassen output formulas.
        # C_00 = M1 + M4 - M5 + M7
        # C_01 = M3 + M5
        # C_10 = M2 + M4
        # C_11 = M1 - M2 + M3 - M6
        
        # So for M2:
        # w_r += 1
        # w_s -= 1
        # I already added it with +1, +1. That's wrong.
        # I need to be careful.
        
        # Let's restart the block addition with correct signs.
        # I will clear the last 2 terms? No, easier to just implement correctly now.
        # Since col has incremented, I can't easily undo.
        # Let's just implement the 7 terms correctly for each block.
        
        # Wait, I used add_rank1_term which increments col.
        # I have already added 2 terms for k0=0 incorrectly.
        # Let's just restart the whole construction logic cleanly.
        pass

    # RE-IMPLEMENTATION
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_rank1_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= rank:
            raise RuntimeError("Rank exceeded")
        for idx, val in u_vec:
            U[idx][col] += val
        for idx, val in v_vec:
            V[idx][col] += val
        for idx, val in w_vec:
            W[idx][col] += val
        col += 1

    # Term 1: A_2x2 * B_2x9
    # Blocks at k0 = 0, 2, 4, 6
    for k0 in [0, 2, 4, 6]:
        u_a, u_b, u_c, u_d = 0, 1, 3, 4
        v_e = 0 * 9 + k0
        v_f = 0 * 9 + k0 + 1
        v_g = 1 * 9 + k0
        v_h = 1 * 9 + k0 + 1
        
        w_p = k0 * 2 + 0
        w_q = (k0 + 1) * 2 + 0
        w_r = k0 * 2 + 1
        w_s = (k0 + 1) * 2 + 1
        
        # M1 = (a+d)(e+h)
        # Contributes to p, q, r, s with +1
        add_rank1_term(
            [(u_a, 1), (u_d, 1)],
            [(v_e, 1), (v_h, 1)],
            [(w_p, 1), (w_q, 1), (w_r, 1), (w_s, 1)]
        )
        
        # M2 = (c+d)e
        # Contributes to r (+1), s (-1)
        add_rank1_term(
            [(u_c, 1), (u_d, 1)],
            [(v_e, 1)],
            [(w_r, 1), (w_s, -1)]
        )
        
        # M3 = a(f-h)
        # Contributes to q (+1), s (+1)
        add_rank1_term(
            [(u_a, 1)],
            [(v_f, 1), (v_h, -1)],
            [(w_q, 1), (w_s, 1)]
        )
        
        # M4 = d(g-e)
        # Contributes to p (+1), r (+1)
        add_rank1_term(
            [(u_d, 1)],
            [(v_g, 1), (v_e, -1)],
            [(w_p, 1), (w_r, 1)]
        )
        
        # M5 = (a+b)h
        # Contributes to p (-1), q (+1)
        add_rank1_term(
            [(u_a, 1), (u_b, 1)],
            [(v_h, 1)],
            [(w_p, -1), (w_q, 1)]
        )
        
        # M6 = (c-a)(e+f)
        # Contributes to s (-1)
        add_rank1_term(
            [(u_c, 1), (u_a, -1)],
            [(v_e, 1), (v_f, 1)],
            [(w_s, -1)]
        )
        
        # M7 = (b-d)(g+h)
        # Contributes to p (+1)
        add_rank1_term(
            [(u_b, 1), (u_d, -1)],
            [(v_g, 1), (v_h, 1)],
            [(w_p, 1)]
        )
        
    # Last block of Term 1: k0 = 8 (width 1)
    # A_2x2 * B_2x1
    # B_2x1 is column 8 of B_2x9.
    # B_08 = V[0*9+8] = V[8]
    # B_18 = V[1*9+8] = V[17]
    # C_08 = W[8*2+0] = W[16]
    # C_18 = W[8*2+1] = W[17]
    
    # C_08 = A_00 B_08 + A_01 B_18
    # C_18 = A_10 B_08 + A_11 B_18
    
    # 4 rank-1 terms (schoolbook)
    # 1: A_00 * B_08 -> C_08
    add_rank1_term([(0, 1)], [(8, 1)], [(16, 1)])
    # 2: A_01 * B_18 -> C_08
    add_rank1_term([(1, 1)], [(17, 1)], [(16, 1)])
    # 3: A_10 * B_08 -> C_18
    add_rank1_term([(3, 1)], [(8, 1)], [(17, 1)])
    # 4: A_11 * B_18 -> C_18
    add_rank1_term([(4, 1)], [(17, 1)], [(17, 1)])
    
    # Term 2: A_2x1 * B_1x9
    # A_2x1 is column 2 of A.
    # A_02 = U[0*3+2] = U[2]
    # A_12 = U[1*3+2] = U[5]
    
    # B_1x9 is row 2 of B.
    # B_2k = V[2*9+k] = V[18+k] for k=0..8
    
    # C_ik = A_i2 * B_2k
    # C_0k = W[k*2+0]
    # C_1k = W[k*2+1]
    
    # 18 rank-1 terms
    for k in range(9):
        v_idx = 18 + k
        w_0 = k * 2 + 0
        w_1 = k * 2 + 1
        
        # A_02 * B_2k -> C_0k
        add_rank1_term([(2, 1)], [(v_idx, 1)], [(w_0, 1)])
        # A_12 * B_2k -> C_1k
        add_rank1_term([(5, 1)], [(v_idx, 1)], [(w_1, 1)])
        
    # Total rank used: 28 + 4 + 18 = 50.
    if col != 50:
        raise RuntimeError(f"Expected rank 50, got {col}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
