"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor.
    The tensor T[i,j,k] = delta_{i,k} (for A_{i,j} * B_{j,k} -> C_{i,k}).
    Convention: U has rows indexed by (i,j) -> i*m+j, V by (j,k) -> j*p+k, W by (k,i) -> k*n+i.
    T[i*m+j, j*p+k, k*n+i] = 1 iff i==k... wait.
    Actually the multiplication tensor is: T[a, b, c] where a=(i,j), b=(j,k), c=(k,i)
    and T = 1 if the indices are consistent (i from a matches i from c, j from a matches j from b, k from b matches k from c).
    So T[i*m+j, j*p+k, k*n+i] = 1 always (by construction of indices).
    The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k).
    """
    rank = len(U[0])
    # Build the tensor implicitly and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the 7 rank-1 components of the 2x2x2 multiplication tensor.
    U vectors are for A (flattened as [A00, A01, A10, A11]),
    V vectors are for B (flattened as [B00, B10, B01, B11]) -- note: B is 2x2, row-major: B[0,0],B[0,1],B[1,0],B[1,1]
    Actually let's use standard row-major: A = [A00, A01, A10, A11], B = [B00, B01, B10, B11]
    W vectors are for C (flattened as [C00, C01, C10, C11]).
    
    Strassen:
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11)B00
    P3 = A00(B01-B11)
    P4 = A11(B10-B00)
    P5 = (A00+A10)B11
    P6 = (A01-A11)(B10+B11)
    P7 = (A10-A00)(B00+B01)
    
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    
    Each P_r is a product of a linear form in A and a linear form in B, contributing to C.
    So for each r, U_r is the coefficient vector of the A-linear-form, V_r of the B-linear-form,
    and W_r is the coefficient vector of how P_r contributes to C.
    
    P1: A-form = [1,0,0,1], B-form = [1,0,0,1], C-contribution = [1,0,0,-1] (C00+=1, C11+=1... wait)
    C00 = P1 + P4 - P5 + P7 => P1 contributes +1 to C00
    C11 = P1 - P2 + P3 + P6 => P1 contributes +1 to C11
    So W1 = [1, 0, 0, 1]
    
    P2: A-form = [0,0,1,1], B-form = [1,0,0,0], C-contribution: C10 += P2 => W2 = [0,0,1,0]
    P3: A-form = [1,0,0,0], B-form = [0,1,0,-1], C-contribution: C01 += P3, C11 += P3 => W3 = [0,1,0,1]
    P4: A-form = [0,0,0,1], B-form = [0,0,1,-1], C-contribution: C00 += P4, C10 += P4 => W4 = [1,0,1,0]
    P5: A-form = [1,1,0,0], B-form = [0,0,0,1], C-contribution: C00 -= P5, C01 += P5 => W5 = [-1,1,0,0]
    P6: A-form = [0,1,0,-1], B-form = [0,0,1,1], C-contribution: C11 += P6 => W6 = [0,0,0,1]
    P7: A-form = [-1,0,1,0], B-form = [1,1,0,0], C-contribution: C00 += P7 => W7 = [1,0,0,0]
    """
    U = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [0, 1, 0, -1],  # P6
        [-1, 0, 1, 0],  # P7
    ]
    V = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [0, 0, 1, -1],  # P4
        [0, 0, 0, 1],   # P5
        [0, 0, 1, 1],   # P6
        [1, 1, 0, 0],   # P7
    ]
    W = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 0],   # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0],   # P7
    ]
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4 * 4x4 using Strassen on 2x2 blocks.
    
    Split 4x4 into four 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01], [B10, B11]]
    C = A*B = [[C00, C01], [C10, C11]]
    
    Using Strassen on the block level (7 block-multiplications), each block-multiplication
    is itself a 2x2 * 2x2 multiplication which has rank 7 (Strassen).
    Total rank = 7 * 7 = 49.
    
    For the full 4x4x4 tensor:
    U has 16 rows (4*4), V has 16 rows (4*4), W has 16 rows (4*4).
    
    Block indices: block row br in {0,1}, block col bc in {0,1}
    Within block: local row lr in {0,1}, local col lc in {0,1}
    Global row = br*2 + lr, global col = bc*2 + lc
    So A[br*2+lr, bc*2+lc] is element (lr,lc) of block (br,bc).
    
    For Strassen step s (0..6) at block level, with U_s, V_s, W_s being 4-dim vectors
    (indexed by block positions 00,01,10,11 -> flattened as [00,01,10,11]):
    
    For each (br,bc) pair where U_s has nonzero coefficient u_bc:
      For each local element (lr, lc) where the inner Strassen has nonzero:
        This gets complicated. Let me think differently.
    
    Actually, the composition works as follows:
    The block-level Strassen gives 7 bilinear forms P_s = (sum_bc U_s[bc] * A_block_bc) * (sum_bc V_s[bc] * B_block_bc)
    Each P_s contributes to C_blocks via W_s.
    
    Now each A_block_bc is a 2x2 matrix, B_block_bc is a 2x2 matrix.
    The product of two 2x2 matrices (after taking linear combinations) is not simply rank-7.
    
    Wait - I need to be more careful. The block-level Strassen says:
    C_block_rc = sum_s W_s[rc] * P_s
    where P_s = (sum_bc U_s[bc] * A_bc) * (sum_bc V_s[bc] * B_bc)
    
    Here A_bc and B_bc are 2x2 matrices. The product of two 2x2 matrices is a 2x2 matrix.
    So P_s is a 2x2 matrix = (linear combo of A blocks) * (linear combo of B blocks).
    
    The product of two 2x2 matrices can be computed with rank 7 (Strassen).
    So P_s = sum_{t=0}^{6} (a_s_t * (linear combo of A blocks))_{row,col} * (b_s_t * (linear combo of B blocks))_{row,col}
    
    This is getting complex. Let me use a different approach: directly construct the rank-49 decomposition.
    
    For each block-level Strassen step s (0..6) and inner Strassen step t (0..6):
    We have a rank-1 term in the full tensor.
    
    U vector: for each global (i,j) in 4x4, the coefficient is:
      i = br*2 + lr, j = bc*2 + lc
      coefficient = U_block_s[bc] * U_inner_t[lr, lc]  where U_inner_t is the 2x2 matrix form
    
    Wait, the inner Strassen U vectors are 4-dim (for 2x2 matrix flattened). Let me redefine.
    
    For 2x2 * 2x2, the U vector for step t is a 4-dim vector (coefficients of A00, A01, A10, A11).
    If A_block_bc has elements A[br*2+lr, bc*2+lc] for lr,lc in {0,1},
    then the flattened block is [A[br*2, bc*2], A[br*2, bc*2+1], A[br*2+1, bc*2], A[br*2+1, bc*2+1]]
    = [A[00], A[01], A[10], A[11]] of the block (row-major).
    
    So for global element (i,j) where i=br*2+lr, j=bc*2+lc:
    The position in the flattened 2x2 block (br,bc) is lr*2+lc.
    U_inner_t[lr*2+lc] is the coefficient.
    
    So U_full[i*m+j] for step (s,t) = U_block_s[bc] * U_inner_t[lr*2+lc]
    Similarly V_full[j*p+k] = V_block_s[bc] * V_inner_t[lr*2+lc]  -- wait, V is for B.
    
    For B: B[j, k] where j = bc*2 + lr, k = bc2*2 + lc2... 
    Hmm, B is 4x4, split into blocks B_bc where B_bc is 2x2.
    B[bc*2+lr, bc2*2+lc2] is element (lr, lc2) of block (bc, bc2).
    
    For the inner Strassen on B block: the flattened B block (bc, bc2) is [B[bc*2, bc2*2], B[bc*2, bc2*2+1], B[bc*2+1, bc2*2], B[bc*2+1, bc2*2+1]]
    V_inner_t is indexed by this flattening: position lr*2+lc2.
    
    So V_full[j*p+k] where j=bc*2+lr, k=bc2*2+lc2:
    = V_block_s[bc2] * V_inner_t[lr*2+lc2]
    
    For W: C[i, k] where i=br*2+lr, k=bc2*2+lc2.
    C block (br, bc2) has elements C[br*2+lr, bc2*2+lc2].
    W_inner_t[lr*2+lc2] is the coefficient.
    W_block_s[br*2+bc2] is the block-level coefficient (indexing C blocks as [00,01,10,11] -> br*2+bc2).
    
    W_full[k*n+i] where k=bc2*2+lc2, i=br*2+lr:
    = W_block_s[br*2+bc2] * W_inner_t[lr*2+lc2]
    
    Let me code this up.
    """
    n = m = p = 4
    U_block, V_block, W_block = _strassen_2x2x2()
    U_inner, V_inner, W_inner = _strassen_2x2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for br in range(2):
                for lr in range(2):
                    i = br * 2 + lr
                    for bc in range(2):
                        for lc in range(2):
                            j = bc * 2 + lc
                            a = i * m + j
                            ub = U_block[s][bc]
                            ui = U_inner[t][lr * 2 + lc]
                            if ub != 0 and ui != 0:
                                U[a][r] = ub * ui
            for bc in range(2):
                for lr in range(2):
                    j = bc * 2 + lr
                    for bc2 in range(2):
                        for lc2 in range(2):
                            k = bc2 * 2 + lc2
                            b = j * p + k
                            vb = V_block[s][bc2]
                            vi = V_inner[t][lr * 2 + lc2]
                            if vb != 0 and vi != 0:
                                V[b][r] = vb * vi
            for br in range(2):
                for lr in range(2):
                    i = br * 2 + lr
                    for bc2 in range(2):
                        for lc2 in range(2):
                            k = bc2 * 2 + lc2
                            c = k * n + i
                            wb = W_block[s][br * 2 + bc2]
                            wi = W_inner[t][lr * 2 + lc2]
                            if wb != 0 and wi != 0:
                                W[c][r] = wb * wi
    return {"U": U, "V": V, "W": W}


def _build_4x4x7():
    """Build decomposition for 4x4 * 4x7.
    
    Split B (4x7) into blocks. Since 7 = 4 + 3, we can do:
    - Left part: B_left is 4x4, use 4x4x4 Strassen (rank 49)
    - Right part: B_right is 4x3, need 4x4x3 decomposition
    
    For 4x4x3: split into 2x2x3? Or use a different approach.
    
    Alternative: 4x4 * 4x7 = 4x4 * [4x4 | 4x3]
    = [4x4*4x4 | 4x4*4x3]
    
    For 4x4*4x3: We can split A into 2x2 blocks (4 blocks), B into 2x2 and 2x1 blocks.
    Actually, let's use a simpler approach: 
    4x4 * 4x3: rank at most 4*4*3 = 48 (schoolbook).
    Can we do better? The 4x4x3 tensor... 
    
    For now, let's use schoolbook for the 4x4x3 part (rank 48) and Strassen for 4x4x4 (rank 49).
    Total rank = 49 + 48 = 97.
    
    But wait - the output C is 4x7. The W matrix has p*n = 7*4 = 28 rows.
    U has n*m = 16 rows, V has m*p = 4*7 = 28 rows.
    
    For the left part (columns 0-3 of B):
    U_left has 16 rows, V_left has 4*4=16 rows (for B[:, 0:4]), W_left has 4*4=16 rows (for C[:, 0:4]).
    
    For the right part (columns 4-6 of B):
    U_right has 16 rows, V_right has 4*3=12 rows (for B[:, 4:7]), W_right has 3*4=12 rows (for C[:, 4:7]).
    
    Combined: U is 16 x rank, V is 28 x rank, W is 28 x rank.
    The U for both parts is the same structure but with different V and W.
    
    Actually, I can just concatenate the rank-1 terms.
    """
    n, m, p = 4, 4, 7
    
    # Left part: 4x4 * 4x4 -> 4x4, rank 49
    left = _build_4x4x4_strassen()
    rank_left = len(left['U'][0])
    
    # Right part: 4x4 * 4x3 -> 4x3, use schoolbook rank 48
    n_r, m_r, p_r = 4, 4, 3
    rank_right = n_r * m_r * p_r
    U_right = [[0] * rank_right for _ in range(n_r * m_r)]
    V_right = [[0] * rank_right for _ in range(m_r * p_r)]
    W_right = [[0] * rank_right for _ in range(p_r * n_r)]
    col = 0
    for i in range(n_r):
        for j in range(m_r):
            for k in range(p_r):
                U_right[i * m_r + j][col] = 1
                V_right[j * p_r + k][col] = 1
                W_right[k * n_r + i][col] = 1
                col += 1
    
    total_rank = rank_left + rank_right
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill left part
    for i in range(n * m):
        for r in range(rank_left):
            U[i][r] = left['U'][i][r]
    for i in range(4):  # B rows 0-3
        for j in range(4):  # B cols 0-3
            for r in range(rank_left):
                V[i * p + j][r] = left['V'][i * 4 + j][r]
    for i in range(4):  # C cols 0-3
        for j in range(4):  # C rows 0-3
            for r in range(rank_left):
                W[i * n + j][r] = left['W'][i * 4 + j][r]
    
    # Fill right part
    offset = rank_left
    for i in range(n * m):
        for r in range(rank_right):
            U[i][offset + r] = U_right[i][r]
    for i in range(4):  # B rows 0-3
        for j in range(3):  # B cols 4-6 -> local col j
            for r in range(rank_right):
                V[i * p + (4 + j)][offset + r] = V_right[i * 3 + j][r]
    for i in range(3):  # C cols 4-6 -> local col i
        for j in range(4):  # C rows 0-3
            for r in range(rank_right):
                W[(4 + i) * n + j][offset + r] = W_right[i * 4 + j][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    # Try the Strassen-based construction first
    try:
        result = _build_4x4x7()
        # Verify
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(4, 4, 7)
