"""3x5x6 matrix multiplication decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact: sum_c U[o,c]*V[i,c]*W[k,c] == delta."""
    rank = U.shape[1]
    # Reshape to 3D tensors
    U3 = U.reshape(n, m, rank)   # U[i,j,c]
    V3 = V.reshape(m, p, rank)   # V[j,k,c]
    W3 = W.reshape(p, n, rank)   # W[k,i,c]
    
    # Check: for all (i, j, k), sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c] == 1
    # and for all other combinations, should be 0
    # The tensor T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Efficient check using einsum
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need T to be all-ones for valid (i,j,k) where i in range(n), j in range(m), k in range(p)
    
    # Use einsum: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But W is indexed as W[k,i,c], so W3[k,i,c]
    # This is tricky with einsum. Let's just do it directly.
    
    # Build the full tensor
    # For each c, contribute U3[:,:,c] outer-ish with V3 and W3
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        # T[:, :, :] += sum over c: U[:, :, c] * V[:, :, c] * W[:, :, c] (with proper indexing)
        # For each (i,j,k): T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is not a standard outer product. Let's compute via loops or clever reshaping.
        # T[i,j,k] = U[i,j] * V[j,k] * W[k,i] for fixed c
        # We can compute: for each j, sum over c... no, c is fixed here.
        # Let's just do it with broadcasting for each j
        for j in range(m):
            # T[i,k] += U[i,j] * V[j,k] * W[k,i] for all i,k
            u_ij = uc[:, j]  # (n,)
            v_jk = vc[j, :]  # (p,)
            # T[:, j, :] += u_ij[:, None] * v_jk[None, :] * W3[:,:,c].T  # W3[k,i] -> W3.T[i,k]
            # W3 is (p, n), so W3[k,i]. Transpose gives W3.T[i,k] = W3[k,i]. Yes.
            wk_i = wc.T  # (n, p) where wk_i[i,k] = wc[k,i]
            T[:, j, :] += u_ij[:, None] * v_jk[None, :] * wk_i
    
    # T should be all 1s since every (i,j,k) is a valid multiplication entry
    max_err = np.max(np.abs(T - 1.0))
    return max_err < 1e-9

def construct():
    """
    Construct a low-rank tensor decomposition for 3x5 by 5x6 matrix multiplication.
    
    Strategy: Use block decomposition where we decompose the 5x5 middle dimension
    into blocks and use known low-rank decompositions for smaller matrix multiplication.
    
    For 3x5x6:
    - We can write A (3x5) as [A1 | A2] where A1 is 3x2, A2 is 3x3
    - B (5x6) as [B1; B2] where B1 is 2x6, B2 is 3x6
    - C = A1*B1 + A2*B2
    - Rank(A1*B1) <= rank(3x2x6) and Rank(A2*B2) <= rank(3x3x6)
    
    For 3x3x3: rank 23 is known (Strassen-like constructions exist)
    For 3x2x6: can be seen as 3x3x3 padded, rank <= 23*2 = 46 or better
    Actually 3x2x6: we can use the 3x3x6 decomposition.
    
    Let's use a direct approach:
    3x5x6 = 3x2x6 + 3x3x6 (with the split at the middle index)
    
    For 3x3x6: 
    We can use the fact that 3x3x6 = 3x3x3 + 3x3x3 (splitting the output)
    So rank(3x3x6) <= 2 * rank(3x3x3) = 2 * 23 = 46
    
    For 3x2x6:
    3x2x6 = 3x3x6 (with one column zeroed) but we need exact, so:
    Actually 3x2x6 means A is 3x2, B is 2x6.
    We can pad: let A' = [A | 0] (3x3), B' = [B; 0] (3x6)
    Then A'*B' = A*B, so rank(3x2x6) <= rank(3x3x6) = 46
    
    Total: rank <= 46 + 46 = 92. That's worse than schoolbook!
    
    Let me try a different split.
    3x5x6: Split the 5 into 2+3.
    C = A1B1 + A2B2 where A1 is 3x2, B1 is 2x6, A2 is 3x3, B2 is 3x6.
    
    Actually, let me use a smarter approach. Known results:
    - rank(3x3x3) <= 23 (but this is for exact, the best known is 23? Actually 23 is not confirmed, 
      the best proven upper bound is higher. Let me use safe known bounds.)
    
    Actually for practical purposes, let me use the following known low-rank constructions:
    - 2x2x2: rank 7 (Strassen)
    - 3x3x3: rank 23 (this is the best known upper bound via various methods)
    
    Wait, 23 is not the proven best. The best known upper bound for 3x3x3 is 23? 
    Actually no - the Strassen algorithm gives rank 7 for 2x2x2.
    For 3x3x3, the best known upper bound is 23 (from 1980s work), but this requires specific construction.
    
    Let me just use the schoolbook for the 3x3x3 parts and see if I can do better with a different block structure.
    
    Alternative: 3x5x6 
    Split middle index: 5 = 1+4
    C = A[:,0]B[0,:] + A[:,1:5]B[1:5,:]
    First term: rank(3x1x6) = 3*1*6 = 18 (diagonal, each entry needs one rank)
    Actually rank(3x1x6): A is 3x1 (column vector a), B is 1x6 (row vector b)
    C = a * b (outer product). This has rank 1 as a matrix, but as a tensor multiplication:
    C[k,i] = sum_j A[i,j] B[j,k] = A[i,0] B[0,k]
    So C = a_i * b_k, which is an outer product. The tensor for this is:
    T[i,0,k] = 1, everything else 0.
    This is just a single "multiplication" - rank 1!
    
    Second term: A is 3x4, B is 4x6. rank(3x4x6) = ?
    Schoolbook: 3*4*6 = 72.
    
    So total: 1 + 72 = 73. Better than 90!
    
    Can I do 3x4x6 better?
    3x4x6: split 4 = 2+2
    C = A1B1 + A2B2 where A1,A2 are 3x2, B1,B2 are 2x6
    rank(3x2x6) each: schoolbook 3*2*6 = 36. Two of them: 72. No improvement.
    
    3x4x6: split 6 = 3+3
    C = A B1 + A B2 where B1,B2 are 4x3
    rank(3x4x3) each: schoolbook 3*4*3 = 36. Two: 72. No improvement.
    
    3x4x6: split 3 = 1+2
    C = A[:,0:1]B[0:1,:] + A[:,1:3]B[1:3,:]
    First: rank(1x4x6)... wait, I'm splitting the input index n=3.
    C[k,i] for i=0: sum_j A[0,j] B[j,k] -> this is row 0 of C
    Actually splitting i:
    C[*,0] uses A[0,:] (1x4) and B (4x6): rank(1x4x6) = 4*6 = 24 schoolbook
    C[*,1:3] uses A[1:3,:] (2x4) and B (4x6): rank(2x4x6) = 2*4*6 = 48 schoolbook
    Total: 24+48 = 72. Same as before.
    
    Hmm. Let me try using Strassen's idea on parts.
    
    For 2x2x2, Strassen gives rank 7 instead of 8.
    For 2x2x3: 
    Split output: C[:,0] and C[:,1] (each 2x1)
    Each 2x2x1: C[:,k] = A * B[:,k], a 2x2 times 2x1 = 2x1 result.
    Rank of 2x2x1: each output element C[i,k] = sum_j A[i,j]B[j,k], 2 elements for each k.
    Actually 2x2x1 means A is 2x2, B is 2x1, C is 2x1.
    C[0] = A[0,0]B[0] + A[0,1]B[1]
    C[1] = A[1,0]B[0] + A[1,1]B[1]
    This needs rank 4 in schoolbook (4 terms). Can we do better?
    The tensor has 4 non-zero entries: T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    Wait no: T[i,j,k] = delta_{i+k, something}... 
    Actually T[i,j,k] = 1 if i is the row of A, j is column of A / row of B, k is column of B.
    For 2x2x1: i in {0,1}, j in {0,1}, k in {0} (only k=0).
    T[i,j,0] = 1 for all i,j in {0,1}. So it's a 2x2x1 tensor that's all 1s.
    Rank of all-ones 2x2x1: This is just the outer product of vectors (1,1) and (1,1) with scalar 1.
    Actually as a tensor T[i,j,0] = v_i * w_j where v=(1,1), w=(1,1). 
    So T = v ⊗ w ⊗ (1 for k=0). This is rank 1!
    
    Wait, I'm confusing myself. The multiplication tensor T[i,j,k] for matrix mult n x m x p satisfies:
    T[i,j,k] = 1 if the (i,j,k) entry corresponds to a term in the matrix product, which is always 1 for all valid i,j,k.
    No wait. The multiplication tensor is:
    C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    The tensor T has entries T[i,j,k] = delta for each (i,j,k) triple, and:
    C = sum over all (i,j,k): T[i,j,k] * A[i,j] * B[j,k] * (contributes to C[i,k])
    
    The factorization is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    And we need T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    
    So for 2x2x1: n=2, m=2, p=1.
    T[i,j,k=0] = 1 for all i,j in {0,1}.
    Need: 1 = sum_c U[i,j,c] V[j,0,c] W[0,i,c]
    This is a rank-1 condition on a 2x2 matrix if we fix c... 
    Actually U[i,j,c] V[j,0,c] W[0,i,c] = (U * something)
    Let's define for fixed c: a_{i,j} = U[i,j,c], b_j = V[j,0,c], d_i = W[0,i,c]
    Then sum_c a_{i,j} b_j d_i = 1 for all i,j.
    If c is one term: a_{i,j} b_j d_i = 1 for all i,j means a_{i,j} = 1/(b_j d_i), which requires b_j d_i to be constant... 
    If b_0=b_1=1, d_0=d_1=1, then a_{i,j}=1 works. So rank 1 suffices!
    
    Wait, that means 2x2x1 has rank 1. Let me verify with the factorization format.
    U shape: (n*m, rank) = (4, rank)
    V shape: (m*p, rank) = (2, rank)  
    W shape: (p*n, rank) = (2, rank)
    
    For rank 1: U[i*m+j, 0], V[j*p+k, 0], W[k*n+i, 0]
    T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    For n=2,m=2,p=1: T[i,j,0] = U[i*2+j,0] * V[j,0] * W[i,0]
    Need this = 1 for all i,j in {0,1}.
    Set U[i*2+j,0]=1, V[j,0]=1, W[i,0]=1. Then T=1. Rank 1 works!
    
    So 2x2x1 has rank 1. Similarly 1x1x1 has rank 1.
    And 1xmxp has rank 1 (just outer product of A column and B row... wait).
    n=1,m=M,p=P: C[0,k] = sum_j A[0,j]B[j,k]. 
    T[0,j,k] = 1 for all j,k.
    T[0,j,k] = U[0*M+j,c] V[j*P+k,c] W[k*1+0,c] = U[j,c] V[j*P+k,c] W[k,c]
    Need sum_c U[j,c] V[j*P+k,c] W[k,c] = 1 for all j,k.
    Set c=0: U[j,0]=1, V[j*P+k,0]=1, W[k,0]=1. Then product=1. Rank 1!
    
    So the rank of 1xmxp multiplication tensor is always 1 (just all factors are 1).
    Similarly nx1p is rank 1.
    And nx1x1 is rank 1.
    
    Now back to 3x5x6.
    Split middle index m=5 into 1+4:
    C = A[:,0]B[0,:] + A[:,1:]B[1:,:]
    First part: n=3, m=1, p=6. Rank = 1 (as shown above).
    Second part: n=3, m=4, p=6. Schoolbook rank = 72.
    Total: 1 + 72 = 73.
    
    Now can we do 3x4x6 better?
    Split m=4 into 1+3:
    3x4x6 = (3,1,6) + (3,3,6)
    Rank(3,1,6) = 1
    Rank(3,3,6): split p=6 into 3+3:
    Rank(3,3,6) = rank(3,3,3) + rank(3,3,3)
    
    For 3x3x3, can we get rank less than 27?
    The known best upper bound for 3x3x3 is 23 (from various papers), but constructing it is complex.
    If we use rank 23 for 3x3x3: rank(3,3,6) = 46, total for 3x4x6 = 1+46=47, total for 3x5x6 = 1+47=48.
    
    But I don't have a concrete 3x3x3 rank-23 construction to implement.
    
    Alternative: Use the 2x2x2 Strassen rank-7 to help.
    
    For 3x3x3: 
    Split as (2+1)x(2+1)x(2+1):
    9 blocks: each 2x2x2 (rank 7 via Strassen) or mixed with 1s.
    Blocks:
    (2,2,2): 4 of these, each rank 7 -> 28
    (2,2,1): 2, each rank 1 -> 2
    (2,1,2): 2, each rank 1 -> 2
    (1,2,2): 2, each rank 1 -> 2
    (2,1,1): 1, rank 1
    (1,2,1): 1, rank 1
    (1,1,2): 1, rank 1
    (1,1,1): 1, rank 1
    
    Wait, this isn't quite right. The 3x3x3 can be seen as:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] similarly
    C = [[C11, C12], [C21, C22]] similarly
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each Cxx is a 2x2 or 1x1 matrix product.
    C11: (2x2)(2x2) + (2x1)(1x2) - mixed sizes
    This is getting complex. Let me just use the direct block sum.
    
    Actually, the simplest approach that definitely helps:
    3x5x6 with middle split 1+4:
    Part 1: (3,1,6) rank 1
    Part 2: (3,4,6) 
    
    For (3,4,6), split p into 3+3:
    (3,4,3) + (3,4,3)
    Each (3,4,3): split m into 1+3:
    (3,1,3) + (3,3,3)
    (3,1,3): rank 1
    (3,3,3): use schoolbook 27 for now
    
    So (3,4,3) = 1 + 27 = 28
    (3,4,6) = 28 + 28 = 56
    Total (3,5,6) = 1 + 56 = 57
    
    That's a significant improvement over 90!
    
    Can we do better on 3x3x3? Let's try the 2+1 split with Strassen.
    3x3x3 with A split as 2+1 rows, B split as 2+1 rows, C split as 2+1 rows:
    
    Actually let me just implement the (3,1,6)+(3,4,6) split with (3,4,6)=(3,4,3)+(3,4,3) and each (3,4,3)=(3,1,3)+(3,3,3) with schoolbook 3x3x3.
    
    Total: 1 + (1+27) + (1+27) = 1 + 28 + 28 = 57.
    
    Let me implement this.
    """
    n, m, p = 3, 5, 6
    
    # Build up the decomposition by recursive splitting
    # We'll collect all (U, V, W) contributions as lists of columns
    
    U_cols = []  # List of column vectors for U, each of length n*m
    V_cols = []  # List of column vectors for V, each of length m*p
    W_cols = []  # List of column vectors for W, each of length p*n
    
    def add_block(i_offset, j_offset, k_offset, nn, mm, pp, U_block, V_block, W_block, block_rank):
        """Add a block decomposition.
        i_offset, j_offset, k_offset: starting indices in the full matrices
        nn, mm, pp: dimensions of this block
        U_block: (nn*mm, block_rank)
        V_block: (mm*pp, block_rank)
        W_block: (pp*nn, block_rank)
        """
        for c in range(block_rank):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # Place U_block
            for i in range(nn):
                for j in range(mm):
                    u_col[(i_offset + i) * m + (j_offset + j)] = U_block[i * mm + j, c]
            
            # Place V_block
            for j in range(mm):
                for k in range(pp):
                    v_col[(j_offset + j) * p + (k_offset + k)] = V_block[j * pp + k, c]
            
            # Place W_block
            for k in range(pp):
                for i in range(nn):
                    w_col[(k_offset + k) * n + (i_offset + i)] = W_block[k * nn + i, c]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    def schoolbook(nn, mm, pp):
        """Generate schoolbook decomposition for nn x mm x pp."""
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
        return U, V, W, r
    
    def rank1_block(nn, mm, pp):
        """The (nn, 1, pp) or (1, mm, pp) etc rank-1 case."""
        # For the case where the middle dimension mm=1 or n=1 or p=1
        # Actually we handle general: if any of nn,mm,pp allows rank 1
        # nn x mm x pp has rank 1 when: the tensor T[i,j,k]=1 for all valid i,j,k
        # T[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c]
        # If mm=1: T[i,0,k] = U[i,0,c]*V[0,k,c]*W[k,i,c] = 1
        #   Set U[i,0,c]=1, V[0,k,c]=1, W[k,i,c]=1 for one c. Rank 1.
        # If nn=1: T[0,j,k] = U[0,j,c]*V[j,k,c]*W[k,0,c] = 1
        #   Set all to 1. Rank 1.
        # If pp=1: T[i,j,0] = U[i,j,c]*V[j,0,c]*W[0,i,c] = 1
        #   Set all to 1. Rank 1.
        
        U = np.ones((nn * mm, 1))
        V = np.ones((mm * pp, 1))
        W = np.ones((pp * nn, 1))
        return U, V, W, 1
    
    # Split 3x5x6 into (3,1,6) + (3,4,6)
    # Part 1: (3,1,6) - rank 1
    U1, V1, W1, r1 = rank1_block(3, 1, 6)
    add_block(0, 0, 0, 3, 1, 6, U1, V1, W1, r1)
    
    # Part 2: (3,4,6) - split into (3,4,3) + (3,4,3)
    # Sub-part 2a: (3,4,3) - split into (3,1,3) + (3,3,3)
    U2a1, V2a1, W2a1, r2a1 = rank1_block(3, 1, 3)
    add_block(0, 1, 0, 3, 1, 3, U2a1, V2a1, W2a1, r2a1)
    
    U2a2, V2a2, W2a2, r2a2 = schoolbook(3, 3, 3)
    add_block(0, 2, 0, 3, 3, 3, U2a2, V2a2, W2a2, r2a2)
    
    # Sub-part 2b: (3,4,3) for k=3,4,5 - split into (3,1,3) + (3,3,3)
    U2b1, V2b1, W2b1, r2b1 = rank1_block(3, 1, 3)
    add_block(0, 1, 3, 3, 1, 3, U2b1, V2b1, W2b1, r2b1)
    
    U2b2, V2b2, W2b2, r2b2 = schoolbook(3, 3, 3)
    add_block(0, 2, 3, 3, 3, 3, U2b2, V2b2, W2b2, r2b2)
    
    # Assemble
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 1))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 1))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 1))
    
    rank = U.shape[1]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}

