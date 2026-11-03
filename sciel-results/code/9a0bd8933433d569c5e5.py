"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _build_strassen_factors():
    """
    Build U, V, W factor matrices for 2x4 by 4x6 matrix multiplication
    using Strassen-style block decomposition.

    We split A (2x4) into two 2x2 blocks: A_L (cols 0,1), A_R (cols 2,3)
    We split B (4x6) into four 2x3 blocks: B_TL, B_TR, B_BL, B_BR
    Each block is further split into 2x2 and 2x3 sub-blocks as needed.

    Actually, let's think more carefully. A is 2x4, B is 4x6, C is 2x6.
    
    Split A into two 2x2 blocks: A_L (cols 0,1), A_R (cols 2,3)
    Split B into two 2x3 blocks: B_T (rows 0,1), B_B (rows 2,3)
    
    C = A_L * B_T + A_R * B_B
    
    Now each product is 2x2 * 2x3 = 2x3.
    
    For 2x2 * 2x3, we can split the 2x3 into two 2x2 and one 2x1, or use
    a direct decomposition. Let's use the approach of splitting B_T into
    B_T1 (cols 0,1) and B_T2 (cols 2) [2x1], and similarly for B_B.
    
    Actually, 2x2 * 2x2 has rank 7 (Strassen). 2x2 * 2x1 has rank 3 (or less).
    2x2 * 2x3 = 2x2 * (2x2 + 2x1). We can do 2x2*2x2 with rank 7, and 2x2*2x1 
    with some rank.
    
    For 2x2 * 2x1: A (2x2) * b (2x1) = [a11*b1 + a12*b2, a21*b1 + a22*b2]
    This is a 2x2x1 tensor. Its rank is 2 (or possibly less, but let's say 2).
    Actually, the tensor for 2x2*2x1: C_i = sum_j A_ij * b_j. 
    This is a bilinear map from R^4 x R^2 -> R^2. 
    The rank of this tensor... Let me think. C_0 = a11*b1 + a12*b2, C_1 = a21*b1 + a22*b2.
    We need factors U (4 x r), V (2 x r), W (2 x r) such that sum_c U[i,c]*V[j,c]*W[k,c] = tensor[i,j,k].
    Here i indexes A (4 elements), j indexes b (2 elements), k indexes C (2 elements).
    Tensor[i,j,k] is 1 if A_ij contributes to C_k.
    A has indices (a_row, a_col) where a_row in {0,1}, a_col in {0,1}, so i = a_row*2 + a_col.
    b has index j in {0,1}.
    C has index k in {0,1}.
    C_k = sum_{a_col} A_{k, a_col} * b_{a_col}.
    So Tensor[i, j, k] = 1 if i = k*2 + j, else 0.
    So the tensor is: for each (j, k), the only nonzero i is k*2 + j.
    This is essentially the "diagonal" tensor. Rank is 2: 
    U[k*2+j, c] = 1 for c corresponding to (j,k), V[j, c] = 1, W[k, c] = 1.
    So we need 2 factors: one for (j=0,k=0), one for (j=1,k=0), wait no.
    For (j=0, k=0): i = 0, V[0,0]=1, W[0,0]=1
    For (j=1, k=0): i = 1, V[1,0]=1, W[0,0]=1 -- wait, this doesn't work with same c.
    
    Let me reconsider. We need sum_c U[i,c] V[j,c] W[k,c] = delta_{i, k*2+j}.
    For i=0, j=0, k=0: sum_c U[0,c] V[0,c] W[0,c] = 1
    For i=1, j=1, k=0: sum_c U[1,c] V[1,c] W[0,c] = 1
    For i=2, j=0, k=1: sum_c U[2,c] V[0,c] W[1,c] = 1
    For i=3, j=1, k=1: sum_c U[3,c] V[1,c] W[1,c] = 1
    All other (i,j,k) should be 0.
    
    If we use r=4 factors:
    c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> contributes to (0,0,0)
    c=1: U[1,1]=1, V[1,1]=1, W[0,1]=1 -> contributes to (1,1,0)
    c=2: U[2,2]=1, V[0,2]=1, W[1,2]=1 -> contributes to (2,0,1)
    c=3: U[3,3]=1, V[1,3]=1, W[1,3]=1 -> contributes to (3,1,1)
    Check (0,0,0): only c=0 has all three nonzero, sum=1. Good.
    Check (1,1,0): only c=1, sum=1. Good.
    Check (0,1,0): c=0 has V[0,0]=1 but V[0,1]=0... wait V is (2 x r), V[j,c].
    (i=0, j=1, k=0): sum_c U[0,c] V[1,c] W[0,c]. 
    c=0: U[0,0]=1, V[1,0]=0, W[0,0]=1 -> 0
    c=1: U[0,1]=0, ... -> 0
    All zero. Good.
    So rank 4 for 2x2*2x1? That seems high. Let me check if rank 2 works.
    
    Actually, the tensor T[i,j,k] = delta_{i, 2k+j} for i in {0,1,2,3}, j in {0,1}, k in {0,1}.
    This is a 4x2x2 tensor. Let's see if rank 2 suffices.
    We need U (4x2), V (2x2), W (2x2) such that sum_c U[i,c]V[j,c]W[k,c] = T[i,j,k].
    For (0,0,0): U[0,0]V[0,0]W[0,0] + U[0,1]V[0,1]W[0,1] = 1
    For (1,1,0): U[1,0]V[1,0]W[0,0] + U[1,1]V[1,1]W[0,1] = 1
    For (2,0,1): U[2,0]V[0,0]W[1,0] + U[2,1]V[0,1]W[1,1] = 1
    For (3,1,1): U[3,0]V[1,0]W[1,0] + U[3,1]V[1,1]W[1,1] = 1
    For (0,1,0): U[0,0]V[1,0]W[0,0] + U[0,1]V[1,1]W[0,1] = 0
    For (1,0,0): U[1,0]V[0,0]W[0,0] + U[1,1]V[0,1]W[0,1] = 0
    For (2,1,1): U[2,0]V[1,0]W[1,0] + U[2,1]V[1,1]W[1,1] = 0
    For (3,0,1): U[3,0]V[0,0]W[1,0] + U[3,1]V[0,1]W[1,1] = 0
    
    This is a system of equations. It might have a solution. But let's just use rank 4 to be safe,
    or better, let me just use a known decomposition.
    
    Actually, let's step back. The total rank for 2x4 * 4x6 should be bounded.
    Schoolbook: 2*4*6 = 48.
    
    Let me use a different approach. Split into 2x2 blocks for A, and for B use the fact
    that it's 4x6. 
    
    A (2x4) = [A_L | A_R] where A_L, A_R are 2x2.
    B (4x6): rows 0,1 = B_T (2x6), rows 2,3 = B_B (2x6).
    
    C = A_L * B_T + A_R * B_B.
    
    Now B_T is 2x6. Split into B_T1 (2x2, cols 0,1), B_T2 (2x2, cols 2,3), B_T3 (2x2, cols 4,5).
    Similarly B_B into B_B1, B_B2, B_B3.
    
    C = A_L * [B_T1 | B_T2 | B_T3] + A_R * [B_B1 | B_B2 | B_B3]
      = [A_L*B_T1 + A_R*B_B1 | A_L*B_T2 + A_R*B_B2 | A_L*B_T3 + A_R*B_B3]
    
    Each block A_L*B_Ti + A_R*B_Bi is 2x2.
    For each output column block (2x2), we need to compute A_L*B_Ti + A_R*B_Bi.
    
    This is: [A_L | A_R] * [[B_Ti],[B_Bi]] where the second matrix is 4x2.
    So it's a 2x4 * 4x2 multiplication, which is equivalent to 2x2 * 2x2 twice (for the two columns of the 4x2).
    
    Actually, A_L*B_Ti + A_R*B_Bi where all are 2x2.
    Let X = A_L, Y = A_R, P = B_Ti, Q = B_Bi.
    Result = X*P + Y*Q.
    
    We can compute X*P with Strassen (rank 7), Y*Q with Strassen (rank 7), total 14.
    Or we can find a better way.
    
    Total: 3 column blocks * 14 = 42. Same as before.
    
    Let me just implement the Strassen decomposition for 2x2*2x2 and apply it 6 times
    (3 for A_L*B_Ti, 3 for A_R*B_Bi).
    
    Strassen for 2x2*2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22) B11
    M3 = A11 (B21 - B22)
    M4 = A22 (B12 - B11)
    M5 = (A11 + A12) B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 4, 6
    U = [[0.0] * 42 for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * 42 for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * 42 for _ in range(p * n)]  # 12 x 42
    
    # Strassen factors for 2x2 * 2x2
    # U indices: (i, j) -> i*2+j for i,j in {0,1}, but in our global U it's i*m+j = i*4+j
    # V indices: (j, k) -> j*3+k for j,k in {0,1,2}, but in global V it's j*p+k = j*6+k
    # W indices: (k, i) -> k*2+i for k in {0,1,2}, i in {0,1}, in global W it's k*n+i = k*2+i
    
    # For each of the 6 products (A_L * B_Ti and A_R * B_Bi for i=0,1,2):
    # A_L is rows 0,1 of A, cols 0,1 -> U indices (0,0),(0,1),(1,0),(1,1) -> global 0,1,4,5
    # A_R is rows 0,1 of A, cols 2,3 -> U indices (0,2),(0,3),(1,2),(1,3) -> global 2,3,6,7
    # B_Ti is rows 0,1 of B, cols 2i,2i+1 -> V indices (0,2i),(0,2i+1),(1,2i),(1,2i+1) -> global 2i,2i+1,6+2i,6+2i+1
    # B_Bi is rows 2,3 of B, cols 2i,2i+1 -> V indices (2,2i),(2,2i+1),(3,2i),(3,2i+1) -> global 12+2i,12+2i+1,18+2i,18+2i+1
    # Output C is rows 0,1, cols 2i,2i+1 -> W indices (2i,0),(2i,1),(2i+1,0),(2i+1,1) -> global 2i,2i+1,4+2i,4+2i+1
    
    strassen = [
        # (U_entries, V_entries, W_coeff)
        # Each entry is (global_index, coefficient)
        # M1 = (A11+A22)(B11+B22)
        [
            # U for (A11+A22): A11 at (0,0), A22 at (1,1)
            [(0, 1), (5, 1)],
            # V for (B11+B22): B11 at (0,0), B22 at (1,1)
            [(0, 1), (7, 1)],
            # W coefficients for C: C11+=M1, C22+=M1
            [(0, 1), (5, 1)]  # W[0,0]=1, W[1,1]=1
        ],
        # M2 = (A21+A22)B11
        [
            [(5, 1), (5, 1)],  # wait, A21 is (1,0) -> global 4, A22 is (1,1) -> global 5
            [(4, 1), (5, 1)],
            [(0, 1), (7, 1)],
            [(3, 1), (5, 1)]  # C21+=M2, C22+=-M2... 
        ],
    ]
    
    # Let me be more careful. I'll define the Strassen decomposition explicitly.
    
    # For a 2x2*2x2 product with local indices:
    # A has entries A[0,0], A[0,1], A[1,0], A[1,1]
    # B has entries B[0,0], B[0,1], B[1,0], B[1,1]
    # C has entries C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Strassen factors (U_local, V_local, W_local) where:
    # U_local has 4 rows (for A's 4 entries), V_local has 4 rows (for B's 4 entries), W_local has 4 rows (for C's 4 entries)
    
    # M1: U = [1,0,0,1] (A00+A11), V = [1,0,0,1] (B00+B11), W = [1,0,0,1] (C00+=1, C11+=1)
    # M2: U = [0,0,1,1] (A10+A11), V = [1,0,0,0] (B00), W = [0,0,1,-1] (C10+=1, C11+=-1)
    # M3: U = [1,0,0,0] (A00), V = [0,0,1,-1] (B10-B11), W = [0,1,0,0] (C01+=1)
    # M4: U = [0,0,0,1] (A11), V = [0,1,-1,0] (B01-B00), W = [1,0,1,0] (C00+=1, C10+=1)
    # M5: U = [1,1,0,0] (A00+A01), V = [0,0,0,1] (B11), W = [-1,1,0,0] (C00+=-1, C01+=1)
    # M6: U = [0,0,1,-1] (A10-A00), V = [1,1,0,0] (B00+B01), W = [0,0,0,1] (C11+=1)
    # M7: U = [0,1,0,-1] (A01-A11), V = [0,0,1,1] (B10+B11), W = [1,0,0,0] (C00+=1)
    
    strassen_factors = [
        # (U_local, V_local, W_local)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Now for each of the 6 blocks, assign 7 columns.
    # Block (a, i) where a in {L, R} and i in {0, 1, 2}
    # a=L: A rows 0,1, cols 0,1 -> global U indices: 0,1,4,5
    # a=R: A rows 0,1, cols 2,3 -> global U indices: 2,3,6,7
    # i=0: B cols 0,1
    # i=1: B cols 2,3
    # i=2: B cols 4,5
    
    # For a=L, B rows 0,1 (B_T), for a=R, B rows 2,3 (B_B)
    # B_T rows 0,1: global V indices for cols 2i, 2i+1: 2i, 2i+1, 6+2i, 6+2i+1
    # B_B rows 2,3: global V indices for cols 2i, 2i+1: 12+2i, 12+2i+1, 18+2i, 18+2i+1
    
    # Output C rows 0,1, cols 2i, 2i+1: global W indices: 2i, 2i+1, 4+2i, 4+2i+1
    
    col = 0
    for a in ['L', 'R']:
        for i in range(3):
            # Determine global index mappings
            if a == 'L':
                # A_L: local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
                u_map = [0, 1, 4, 5]
                # B_T: local (0,0)->2i, (0,1)->2i+1, (1,0)->6+2i, (1,1)->6+2i+1
                v_map = [2*i, 2*i+1, 6+2*i, 6+2*i+1]
            else:
                # A_R: local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
                u_map = [2, 3, 6, 7]
                # B_B: local (0,0)->12+2i, (0,1)->12+2i+1, (1,0)->18+2i, (1,1)->18+2i+1
                v_map = [12+2*i, 12+2*i+1, 18+2*i, 18+2*i+1]
            
            # W: output C cols 2i, 2i+1, rows 0,1
            # local (0,0)->2i, (0,1)->2i+1, (1,0)->4+2i, (1,1)->4+2i+1
            w_map = [2*i, 2*i+1, 4+2*i, 4+2*i+1]
            
            for factor in strassen_factors:
                u_local, v_local, w_local = factor
                for li in range(4):
                    if u_local[li] != 0:
                        U[u_map[li]][col] += u_local[li]
                    if v_local[li] != 0:
                        V[v_map[li]][col] += v_local[li]
                    if w_local[li] != 0:
                        W[w_map[li]][col] += w_local[li]
                col += 1
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    try:
        U, V, W = _build_strassen_factors()
        # Convert to lists if needed, they're already lists
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END

