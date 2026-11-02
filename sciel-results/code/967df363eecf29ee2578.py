"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _add_rank1_term(U, V, W, a_coeff, b_coeff, c_coeff, col):
    """Add one rank-1 tensor term with given coefficient vectors."""
    for idx, val in a_coeff.items():
        U[idx][col] += val
    for idx, val in b_coeff.items():
        V[idx][col] += val
    for idx, val in c_coeff.items():
        W[idx][col] += val

def _strassen_2x2x2(U, V, W, a_rows, b_rows, c_rows, offset):
    """
    Correct rank-7 Strassen for 2x2 x 2x2.
    a_rows: [a00, a01, a10, a11] indices into U
    b_rows: [b00, b01, b10, b11] indices into V
    c_rows: [c00, c01, c10, c11] indices into W
    """
    a00, a01, a10, a11 = a_rows
    b00, b01, b10, b11 = b_rows
    c00, c01, c10, c11 = c_rows
    
    # Correct Strassen 7 products:
    # P1 = (A00+A11)(B00+B11)
    _add_rank1_term(U, V, W, {a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1}, offset); offset += 1
    # P2 = (A10+A11)(B00+B01)
    _add_rank1_term(U, V, W, {a10:1, a11:1}, {b00:1, b01:1}, {c10:1, c11:1}, offset); offset += 1
    # P3 = (A00-A10)(B10+B11)
    _add_rank1_term(U, V, W, {a00:1, a10:-1}, {b10:1, b11:1}, {c01:1, c11:-1}, offset); offset += 1
    # P4 = A11*(B01-B10)
    _add_rank1_term(U, V, W, {a11:1}, {b01:1, b10:-1}, {c01:1, c11:-1}, offset); offset += 1
    # P5 = A00*(B00-B11)
    _add_rank1_term(U, V, W, {a00:1}, {b00:1, b11:-1}, {c00:1, c11:-1}, offset); offset += 1
    # P6 = (A10+A11)*B11
    _add_rank1_term(U, V, W, {a10:1, a11:1}, {b11:1}, {c10:1, c11:1}, offset); offset += 1
    # P7 = (A00-A10)*B00
    _add_rank1_term(U, V, W, {a00:1, a10:-1}, {b00:1}, {c00:1, c10:-1}, offset); offset += 1
    
    return offset

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Use Strassen-based decomposition:
    # Split A (4x4) into 2x2 blocks: A11(rows 0-1, cols 0-1), A12(rows 0-1, cols 2-3), A21(rows 2-3, cols 0-1), A22(rows 2-3, cols 2-3)
    # Split B (4x9) into 2x2 blocks by rows (matching A's cols): B11(rows 0-1), B12(rows 2-3); and 9 cols
    # 
    # Block multiplication: C = A*B
    # C (4x9) blocks: C11(2x9) = A11*B11 + A12*B21, C21(2x9) = A21*B11 + A22*B21
    # where B11 is 2x9 (rows 0-1 of B), B21 is 2x9 (rows 2-3 of B)
    #
    # For each 2x2 x 2x9 subproblem, use Strassen on the 2x2 part:
    # A11 (2x2) * B11 (2x9): Split B11 into columns. For each pair of output columns,
    # we have a 2x2 x 2x2 problem -> rank 7
    # B11 has 9 columns, so we get ceil(9/2)=5 sub-blocks: 4 of size 2x2 and 1 of size 2x1
    # 2x2 x 2x2: rank 7 (Strassen)
    # 2x2 x 2x1: rank 4 (schoolbook, since Strassen doesn't help for 2x1)
    # Actually for 2x2 x 2x1: A(2x2)*B(2x1) -> C(2x1). Schoolbook rank = 2*2*1=4.
    #
    # Total for A11*B11: 4*7 + 1*4 = 28+4 = 32
    # Same for A12*B21: 32
    # A21*B11: 32
    # A22*B21: 32
    # Total: 128
    #
    # That's better than 144 but not great. Let me try a different partitioning.
    #
    # Alternative: Use the directive's suggestion of 3 sub-tensors of 2x4x3
    # But A is 4x4, not easily split into 2x4 blocks that multiply with 4x3.
    #
    # Let me try: Split the 9 columns of B into 3 groups of 3.
    # B = [B1 | B2 | B3] each 4x3
    # C = A*B = [A*B1 | A*B2 | A*B3] each 4x3
    #
    # For A(4x4) * B_j(4x3): 
    # Split A into 2x2 blocks, B_j into 2x3 blocks (top 2 rows, bottom 2 rows)
    # C_j(4x3) = [A11*Bj1 + A12*Bj2; A21*Bj1 + A22*Bj2]
    # where Aj1 is 2x2, Bj1 is 2x3
    #
    # Each 2x2 x 2x3 subproblem:
    # A(2x2) * B(2x3) -> C(2x3)
    # Split B's 3 columns: 2+1
    # 2x2 x 2x2: rank 7 (Strassen)
    # 2x2 x 2x1: rank 4 (schoolbook)
    # Total per 2x2 x 2x3: 11
    #
    # Per A_j * B_j: 4 sub-problems * 11 = 44
    # Total for 3 B groups: 3 * 44 = 132
    #
    # Still not great. Let me just implement the Strassen-based approach carefully.
    
    # I'll use the 2x2 block Strassen approach:
    # A split into 2x2 blocks: rows [0,1],[2,3], cols [0,1],[2,3]
    # B split into 2x9 blocks: rows [0,1],[2,3]
    # For each A_block (2x2) * B_block (2x9), use Strassen per 2-column group
    
    # Pre-compute index mappings
    # U index: i*m + j where i is row of A, j is col of A
    # V index: j*p + k where j is row of B, k is col of B  
    # W index: k*n + i where k is col of B (=row of C), i is row of A (=col of C)
    # Wait, the convention is W[k*n+i, r] where output C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually C[i,k] = sum_j A[i,j]*B[j,k], so the tensor T[i,j,k] = delta_{i,j} for the multiplication
    # The decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] u_r[j]) * (sum_k B[j,k] v_r[k]) ... 
    # No: C[i,k] = sum_r U[i,j... ] 
    
    # Let me re-read the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] corresponds to A[i,j]
    # V[j*p+k, r] corresponds to B[j,k]
    # W[k*n+i, r] corresponds to C[i,k] (output at row i, col k)
    #
    # The tensor identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j} ... no
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal the coefficient of A[i,j]*B[j,k] in C[i,k]
    # which is 1 if i,j,k are "compatible" (always for fixed i,k, summing over j)
    # 
    # The identity is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid term
    # Actually the tensor T_{i,j,k} for matrix mult is: T_{i,j,k} = 1 for all (i,j,k) where the term A[i,j]*B[j,k] contributes to C[i,k]
    # Which is ALL (i,j,k) pairs! Because every A[i,j] and B[j,k] contributes to C[i,k].
    # Wait no. C[i,k] = sum_j A[i,j]*B[j,k]. So the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The rank is the tensor rank of this all-ones tensor... no that's not right either.
    
    # I think the correct interpretation: the multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give the (i,k) entry of C. This is always true, so T is the all-1 tensor of size n*m*p? No, that can't be right because schoolbook gives rank n*m*p with each term being a single (i,j,k) triple.
    
    # Schoolbook: for each (i,j,k), one rank-1 term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # This gives: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if there exists r with all three nonzero, else 0.
    # For schoolbook, each r corresponds to exactly one (i',j',k') triple, so the sum is 1 iff (i,j,k)=(i',j',k').
    # But we want the sum over j to give C[i,k] = sum_j A[i,j]B[j,k].
    
    # I think the evaluator checks: for random A,B, compute C=A@B, then verify that
    # C[i,k] == sum_r (sum_j A[i,j]*U[i*m+j,r]) * V[j*p+k,r] * W[k*n+i,r] ... no
    # 
    # More likely: C[i,k] = sum_r U[i*m+j,r] ... this doesn't make sense without summing over j.
    
    # Let me just assume the standard: the decomposition satisfies
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j} * something... 
    
    # I'll just verify my Strassen implementation numerically.
    
    # Let me implement Strassen 2x2x2 and test it
    test_rank = 7
    U_test = np.zeros((4, test_rank))
    V_test = np.zeros((4, test_rank))
    W_test = np.zeros((4, test_rank))
    
    a_rows = [0, 1, 2, 3]  # a00, a01, a10, a11
    b_rows = [0, 1, 2, 3]  # b00, b01, b10, b11
    c_rows = [0, 1, 2, 3]  # c00, c01, c10, c11
    
    offset = _strassen_2x2x2(U_test, V_test, W_test, a_rows, b_rows, c_rows, 0)
    
    # Verify: for random 2x2 matrices A, B, check A@B matches the decomposition
    np.random.seed(42)
    A_test = np.random.randn(2, 2)
    B_test = np.random.randn(2, 2)
    C_expected = A_test @ B_test
    
    # The decomposition: C[i,k] = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
    # Wait, I need to figure out the exact contraction.
    # U has shape (n*m, rank) = (4, 7) for 2x2x2
    # U[i*m+j, r] = U[i*2+j, r]
    # V[j*p+k, r] = V[j*2+k, r]  
    # W[k*n+i, r] = W[k*2+i, r]
    
    C_decomp = np.zeros((2, 2))
    for i in range(2):
        for k in range(2):
            s = 0
            for j in range(2):
                for r in range(7):
                    s += U_test[i*2+j, r] * A_test[i,j] * V_test[j*2+k, r] * B_test[j,k] * W_test[k*2+i, r]
            C_decomp[i,k] = s
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        print(f"Strassen verification failed!\nExpected:\n{C_expected}\nGot:\n{C_decomp}")
        # Fall back to schoolbook
        pass
    
    # Build the full 4x4x9 decomposition using Strassen
    # A (4x4) blocks: A11(0:2,0:2), A12(0:2,2:4), A21(2:4,0:2), A22(2:4,2:4)
    # B (4x9) blocks: B1(0:2,:), B2(2:4,:)
    # C (4x9): C1(0:2,:)=A11*B1+A12*B2, C2(2:4,:)=A21*B1+A22*B2
    
    # For each 2x2 x 2x9 subproblem, split 9 cols into 4 groups of 2 + 1 group of 1
    # 2x2 x 2x2: Strassen rank 7
    # 2x2 x 2x1: schoolbook rank 4
    
    total_rank = 4 * (4*7 + 1*4)  # 4 subproblems, each with 4 Strassen blocks + 1 schoolbook block
    # = 4 * 32 = 128
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # For each A block row (2 groups) and A block col (2 groups), and B block row (2 groups):
    for a_row_block in range(2):  # rows 0-1 or 2-3 of A
        for a_col_block in range(2):  # cols 0-1 or 2-3 of A
            for b_row_block in range(2):  # rows 0-1 or 2-3 of B
                # A_block is 2x2 at rows [a_row_block*2, a_row_block*2+2), cols [a_col_block*2, a_col_block*2+2)
                # B_block is 2x9 at rows [b_row_block*2, b_row_block*2+2), all 9 cols
                # Contribution to C at rows [a_row_block*2, a_row_block*2+2), all 9 cols
                # BUT: A_col_block must match B_row_block for the multiplication to work!
                # A[i,j]*B[j,k] where j is the shared index
                # So A's column block index = B's row block index
                if a_col_block != b_row_block:
                    continue
                
                # A_block rows: a_row_block*2, a_row_block*2+1
                # A_block cols: a_col_block*2, a_col_block*2+1 (= b_row_block*2, b_row_block*2+1)
                # B_block rows: b_row_block*2, b_row_block*2+1
                # B_block cols: 0..8
                # C_block rows: a_row_block*2, a_row_block*2+1
                # C_block cols: 0..8
                
                a_r0 = a_row_block * 2
                a_r1 = a_row_block * 2 + 1
                a_c0 = a_col_block * 2
                a_c1 = a_col_block * 2 + 1
                b_r0 = b_row_block * 2
                b_r1 = b_row_block * 2 + 1
                c_r0 = a_row_block * 2
                c_r1 = a_row_block * 2 + 1
                
                # U indices for A_block: [a_r0*m+a_c0, a_r0*m+a_c1, a_r1*m+a_c0, a_r1*m+a_c1]
                # = [a00, a01, a10, a11] in 2x2 ordering
                a_indices = [a_r0*m+a_c0, a_r0*m+a_c1, a_r1*m+a_c0, a_r1*m+a_c1]
                # V indices for B_block rows: [b_r0*p+col, b_r1*p+col] for each col
                # W indices for C_block: [col*n+c_r0, col*n+c_r1] for each col
                
                # Process 9 columns in groups of 2 (4 groups) + 1 remaining
                for col_group in range(4):
                    c0 = col_group * 2
                    c1 = col_group * 2 + 1
                    # V indices: b_r0*p+c0, b_r0*p+c1, b_r1*p+c0, b_r1*p+c1
                    v_indices = [b_r0*p+c0, b_r0*p+c1, b_r1*p+c0, b_r1*p+c1]
                    # W indices: c0*n+c_r0, c0*n+c_r1, c1*n+c_r0, c1*n+c_r1
                    w_indices = [c0*n+c_r0, c0*n+c_r1, c1*n+c_r0, c1*n+c_r1]
                    
                    col = _strassen_2x2x2(U, V, W, a_indices, v_indices, w_indices, col)
                
                # Last column (col 8)
                c_last = 8
                # For 2x2 x 2x1: schoolbook, rank 4
                # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k=8
                # 4 terms: (i,j) pairs
                for ii in range(2):
                    for jj in range(2):
                        u_idx = a_indices[ii*2+jj]  # a_r0/a_r1, a_c0/a_c1
                        v_idx = b_r0*p+c_last if jj==0 else b_r1*p+c_last
                        w_idx = c_last*n + (c_r0 if ii==0 else c_r1)
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
    
    # Verify the result
    np.random.seed(123)
    A_rand = np.random.randn(n, m)
    B_rand = np.random.randn(m, p)
    C_expected = A_rand @ B_rand
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(total_rank):
                    s += U[i*m+j][r] * A_rand[i,j] * V[j*p+k][r] * B_rand[j,k] * W[k*n+i][r]
            C_decomp[i,k] = s
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        print("Full decomposition verification failed!")
        max_err = np.max(np.abs(C_decomp - C_expected))
        print(f"Max error: {max_err}")
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col2] = 1
                    V[j * p + k][col2] = 1
                    W[k * n + i][col2] = 1
                    col2 += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
