"""Strassen-based tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# Strassen 2x2x2 rank-7 decomposition
# Indices: A is 2x2 (rows 0,1; cols 0,1), B is 2x2, C is 2x2
# W index convention: W[k*n+i, r] where n=2 for 2x2 block
# For 2x2x2: n=2, m=2, p=2
# U has shape (n*m, rank) = (4, 7)
# V has shape (m*p, rank) = (4, 7)
# W has shape (p*n, rank) = (4, 7)
# Index mapping: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]

STRASSEN = {
    # (u_idx, v_idx, w_idx, sign)
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
}

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen rank-7."""
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i in {0,1}, j in {0,1}, k in {0,1}
    # c in {0..6}
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00 at (0,0)->idx 0, A11 at (1,1)->idx 3
    # V: B00 at (0,0)->idx 0, B11 at (1,1)->idx 3
    # W: C00 at (0,0)->idx 0, C11 at (1,1)->idx 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    # U: A10 at (1,0)->idx 2, A11 at (1,1)->idx 3
    # V: B00 at (0,0)->idx 0
    # W: C10 at (0,1)->idx 2, C11 at (1,1)->idx 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    # U: A00 at (0,0)->idx 0
    # V: B01 at (0,1)->idx 1, B11 at (1,1)->idx 3
    # W: C00 at (0,0)->idx 0, C01 at (1,0)->idx 1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = -1; W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    # U: A11 at (1,1)->idx 3
    # V: B10 at (1,0)->idx 2, B00 at (0,0)->idx 0
    # W: C00 at (0,0)->idx 0, C10 at (0,1)->idx 2
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    # U: A00 at (0,0)->idx 0, A01 at (0,1)->idx 1
    # V: B11 at (1,1)->idx 3
    # W: C00 at (0,0)->idx 0, C01 at (1,0)->idx 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10 at (1,0)->idx 2, A00 at (0,0)->idx 0
    # V: B00 at (0,0)->idx 0, B01 at (0,1)->idx 1
    # W: C11 at (1,1)->idx 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01 at (0,1)->idx 1, A11 at (1,1)->idx 3
    # V: B10 at (1,0)->idx 2, B11 at (1,1)->idx 3
    # W: C00 at (0,0)->idx 0, C11 at (1,1)->idx 3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll build the decomposition using Strassen for the 4x4x4 part
    # and schoolbook for the remaining 3 columns of the 7-column B matrix.
    # 
    # Split: B is 4x7. Split into B1 (4x4) and B2 (4x3).
    # C = A*B = A*B1 + A*B2
    # For A*B1: use Strassen-based 4x4x4 decomposition (rank 49)
    # For A*B2: use schoolbook 4x4x3 (rank 48)
    # Total rank: 49 + 48 = 97
    #
    # But we can do better by using Strassen structure more cleverly.
    # Let's first build the 4x4x4 Strassen decomposition.
    
    # 4x4x4 via two levels of Strassen (4x4 = 2x(2x2))
    # Actually, 4x4x4 can be done with rank 49 using Strassen recursively.
    # But let's think about it differently.
    
    # For 4x4 * 4x4:
    # Split into 2x2 blocks. Use Strassen for each "level".
    # Actually the cleanest is: 4x4x4 = (2x2 blocks) with Strassen at each level.
    # Rank = 7^2 = 49.
    
    # Let me build the 4x4x4 Strassen decomposition.
    # A is 4x4, B is 4x4, C is 4x4.
    # Block structure: A = [[A00, A01], [A10, A11]] each 2x2
    # Similarly B and C.
    # Strassen gives 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with rank 7 (Strassen).
    # Total rank = 7 * 7 = 49.
    
    # For the 4x4x7 case:
    # A is 4x4, B is 4x7.
    # Split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2.
    # A*B1: rank 49 (Strassen 4x4x4)
    # A*B2: rank 4*4*3 = 48 (schoolbook)
    # Total: 97.
    
    # Let's build this.
    
    # First, build 4x4x4 Strassen decomposition.
    # U4 has shape (4*4, 49) = (16, 49)
    # V4 has shape (4*4, 49) = (16, 49)
    # W4 has shape (4*4, 49) = (16, 49)
    
    # The 4x4x4 Strassen: 
    # Level 1: 7 Strassen terms for 2x2 block matrices
    # Level 2: each term is a 2x2x2 Strassen (rank 7)
    # So total 49 terms.
    
    # Index mapping for 4x4x4:
    # U[i*4+j, c] where i,j in 0..3
    # V[j*4+k, c] where j,k in 0..3
    # W[k*4+i, c] where k,i in 0..3
    
    # Strassen level 1 (2x2 blocks of 2x2):
    # A blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # B blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    
    # Strassen terms for 2x2 block level:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For each term, we need a 2x2x2 decomposition (rank 7).
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen 2x2x2 base
    S_U, S_V, S_W = _strassen_222()
    # S_U: (4,7), S_V: (4,7), S_W: (4,7)
    # For 2x2x2: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Define the 7 Strassen terms at block level
    # Each term: (a_indices, b_indices, c_indices, signs)
    # a_indices: list of (block_row, block_col) with sign
    # b_indices: list of (block_row, block_col) with sign  
    # c_indices: list of (block_row, block_col) with sign
    
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)*B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00*(B01-B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,0,-1),(0,1,1)]),
        # M4 = A11*(B10-B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)*B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)*(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)*(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ]
    
    for term_idx, (a_idx, b_idx, c_idx) in enumerate(strassen_terms):
        # For this term, we have a 2x2x2 sub-problem with coefficients
        # The "effective" A for this term is a sum of 2x2 blocks
        # We need to embed the Strassen 2x2x2 into the 4x4x4
        
        # For each of the 7 Strassen sub-terms:
        for sub_c in range(7):
            col = term_idx * 7 + sub_c
            
            # U part: for each (block_i, block_j, sign) in a_idx
            for (bi, bj, sign_a) in a_idx:
                # The 2x2 block A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
                # In U4: row index = (bi*2 + ii) * 4 + (bj*2 + jj)
                # S_U row index = ii*2 + jj
                for ii in range(2):
                    for jj in range(2):
                        u_row = (bi * 2 + ii) * 4 + (bj * 2 + jj)
                        s_row = ii * 2 + jj
                        U4[u_row, col] += sign_a * S_U[s_row, sub_c]
            
            # V part: for each (block_j, block_k, sign) in b_idx
            for (bj, bk, sign_b) in b_idx:
                for jj in range(2):
                    for kk in range(2):
                        v_row = (bj * 2 + jj) * 4 + (bk * 2 + kk)
                        s_row = jj * 2 + kk
                        V4[v_row, col] += sign_b * S_V[s_row, sub_c]
            
            # W part: for each (block_i, block_k, sign) in c_idx
            for (bi, bk, sign_c) in c_idx:
                for ii in range(2):
                    for kk in range(2):
                        w_row = (bk * 2 + kk) * 4 + (bi * 2 + ii)
                        s_row = kk * 2 + ii
                        W4[w_row, col] += sign_c * S_W[s_row, sub_c]
    
    # Verify 4x4x4 decomposition
    if time.time() - start_time < deadline - 5:
        # Check: sum_c U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c] should give delta
        # Actually the tensor T[i,j,k] = sum_c U[i*4+j,c]*V[j*4+k,c]*W[k*4+i,c]
        # should equal 1 if i==k... no, for matrix mult A*B=C:
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = delta_{i,k}... no.
        # The matrix multiplication tensor: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
        # contribute to (i,k) entry of C. So T[i,j,k] = 1 for all i,j,k? No.
        # Actually for matrix mult, the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
        # So T[i,j,k] = 1 for all valid i,j,k. That's just the identity in a sense.
        # Wait no. The tensor for matrix mult of n x m times m x p is:
        # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 (it's just the standard basis, every triple is 1)
        # No that's not right either. Let me think again.
        # 
        # The matrix multiplication tensor M_{n,m,p} has entries:
        # M[i,j,k] = 1 if the computation of C[i,k] involves A[i,j] * B[j,k]
        # So M[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
        # 
        # Actually yes! For matrix multiplication, the tensor is simply all ones.
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # 
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        
        # Let's verify for 4x4x4
        error = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for c in range(49):
                        s += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                    error += abs(s - 1)
        # error should be 0
        pass
    
    # Now build 4x4x7:
    # Split B into B1 (4x4, cols 0-3) and B2 (4x3, cols 4-6)
    # C = A*B1 + A*B2
    
    # For A*B1: use U4, V4 (first 4 cols of V), W4
    # For A*B2: schoolbook 4x4x3
    
    # Total rank = 49 + 48 = 97
    
    rank = 49 + 48
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((28, rank))  # p*n = 7*4 = 28
    
    # Part 1: A*B1 (4x4 * 4x4 -> 4x4)
    # U[0:16, 0:49] = U4
    # V[0:16, 0:49]: V4 but only for k in 0..3 (first 4 rows of B)
    # V4 has shape (16, 49) where V4[j*4+k, c] for j,k in 0..3
    # For B1, k in 0..3, so V[0:16, 0:49] = V4
    # W[0:16, 0:49] = W4 (for C1, which is C[0:4, 0:4])
    # W[k*4+i, c] for k in 0..3, i in 0..3 -> rows 0..15
    
    U[:, 0:49] = U4
    V[:, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Part 2: A*B2 (4x4 * 4x3 -> 4x3)
    # Schoolbook: for each i, j, k in 0..3, 0..3, 0..2
    # U[i*4+j, 49 + (j*3+k)] = 1  ... wait, let me use a cleaner indexing
    # Column index for schoolbook: j*3 + k for j in 0..3, k in 0..2
    # That gives 4*3 = 12... no, 4*3=12? No, 4*3=12. But we need 4*4*3=48.
    # U[i*4+j, col] where col = i*12 + j*3 + k? No.
    # Let's use: col = (i*4 + j)*3 + k for i in 0..3, j in 0..3, k in 0..2
    # That gives 16*3 = 48 columns.
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                c = col_offset + (i * 4 + j) * 3 + k
                U[i * 4 + j, c] = 1
                V[j * 4 + (k + 4), c] = 1  # B2[k] is column k+4 of B
                W[(k + 4) * 4 + i, c] = 1
    
    # Verify the full 4x4x7 decomposition
    if time.time() - start_time < deadline - 3:
        # Check: sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c] = 1 for all i,j,k
        # Wait, V has shape (m*p, rank) = (4*7, rank) = (28, rank)
        # But I defined V with shape (16, rank). That's wrong!
        # V should have shape (m*p, rank) = (4*7, rank) = (28, rank)
        
        # Let me redo this.
        pass
    
    # I made an error. Let me redo with correct shapes.
    # U: (n*m, rank) = (4*4, rank) = (16, rank)
    # V: (m*p, rank) = (4*7, rank) = (28, rank)
    # W: (p*n, rank) = (7*4, rank) = (28, rank)
    
    rank = 49 + 48
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Part 1: A*B1 where B1 is B[:, 0:4] (4x4)
    # U4: (16, 49), V4: (16, 49) for 4x4x4
    # But V for 4x4x7 has shape (28, rank). For B1 (cols 0-3), V[j*7+k, c] for k in 0..3
    # V4 was defined for 4x4x4 as V4[j*4+k, c]. Now V[j*7+k, c].
    # So V[j*7+k, c] = V4[j*4+k, c] for k in 0..3
    
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, 0:49] = V4[j * 4 + k, 0:49]
    # W4: (16, 49) for 4x4x4, W4[k*4+i, c] for k in 0..3
    # For 4x4x7, W[k*4+i, c] for k in 0..6
    # For B1 part, k in 0..3
    W[0:16, 0:49] = W4  # W[k*4+i] for k in 0..3, i in 0..3 -> rows 0..15
    
    # Part 2: A*B2 where B2 is B[:, 4:7] (4x3)
    # Schoolbook
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                c = col_offset + (i * 4 + j) * 3 + k
                U[i * 4 + j, c] = 1
                V[j * 7 + (k + 4), c] = 1
                W[(k + 4) * 4 + i, c] = 1
    
    # Verify
    if time.time() - start_time < deadline - 2:
        max_err = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    s = 0
                    for c in range(rank):
                        s += U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c]
                    max_err = max(max_err, abs(s - 1))
        # If max_err > 0, the decomposition is wrong
    
    # Convert to lists
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U.tolist()]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V.tolist()]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
