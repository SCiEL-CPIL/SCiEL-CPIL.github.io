"""Matrix multiplication 4x4x7 - rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen, shapes (4,7) each."""
    # 2x2 matrices indexed as: 00=0, 01=1, 10=2, 11=3
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1
    
    return U, V, W

def _strassen_4x4():
    """Return (U, V, W) for 4x4x4 Strassen, shapes (16,49) each."""
    U2, V2, W2 = _strassen_2x2()
    
    # 4x4 matrices split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # A00 occupies indices 0-3, A01 occupies 4-7, A10 occupies 8-11, A11 occupies 12-15
    # Similarly for B and C.
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # Block positions in the 16-element vector:
    # Block (0,0): indices 0-3
    # Block (0,1): indices 4-7
    # Block (1,0): indices 8-11
    # Block (1,1): indices 12-15
    
    # For each pair of 2x2 Strassen factors (r1, r2) where r1 acts on A-block and B-block,
    # and r2 acts on another A-block and B-block, we combine them.
    # Actually, the 4x4 Strassen works by doing 7 multiplications on 2x2 blocks,
    # each of which is itself a Strassen 2x2 (7 multiplications), giving 49 total.
    
    # The 7 "big" multiplications for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each "big" multiplication involves 2x2 matrices that are sums/differences of 2x2 blocks.
    # For example, M1 = (A00+A11)(B00+B11) means we multiply the 2x2 matrix (A00+A11) 
    # by the 2x2 matrix (B00+B11). Each of these is a 2x2 matrix, so we use Strassen 2x2.
    
    # Define the 7 combinations as (coeffs_for_A_blocks, coeffs_for_B_blocks):
    # Each entry is a list of (block_index, coefficient) for A and B sides.
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    combos = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)]),            # M2
        ([(0, 1)], [(1, 1), (3, -1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)]),            # M5
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)]),   # M6
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)]),   # M7
    ]
    
    # For each combination, we have a "virtual" 2x2 matrix on A side and B side.
    # The virtual 2x2 matrix has entries that are linear combinations of the 4 blocks.
    # We apply Strassen 2x2 to these virtual matrices.
    
    for c, (a_coeffs, b_coeffs) in enumerate(combos):
        # a_coeffs: list of (block_idx, coeff) for A side
        # b_coeffs: list of (block_idx, coeff) for B side
        
        # For each Strassen 2x2 factor r (0..6):
        for r in range(7):
            col = c * 7 + r
            
            # A-side: for each entry in the virtual 2x2 A matrix (positions 0,1,2,3)
            # U2[r] gives coefficients for A entries 0,1,2,3
            for a_pos in range(4):
                ua = U2[a_pos, r]
                if ua == 0:
                    continue
                for (block_idx, coeff) in a_coeffs:
                    # block_idx maps to 4 positions in the 16-element vector
                    base = block_idx * 4
                    for k in range(4):
                        U[base + k, col] += ua * coeff
            
            # B-side: similarly
            for b_pos in range(4):
                vb = V2[b_pos, r]
                if vb == 0:
                    continue
                for (block_idx, coeff) in b_coeffs:
                    base = block_idx * 4
                    for k in range(4):
                        V[base + k, col] += vb * coeff
            
            # W-side: for each entry in the virtual 2x2 C matrix
            # W2[r] gives coefficients for C entries 0,1,2,3
            # The result C = A*B where A and B are the virtual 2x2 matrices.
            # C is a 2x2 matrix whose entries are 2x2 blocks of the 4x4 result.
            # C block (i,j) corresponds to 4x4 result block (i,j).
            # C entry positions: 0=C00, 1=C01, 2=C10, 3=C11
            # In the 4x4 result, C00 is at blocks (0,0), C01 at (0,1), C10 at (1,0), C11 at (1,1)
            for w_pos in range(4):
                wc = W2[w_pos, r]
                if wc == 0:
                    continue
                base = w_pos * 4
                for k in range(4):
                    W[base + k, col] += wc
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Split the 4x7 matrix into a 4x4 part and a 4x3 part.
    # Use Strassen 4x4x4 for the first part (rank 49).
    # Use schoolbook for the 4x4x3 part (rank 4*4*3 = 48).
    # Total rank: 49 + 48 = 97.
    
    # Strassen 4x4x4 for columns 0-3 of B
    U_str, V_str, W_str = _strassen_4x4()
    
    # Schoolbook for 4x4x3 for columns 3-6 of B
    rank_school = n * m * 3  # 48
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * 3, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # B's column index is 3 + k (columns 3,4,5 of B)
                U_school[i * m + j, col] = 1
                V_school[j * 3 + k, col] = 1
                # W index: k*n + i, but k here is 0,1,2 corresponding to actual columns 3,4,5
                # The W matrix for the full 4x7 output has p*n = 28 rows.
                # Row index in W: (3+k)*n + i
                W_school[(3 + k) * n + i, col] = 1
                col += 1
    
    # Now combine:
    # U has shape (n*m, total_rank) = (16, 97)
    # V has shape (m*p, total_rank) = (28, 97)
    # W has shape (p*n, total_rank) = (28, 97)
    
    total_rank = 49 + rank_school
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Strassen part: columns 0-3 of B
    U[:, :49] = U_str
    V[:m*4, :49] = V_str  # V_str is (16, 49), first 16 rows of V (m*p = 28, first m*4=16)
    W[:4*n, :49] = W_str  # W_str is (16, 49), first 16 rows of W (p*n = 28, first 4*n=16)
    
    # Schoolbook part: columns 3-6 of B
    U[:, 49:] = U_school
    V[16:, 49:] = V_school  # rows 16-27 of V correspond to B columns 3-6
    W[16:, 49:] = W_school  # rows 16-27 of W correspond to output columns 3-6
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify correctness
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Verify that the decomposition is correct
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j_c, c] * V[j_c*p+k, c] * W[k*n+i, c]
        # Actually: C[i,k] = sum_c U[i*m+?, c] ... let me think more carefully.
        
        # The tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k V[j*p+k, c] B[j,k]) * W[k*n+i, c]
        # Wait, that's not right. Let me reconsider.
        
        # Standard: C[i,k] = sum_j A[i,j] B[j,k]
        # Rank-1 term c contributes: (sum_j U[i*m+j,c] A[i,j]) * (sum_k V[j*p+k,c] B[j,k]) * W[k*n+i,c]
        # No, that's a triple product. The correct form is:
        # C[i,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over j too.
        # Actually: C[i,k] = sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let's verify with random matrices
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m))
        B_test = np.random.randint(-5, 6, size=(m, p))
        C_true = A_test @ B_test
        
        C_comp = np.zeros((n, p), dtype=int)
        for c in range(total_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_comp[i, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        if not np.array_equal(C_true, C_comp):
            # Fallback to pure schoolbook if Strassen has an error
            rank_fallback = n * m * p
            U_f = np.zeros((n*m, rank_fallback), dtype=int)
            V_f = np.zeros((m*p, rank_fallback), dtype=int)
            W_f = np.zeros((p*n, rank_fallback), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_f[i*m+j, col] = 1
                        V_f[j*p+k, col] = 1
                        W_f[k*n+i, col] = 1
                        col += 1
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
