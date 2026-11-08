import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Rank 20 decomposition:
    # We view A (3x3) as a 3x1 block of 1x3 rows, and B (3x7) as a 3x3 block of 1x3 cols?
    # No, the directive says: view 3x3 matrix as 3x1 block of 1x3 row vectors 
    # and 3x7 matrix as 3x3 block of 1x3 column vectors? 
    # Actually, let's look at the structure of the problem.
    # C = A B. A is 3x3, B is 3x7.
    # We can use Strassen for the 3x3 * 3x3 part (columns 0-2 of B) to get C[:, 0:3].
    # Strassen for 3x3*3x3 is not standard (Strassen is 2x2). 
    # However, we can decompose the 3x3 multiplication into smaller blocks.
    
    # Let's try a different approach based on the "Kronecker product" hint.
    # The hint suggests using a rank-9 (or rank-7?) decomposition for 3x3 mat mult.
    # Standard schoolbook 3x3*3x3 is rank 27.
    # Strassen is for 2x2.
    # There is no known rank < 27 for 3x3*3x3 over general fields? 
    # Wait, the rank of 3x3*3x3 matrix multiplication is at least 19 (Landsberg) and at most 27.
    # But finding an explicit decomposition of rank < 27 is hard.
    
    # Let's re-read the inspiration.
    # Inspiration 1: "standard schoolbook multiplication has rank n*m*p = 3*3*7 = 63".
    # Current program tries rank 45 but fails/falls back to 63?
    # The current program's `construct` returns rank 45 initially but the verification might fail?
    # Actually, the current program has a bug: it initializes `rank=45` but the schoolbook loops add 27 + 36 = 63 terms.
    # `add_rank1_term` checks `if col >= rank: return`. So it only adds 45 terms?
    # If it only adds 45 terms, the decomposition is incomplete and verification fails.
    # Then it falls back to schoolbook 63.
    # So the current program effectively returns rank 63.
    # Score 1.0 corresponds to rank 63 (since seed value is 63, score = 63/63 = 1.0).
    
    # We need to find a rank < 63 decomposition.
    # Strategy: Use Strassen for parts of the multiplication.
    # B is 3x7. We can split B into B1 (3x3) and B2 (3x4).
    # C = A B1 + A B2.
    # A B1 is 3x3 * 3x3.
    # A B2 is 3x3 * 3x4.
    
    # Can we use Strassen for 3x3 * 3x3?
    # 3x3 can be viewed as a 2x2 block matrix where blocks are 1x1, 1x2, 2x1, 2x2?
    # Or 3x3 as 2x2 with 1x1 blocks plus extra?
    # Standard Strassen applies to 2x2 matrices.
    # If we split A into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # This is messy.
    
    # Alternative: Split A into 2x2 and 1x1?
    # Let's use the known rank-19 decomposition for 3x3*3x3? 
    # Actually, the rank of 3x3*3x3 is not known to be less than 27 for explicit simple formulas, 
    # but we can use the 2x2 Strassen on sub-blocks.
    
    # Let's try to compute A B using Strassen on the 2x2 top-left sub-blocks.
    # A = [ A_2x2  A_2x1 ]
    #     [ A_1x2  A_1x1 ]
    # B = [ B_2x3  B_2x4 ]
    #     [ B_1x3  B_1x4 ]
    
    # C = A B
    # C_2x3 = A_2x2 B_2x3 + A_2x1 B_1x3
    # C_2x4 = A_2x2 B_2x4 + A_2x1 B_1x4
    # C_1x3 = A_1x2 B_2x3 + A_1x1 B_1x3
    # C_1x4 = A_1x2 B_2x4 + A_1x1 B_1x4
    
    # We can use Strassen for A_2x2 * B_2x3 (2x2 * 2x3).
    # Strassen for 2x2 * 2xn:
    # Let M = A_2x2, N = B_2x3 (2x3).
    # N = [ N_2x2 | N_2x1 ]
    # M N_2x2 uses 7 multiplications (Strassen).
    # M N_2x1 uses 7 multiplications? No, 2x2 * 2x1 is just 4 mults schoolbook, or can be optimized?
    # Actually, 2x2 * 2x1 is rank 4.
    # So A_2x2 B_2x3 costs 7 + 4 = 11 multiplications.
    
    # Similarly A_2x2 B_2x4:
    # B_2x4 = [ B_2x2 | B_2x2 ] (two 2x2 blocks).
    # Each 2x2 * 2x2 costs 7. Total 14.
    
    # Now the other terms:
    # A_2x1 B_1x3: 2x1 * 1x3. This is outer product. Rank 2? 
    # (2x1) * (1x3) = 2x3. Each entry is sum of 1 product. Total 2*3=6 products? 
    # Wait, (2x1) * (1x3) is just scaling rows. 
    # A_2x1 = [a; b], B_1x3 = [c d e].
    # Result = [ac ad ae; bc bd be]. 
    # This can be done with 6 multiplications.
    # Can we share? No, they are distinct entries.
    # So cost 6.
    
    # A_2x1 B_1x4: 2x1 * 1x4. Cost 2*4 = 8.
    
    # A_1x2 B_2x3: 1x2 * 2x3. Cost 1*2*3 = 6.
    # A_1x2 B_2x4: 1x2 * 2x4. Cost 1*2*4 = 8.
    
    # A_1x1 B_1x3: 1x1 * 1x3. Cost 3.
    # A_1x1 B_1x4: 1x1 * 1x4. Cost 4.
    
    # Total cost:
    # A_2x2 B_2x3: 11
    # A_2x2 B_2x4: 14
    # A_2x1 B_1x3: 6
    # A_2x1 B_1x4: 8
    # A_1x2 B_2x3: 6
    # A_1x2 B_2x4: 8
    # A_1x1 B_1x3: 3
    # A_1x1 B_1x4: 4
    # Sum = 11 + 14 + 6 + 8 + 6 + 8 + 3 + 4 = 60.
    
    # This is better than 63.
    # Can we do better?
    # The 2x1 * 1x3 term (rank 6) and 1x2 * 2x3 term (rank 6) are schoolbook.
    # Maybe we can combine some of these?
    # The directive suggests: "viewing the 3x3 matrix as a 3x1 block of 1x3 row vectors and the 3x7 matrix as a 3x3 block of 1x3 column vectors"
    # This implies a different block structure.
    
    # Let's try to implement the rank 60 decomposition first.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
        
    def add_term_full(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= rank:
            return
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(v_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(w_vec):
            if val != 0:
                W[i][col] += val
        col += 1

    # Indices:
    # A is 3x3. Indices (i, j) where i,j in 0..2.
    # B is 3x7. Indices (i, k) where i in 0..2, k in 0..6.
    # C is 3x7. Indices (i, k) where i in 0..2, k in 0..6.
    # u_idx(i, j) = i * 3 + j
    # v_idx(i, k) = i * 7 + k
    # w_idx(k, i) = k * 3 + i  (Note: W index convention is W[k*n+i, r] for C[i,k]?)
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # C[i,k] = sum_r U[i*m+j, r] A[i,j] V[j*p+k, r] B[j,k] W[k*n+i, r]
    # So W index is k*n + i.
    
    # Block definitions:
    # A_2x2: rows 0,1; cols 0,1.
    # A_2x1: rows 0,1; col 2.
    # A_1x2: row 2; cols 0,1.
    # A_1x1: row 2; col 2.
    
    # B_2x3: rows 0,1; cols 0,1,2.
    # B_2x4: rows 0,1; cols 3,4,5,6.
    # B_1x3: row 2; cols 0,1,2.
    # B_1x4: row 2; cols 3,4,5,6.
    
    # 1. A_2x2 * B_2x3 (2x2 * 2x3)
    # B_2x3 = [ B_2x2 | B_2x1 ]
    # B_2x2: rows 0,1; cols 0,1.
    # B_2x1: rows 0,1; col 2.
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the 2x2 blocks of A_2x2:
    # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B_2x2: B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    
    # We need to add rank-1 terms for each multiplication.
    # Each Strassen term involves sums of A and B entries.
    # This creates a tensor decomposition where U and V vectors have multiple non-zero entries.
    
    # Helper to add a Strassen-like term for 2x2*2x2
    def add_strassen_2x2(a_offsets, b_offsets, c_offsets):
        # a_offsets: list of (i,j) for A
        # b_offsets: list of (i,k) for B
        # c_offsets: list of (i,k) for C
        # This is not quite right. Strassen combines multiple products.
        # We need to add 7 rank-1 terms, but each "multiplication" is a sum of products.
        # So each of the 7 terms is a sum of up to 4 rank-1 terms?
        # No, (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is 4 rank-1 terms.
        # So total rank would be 7 * 4 = 28 for 2x2*2x2?
        # That's worse than schoolbook (16).
        # Strassen saves multiplications, but if we expand to rank-1 terms, the rank increases?
        # No, the rank of the tensor is the number of rank-1 terms.
        # If we use Strassen, we replace 8 scalar multiplications with 7 "super-multiplications".
        # But each super-multiplication is a bilinear form.
        # The rank of the 2x2*2x2 tensor is 7.
        # So we can add 7 rank-1 terms to the global tensor.
        # Each of these 7 terms corresponds to one of the M1..M7.
        # The U, V, W vectors for each term will have multiple non-zeros.
        
        pass # We need to implement this properly.

    # Let's implement the 7 terms explicitly.
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # U vector: non-zero at A00, A11.
    # V vector: non-zero at B00, B11.
    # W vector: non-zero at C00, C01, C10, C11.
    
    # We will create a function to add a term with sparse U, V, W.
    
    def add_sparse_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    # Indices helper
    def ui(i, j): return i * 3 + j
    def vi(i, k): return i * 7 + k
    def wi(k, i): return k * 3 + i
    
    # 1. A_2x2 * B_2x2 part (contributes to C_2x2, i.e., C[0:2, 0:2])
    # A00=ui(0,0), A01=ui(0,1), A10=ui(1,0), A11=ui(1,1)
    # B00=vi(0,0), B01=vi(0,1), B10=vi(1,0), B11=vi(1,1)
    # C00=wi(0,0), C01=wi(1,0), C10=wi(0,1), C11=wi(1,1)
    
    # M1
    add_sparse_term({ui(0,0):1, ui(1,1):1}, {vi(0,0):1, vi(1,1):1}, {wi(0,0):1, wi(1,0):1, wi(0,1):1, wi(1,1):1})
    # M2
    add_sparse_term({ui(1,0):1, ui(1,1):1}, {vi(0,0):1}, {wi(0,1):-1, wi(1,1):1}) # Wait, C10=M2+M4, C11=M1-M2+M3+M6
    # Let's check Strassen formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M2 = (A10 + A11)B00
    # Contribution to C10: +1, C11: -1
    add_sparse_term({ui(1,0):1, ui(1,1):1}, {vi(0,0):1}, {wi(0,1):1, wi(1,1):-1})
    
    # M3 = A00(B01 - B11)
    # Contribution to C01: +1, C11: +1
    add_sparse_term({ui(0,0):1}, {vi(0,1):1, vi(1,1):-1}, {wi(1,0):1, wi(1,1):1})
    
    # M4 = A11(B10 - B00)
    # Contribution to C00: +1, C10: +1
    add_sparse_term({ui(1,1):1}, {vi(1,0):1, vi(0,0):-1}, {wi(0,0):1, wi(0,1):1})
    
    # M5 = (A00 + A01)B11
    # Contribution to C00: -1, C01: +1
    add_sparse_term({ui(0,0):1, ui(0,1):1}, {vi(1,1):1}, {wi(0,0):-1, wi(1,0):1})
    
    # M6 = (A10 - A00)(B00 + B01)
    # Contribution to C11: +1
    add_sparse_term({ui(1,0):1, ui(0,0):-1}, {vi(0,0):1, vi(0,1):1}, {wi(1,1):1})
    
    # M7 = (A01 - A11)(B10 + B11)
    # Contribution to C00: +1
    add_sparse_term({ui(0,1):1, ui(1,1):-1}, {vi(1,0):1, vi(1,1):1}, {wi(0,0):1})
    
    # This accounts for 7 terms for A_2x2 * B_2x2.
    # Current col = 7.
    
    # 2. A_2x2 * B_2x1 part (contributes to C_2x1, i.e., C[0:2, 2])
    # B_2x1 is column 2 of B, rows 0,1.
    # B20=vi(0,2), B21=vi(1,2).
    # This is 2x2 * 2x1.
    # C20 = A00 B20 + A01 B21
    # C21 = A10 B20 + A11 B21
    # Schoolbook is 4 terms.
    # Can we do better? Rank of 2x2*2x1 is 4.
    add_sparse_term({ui(0,0):1}, {vi(0,2):1}, {wi(2,0):1})
    add_sparse_term({ui(0,1):1}, {vi(1,2):1}, {wi(2,0):1})
    add_sparse_term({ui(1,0):1}, {vi(0,2):1}, {wi(2,1):1})
    add_sparse_term({ui(1,1):1}, {vi(1,2):1}, {wi(2,1):1})
    # Current col = 7 + 4 = 11.
    
    # 3. A_2x2 * B_2x4 part (contributes to C_2x4, i.e., C[0:2, 3:7])
    # B_2x4 is rows 0,1; cols 3,4,5,6.
    # Split into two 2x2 blocks: B_2x2_a (cols 3,4) and B_2x2_b (cols 5,6).
    # Each 2x2*2x2 costs 7 terms.
    # Total 14 terms.
    
    # Block A (cols 3,4)
    # B00=vi(0,3), B01=vi(0,4), B10=vi(1,3), B11=vi(1,4)
    # C00=wi(3,0), C01=wi(4,0), C10=wi(3,1), C11=wi(4,1)
    add_sparse_term({ui(0,0):1, ui(1,1):1}, {vi(0,3):1, vi(1,4):1}, {wi(3,0):1, wi(4,0):1, wi(3,1):1, wi(4,1):1})
    add_sparse_term({ui(1,0):1, ui(1,1):1}, {vi(0,3):1}, {wi(3,1):1, wi(4,1):-1})
    add_sparse_term({ui(0,0):1}, {vi(0,4):1, vi(1,4):-1}, {wi(4,0):1, wi(4,1):1})
    add_sparse_term({ui(1,1):1}, {vi(1,3):1, vi(0,3):-1}, {wi(3,0):1, wi(3,1):1})
    add_sparse_term({ui(0,0):1, ui(0,1):1}, {vi(1,4):1}, {wi(3,0):-1, wi(4,0):1})
    add_sparse_term({ui(1,0):1, ui(0,0):-1}, {vi(0,3):1, vi(0,4):1}, {wi(4,1):1})
    add_sparse_term({ui(0,1):1, ui(1,1):-1}, {vi(1,3):1, vi(1,4):1}, {wi(3,0):1})
    
    # Block B (cols 5,6)
    # B00=vi(0,5), B01=vi(0,6), B10=vi(1,5), B11=vi(1,6)
    # C00=wi(5,0), C01=wi(6,0), C10=wi(5,1), C11=wi(6,1)
    add_sparse_term({ui(0,0):1, ui(1,1):1}, {vi(0,5):1, vi(1,6):1}, {wi(5,0):1, wi(6,0):1, wi(5,1):1, wi(6,1):1})
    add_sparse_term({ui(1,0):1, ui(1,1):1}, {vi(0,5):1}, {wi(5,1):1, wi(6,1):-1})
    add_sparse_term({ui(0,0):1}, {vi(0,6):1, vi(1,6):-1}, {wi(6,0):1, wi(6,1):1})
    add_sparse_term({ui(1,1):1}, {vi(1,5):1, vi(0,5):-1}, {wi(5,0):1, wi(5,1):1})
    add_sparse_term({ui(0,0):1, ui(0,1):1}, {vi(1,6):1}, {wi(5,0):-1, wi(6,0):1})
    add_sparse_term({ui(1,0):1, ui(0,0):-1}, {vi(0,5):1, vi(0,6):1}, {wi(6,1):1})
    add_sparse_term({ui(0,1):1, ui(1,1):-1}, {vi(1,5):1, vi(1,6):1}, {wi(5,0):1})
    
    # Current col = 11 + 14 = 25.
    
    # 4. A_2x1 * B_1x3 part (contributes to C_2x3, i.e., C[0:2, 0:3])
    # A_2x1: rows 0,1; col 2. A20=ui(0,2), A21=ui(1,2).
    # B_1x3: row 2; cols 0,1,2. B30=vi(2,0), B31=vi(2,1), B32=vi(2,2).
    # C30 = A20 B30
    # C31 = A20 B31
    # C32 = A20 B32
    # C40 = A21 B30
    # C41 = A21 B31
    # C42 = A21 B32
    # 6 terms.
    for i in range(2):
        for k in range(3):
            add_sparse_term({ui(i,2):1}, {vi(2,k):1}, {wi(k,i):1})
            
    # Current col = 25 + 6 = 31.
    
    # 5. A_2x1 * B_1x4 part (contributes to C_2x4, i.e., C[0:2, 3:7])
    # 8 terms.
    for i in range(2):
        for k in range(3, 7):
            add_sparse_term({ui(i,2):1}, {vi(2,k):1}, {wi(k,i):1})
            
    # Current col = 31 + 8 = 39.
    
    # 6. A_1x2 * B_2x3 part (contributes to C_1x3, i.e., C[2, 0:3])
    # A_1x2: row 2; cols 0,1. A30=ui(2,0), A31=ui(2,1).
    # B_2x3: rows 0,1; cols 0,1,2.
    # C50 = A30 B00 + A31 B10
    # C51 = A30 B01 + A31 B11
    # C52 = A30 B02 + A31 B12
    # 6 terms.
    for j in range(2):
        for k in range(3):
            add_sparse_term({ui(2,j):1}, {vi(j,k):1}, {wi(k,2):1})
            
    # Current col = 39 + 6 = 45.
    
    # 7. A_1x2 * B_2x4 part (contributes to C_1x4, i.e., C[2, 3:7])
    # 8 terms.
    for j in range(2):
        for k in range(3, 7):
            add_sparse_term({ui(2,j):1}, {vi(j,k):1}, {wi(k,2):1})
            
    # Current col = 45 + 8 = 53.
    
    # 8. A_1x1 * B_1x3 part (contributes to C_1x3, i.e., C[2, 0:3])
    # A_1x1: A32=ui(2,2).
    # B_1x3: B30, B31, B32.
    # 3 terms.
    for k in range(3):
        add_sparse_term({ui(2,2):1}, {vi(2,k):1}, {wi(k,2):1})
        
    # Current col = 53 + 3 = 56.
    
    # 9. A_1x1 * B_1x4 part (contributes to C_1x4, i.e., C[2, 3:7])
    # 4 terms.
    for k in range(3, 7):
        add_sparse_term({ui(2,2):1}, {vi(2,k):1}, {wi(k,2):1})
        
    # Current col = 56 + 4 = 60.
    
    # Total rank used: 60.
    
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook 63
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

    return {"U": U, "V": V, "W": W}
