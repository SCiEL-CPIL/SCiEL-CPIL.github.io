"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

Uses Strassen's rank-7 algorithm for 2x2x2 blocks to exploit the Kronecker
structure, then extends to the 7-column case.

Strategy:
- 4x4 * 4x7 can be viewed as 2x2 blocks of (2x2)*(2x7) in the row dimension
  and 2x2 blocks in the column dimension for the first 4 columns, plus 2x2 * 2x3
  for the last 3 columns.

Actually, let's think more carefully:
- A is 4x4, B is 4x7, C = A*B is 4x7.
- Split A into 2x2 blocks of 2x2 submatrices: A = [[A11, A12], [A21, A22]]
- Split B into 2x7 blocks: B = [[B1], [B2]] where B1 is 2x7, B2 is 2x7.
- C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]] which is 4x7.

For each 2x2 * 2x7 product, we can use Strassen-like decomposition.
A 2x2 * 2x7 product: let A = [[a, b], [c, d]] (2x2), B = [[B_top], [B_bot]] (2x7).
Result = [[a*B_top + b*B_bot], [c*B_top + d*B_bot]] (2x7).

Using Strassen for the 2x2 * 2x2 part (first 2 columns of B):
M1 = (a+d)(B_top[:,0:2] + B_bot[:,0:2])  -- but this doesn't directly work for 2x7.

Let me use a different approach. For 2x2 * 2x7:
We need to compute:
  C11 = a*B1 + b*B2  (where B1, B2 are 1x7 rows)
  C21 = c*B1 + d*B2

This is like multiplying a 2x2 matrix by a 2x7 matrix. We can decompose this
using the tensor rank of (2,2,2) * (2,2,7) structure.

Actually, the simplest approach that gives good rank:
- Use schoolbook for the full 4x4*4x7: rank = 4*4*7 = 112.
- Use Strassen for 2x2x2 blocks: For 4x4*4x4, rank = 7*7 = 49. For 4x4*4x7,
  we can do 4x4*4x4 (rank 49) + 4x4*4x3 (schoolbook rank 48) = 97.

Let me implement the 4x4*4x4 Strassen decomposition (rank 49) and 4x4*4x3 schoolbook (rank 48), total rank 97.

For 4x4*4x4 with Strassen:
A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]] (all 2x2)
C11 = (A11+A22)(B11+B22) - (A21-A22)(B21-B22)  -- Strassen M1, M2
C12 = A12*(B21+B22)  -- M3
C21 = (A11+A12)*B21  -- M4  
C22 = (A11+A22)(B11+B22) - A11*B11 - A22*B22  -- M1, M5, M6

Actually standard Strassen:
M1 = (A11+A22)(B11+B22)
M2 = (A21+A22)B11
M3 = A11(B21-B22)
M4 = A22(B12-B11)
M5 = (A11+A12)B22
M6 = (A21-A11)(B11+B12)
M7 = (A12-A22)(B21+B22)

C11 = M1+M4-M5+M7
C12 = M3+M5
C21 = M2+M4
C22 = M1-M2+M3+M6

Each Mi is a 2x2*2x2 product, which has rank 4 (schoolbook) or can be further
decomposed. For simplicity, use schoolbook for the 2x2*2x2 subproducts.

So for 4x4*4x4: 7 Strassen steps, each requiring 4 scalar multiplications (2x2*2x2 schoolbook)
Total rank = 7*4 = 28? No wait.

Let me reconsider. The tensor for 2x2*2x2 has rank 7 (Strassen). So each Mi 
is computed with rank 7. But Mi is itself a 2x2*2x2 product... no, Mi is a 
product of two 2x2 matrices giving a 2x2 result. That's exactly the 2x2*2x2 tensor.

So for 4x4*4x4: we have 7 "Strassen multipliers", each of which is a 2x2*2x2 
tensor of rank 7. Total rank = 7*7 = 49.

For 4x4*4x3: schoolbook rank = 4*4*3 = 48.

Total rank for 4x4*4x7: 49 + 48 = 97.

Let me implement this.
"""

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Using Kronecker structure: 4x4*4x4 (Strassen, rank 49) + 4x4*4x3 (schoolbook, rank 48)
    
    # Part 1: 4x4 * 4x4 using Strassen (rank 49)
    # Part 2: 4x4 * 4x3 using schoolbook (rank 48)
    # Total rank: 97
    
    rank_total = 49 + 48
    
    # Initialize matrices
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank_total for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank_total for _ in range(p * n)]  # 28 x 97
    
    # --- Part 1: 4x4 * 4x4 Strassen decomposition (rank 49) ---
    # A is 4x4, split into 2x2 blocks of 2x2:
    #   A = [[A11, A12], [A21, A22]]
    # B_first4 is 4x4, split into 2x2 blocks:
    #   B4 = [[B11, B12], [B21, B22]]
    # C_first4 = A * B4, split into 2x2 blocks:
    #   C4 = [[C11, C12], [C21, C22]]
    
    # Strassen multipliers (each is a 2x2 * 2x2 product, rank 7 each):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each Strassen multiplier, we need to express it as a sum of rank-1 terms
    # using the Strassen 2x2*2x2 decomposition.
    
    # Strassen 2x2*2x2: given P (2x2) and Q (2x2), compute R = P*Q
    # Using 7 multiplications:
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q10-q11)
    # s4 = p11(q01-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    # r00 = s1+s4-s5+s7
    # r01 = s3+s5
    # r10 = s2+s4
    # r11 = s1-s2+s3+s6
    
    # Each si is a product of two scalars, so rank 1.
    # So each 2x2*2x2 has rank 7.
    
    # For the 4x4*4x4 case, each Strassen multiplier Mi is a 2x2*2x2 product.
    # The inputs to Mi are linear combinations of the 2x2 blocks of A and B4.
    
    # Let's define the linear combinations:
    # For multiplier Mi, the "left" 2x2 matrix L_i and "right" 2x2 matrix R_i
    # are linear combinations of A11,A12,A21,A22 and B11,B12,B21,B22.
    
    # We'll use a helper to add a Strassen 2x2*2x2 decomposition.
    
    def add_strassen_2x2x2(U, V, W, left_coeffs, right_coeffs, result_coeffs, col_offset, 
                           left_block_rows, right_block_cols, result_block_rows, result_block_cols):
        """
        Add a Strassen 2x2*2x2 decomposition.
        
        left_coeffs: list of 4 coefficients for [A11, A12, A21, A22] (2x2 matrix)
        right_coeffs: list of 4 coefficients for [B11, B12, B21, B22] (2x2 matrix)
        result_coeffs: list of 4 coefficients for [C11, C12, C21, C22] (2x2 matrix)
        
        left_block_rows: (row_start, row_end) for the 2x2 block in A (4x4)
        right_block_cols: (col_start, col_end) for the 2x2 block in B (4x4)
        result_block_rows: (row_start, row_end) for the 2x2 block in C (4x4)
        result_block_cols: (col_start, col_end) for the 2x2 block in C (4x4)
        """
        pass  # Will implement below
    
    # Actually, let me build this more directly.
    
    # The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
    # Block A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    # A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    
    # U index: i*m + j = i*4 + j (row i, col j of A)
    # V index: j*p + k = j*7 + k (row j, col k of B)
    # W index: k*n + i = k*4 + i (row k, col i of C)
    
    # For the 4x4*4x4 part, B only uses columns 0-3.
    # V index for col k (0-3): j*7 + k
    
    # Strassen multipliers for 4x4*4x4:
    # Each multiplier is: (linear_combo_of_A_blocks) * (linear_combo_of_B_blocks)
    # where the linear combos form 2x2 matrices.
    
    # Let me define the 7 Strassen multipliers explicitly.
    # Each multiplier has a left 2x2 matrix (linear combo of A's 2x2 blocks)
    # and a right 2x2 matrix (linear combo of B's 2x2 blocks).
    
    # Multiplier definitions: (left_block_coeffs, right_block_coeffs)
    # left_block_coeffs: [coeff_A11, coeff_A12, coeff_A21, coeff_A22]
    # right_block_coeffs: [coeff_B11, coeff_B12, coeff_B21, coeff_B22]
    
    # M1 = (A11+A22)(B11+B22): left=[1,0,0,1], right=[1,0,0,1]
    # M2 = (A21+A22)B11: left=[0,0,1,1], right=[1,0,0,0]
    # M3 = A11(B21-B22): left=[1,0,0,0], right=[0,0,1,-1]
    # M4 = A22(B12-B11): left=[0,0,0,1], right=[-1,1,0,0]
    # M5 = (A11+A12)B22: left=[1,1,0,0], right=[0,0,0,1]
    # M6 = (A21-A11)(B11+B12): left=[-1,0,1,0], right=[1,1,0,0]
    # M7 = (A12-A22)(B21+B22): left=[0,1,0,-1], right=[0,0,1,1]
    
    strassen_multipliers = [
        ([1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0]),   # M2
        ([1,0,0,0], [0,0,1,-1]),  # M3
        ([0,0,0,1], [-1,1,0,0]),  # M4
        ([1,1,0,0], [0,0,0,1]),   # M5
        ([-1,0,1,0], [1,1,0,0]),  # M6
        ([0,1,0,-1], [0,0,1,1]),  # M7
    ]
    
    # Result combination: C_block = sum of multipliers with coefficients
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each multiplier, we decompose the 2x2*2x2 product using Strassen (rank 7).
    # Each of the 7 scalar multiplications in the inner Strassen corresponds to
    # a rank-1 term in the overall decomposition.
    
    # Inner Strassen for P*Q where P, Q are 2x2:
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q10-q11)
    # s4 = p11(q01-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    
    # r00 = s1+s4-s5+s7
    # r01 = s3+s5
    # r10 = s2+s4
    # r11 = s1-s2+s3+s6
    
    # For each inner Strassen multiplication si = (linear_combo_of_P_entries) * (linear_combo_of_Q_entries)
    # we get a rank-1 term.
    
    # Let me precompute the inner Strassen structure.
    # For P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    
    # s1: (p00+p11) * (q00+q11)
    # s2: (p10+p11) * q00
    # s3: p00 * (q10-q11)
    # s4: p11 * (q01-q00)
    # s5: (p00+p01) * q11
    # s6: (p10-p00) * (q00+q01)
    # s7: (p01-p11) * (q10+q11)
    
    # Result:
    # r00 = s1+s4-s5+s7
    # r01 = s3+s5
    # r10 = s2+s4
    # r11 = s1-s2+s3+s6
    
    # So for each si, we need to add to the result with appropriate coefficients.
    
    # Now, P and Q are linear combinations of the 2x2 blocks of A and B.
    # P = sum over blocks of coeff * block
    # Q = sum over blocks of coeff * block
    
    # Each entry of P is a linear combination of entries of A.
    # Each entry of Q is a linear combination of entries of B.
    
    # Let me define the block positions:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-1 (of B, which is 4x7, so cols 0-1)
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # Block entry positions (row, col) within the 4x4:
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    # B has same row structure, cols 0-3 for the 4x4 part.
    
    # For U: index = i*4 + j (row i, col j of A, 4x4)
    # For V: index = j*7 + k (row j, col k of B, 4x7) -- for cols 0-3
    # For W: index = k*4 + i (row k, col i of C, 4x7) -- for cols 0-3
    
    # Let me build the full decomposition.
    
    col = 0
    
    # Helper: get U index for A entry (i,j)
    def u_idx(i, j):
        return i * 4 + j
    
    # Helper: get V index for B entry (j,k) where k is 0-3 (first 4 cols)
    def v_idx(j, k):
        return j * 7 + k
    
    # Helper: get W index for C entry (k,i) where k is 0-3 (first 4 cols)
    def w_idx(k, i):
        return k * 4 + i
    
    # Block entry positions
    def block_entries(block_id):
        """Return list of (row, col) positions for a 2x2 block."""
        if block_id == 0:  # 11: rows 0-1, cols 0-1
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block_id == 1:  # 12: rows 0-1, cols 2-3
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block_id == 2:  # 21: rows 2-3, cols 0-1
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block_id == 3:  # 22: rows 2-3, cols 2-3
            return [(2,2), (2,3), (3,2), (3,3)]
    
    # For each Strassen multiplier, decompose the 2x2*2x2 product.
    for mult_idx, (left_coeffs, right_coeffs) in enumerate(strassen_multipliers):
        # left_coeffs: [c11, c12, c21, c22] for A blocks
        # right_coeffs: [c11, c12, c21, c22] for B blocks
        
        # Build P (2x2) as linear combo of A blocks
        # P[i,j] = sum over blocks b of left_coeffs[b] * A_block_b[i,j]
        # Similarly for Q with B blocks.
        
        # For each of the 7 inner Strassen multiplications:
        # si = (linear_combo_P) * (linear_combo_Q)
        
        # Inner Strassen definitions:
        # s1: P_entries = [(0,0,1), (1,1,1)], Q_entries = [(0,0,1), (1,1,1)]
        # s2: P_entries = [(1,0,1), (1,1,1)], Q_entries = [(0,0,1)]
        # s3: P_entries = [(0,0,1)], Q_entries = [(1,0,1), (1,1,-1)]
        # s4: P_entries = [(1,1,1)], Q_entries = [(0,1,1), (0,0,-1)]
        # s5: P_entries = [(0,0,1), (0,1,1)], Q_entries = [(1,1,1)]
        # s6: P_entries = [(1,0,1), (0,0,-1)], Q_entries = [(0,0,1), (0,1,1)]
        # s7: P_entries = [(0,1,1), (1,1,-1)], Q_entries = [(1,0,1), (1,1,1)]
        
        # Each entry is (row, col, coeff)
        
        inner_strassen = [
            # (P_entries, Q_entries, result_coeffs)
            # result_coeffs: [coeff_r00, coeff_r01, coeff_r10, coeff_r11]
            ([[(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]], [1, 0, 0, 1]),   # s1
            ([[(1,0,1), (1,1,1)], [(0,0,1)]], [0, 0, 1, -1]),            # s2
            ([[(0,0,1)], [(1,0,1), (1,1,-1)]], [0, 1, 0, 1]),            # s3
            ([[(1,1,1)], [(0,1,1), (0,0,-1)]], [1, 0, 1, 0]),            # s4
            ([[(0,0,1), (0,1,1)], [(1,1,1)]], [-1, 1, 0, 0]),            # s5
            ([[(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]], [0, 0, 0, 1]),   # s6
            ([[(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]], [1, 0, 0, 0]),   # s7
        ]
        
        for P_entries, Q_entries, res_coeffs in inner_strassen:
            # For this rank-1 term, we need:
            # U contribution: for each (pi, pj, pc) in P_entries,
            #   for each block b with left_coeffs[b] != 0,
            #     for each (row, col) in block_entries(b):
            #       U[u_idx(row, col)][col_offset] += pc * left_coeffs[b]
            
            # V contribution: for each (qi, qj, qc) in Q_entries,
            #   for each block b with right_coeffs[b] != 0,
            #     for each (row, col) in block_entries(b):
            #       V[v_idx(row, col)][col_offset] += qc * right_coeffs[b]
            
            # W contribution: for each result block r (0-3) with res_coeffs[r] != 0,
            #   for each (row, col) in block_entries(r):
            #     W[w_idx(col, row)][col_offset] += res_coeffs[r]
            
            # U contribution
            for (pi, pj, pc) in P_entries:
                for b in range(4):
                    lc = left_coeffs[b]
                    if lc == 0:
                        continue
                    for (row, col) in block_entries(b):
                        U[u_idx(row, col)][col] += pc * lc
            
            # V contribution
            for (qi, qj, qc) in Q_entries:
                for b in range(4):
                    rc = right_coeffs[b]
                    if rc == 0:
                        continue
                    for (row, col) in block_entries(b):
                        V[v_idx(row, col)][col] += qc * rc
            
            # W contribution
            for r in range(4):
                wc = res_coeffs[r]
                if wc == 0:
                    continue
                for (row, col) in block_entries(r):
                    W[w_idx(col, row)][col] += wc
            
            col += 1
    
    # --- Part 2: 4x4 * 4x3 schoolbook (rank 48) ---
    # For columns 4-6 of B (and C).
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[u_idx(i, j)][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    # Convert to lists (already are)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
