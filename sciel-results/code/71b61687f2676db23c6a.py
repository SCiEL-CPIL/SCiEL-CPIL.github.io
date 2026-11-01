"""Strassen-based block decomposition for 2x3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # For each 2x3x2 block, we use rank 11:
    # 7 Strassen terms + 4 extra diagonal terms
    # The 11 terms are indexed by (U_idx, V_idx, W_idx, coeff)
    # U_idx = i*m + j (0..5), V_idx = j*p_sub + k (0..5), W_idx = k*n + i (0..3)
    # where p_sub = 2 for the sub-problem
    
    # Strassen 2x2x2 terms (rank 7) + 4 diagonal terms for the remaining products
    # For 2x2x2, the schoolbook is rank 8. Strassen gives rank 7.
    # But we need 2x3x2, which has 12 products. 
    # Let me think more carefully.
    
    # Actually, let me reconsider. The 2x3x2 multiplication has 12 scalar products.
    # A block decomposition using Strassen for 2x2x2 sub-blocks:
    # A(2x3) * B(3x2) = C(2x2)
    # Split B into B1(3x1) and B2(3x1):
    # C[:,0] = A*B1, C[:,1] = A*B2
    # Each A*B_i is a 2x3 * 3x1 = 2x1 product, which is 6 scalar mults.
    # Total: 12 scalar mults schoolbook.
    
    # Alternative: Use the known rank-11 for 2x3x2.
    # Let me construct it explicitly.
    
    # The 12 products are:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Strassen for 2x2: M1..M7
    # Let me use a known rank-11 decomposition for 2x3x2.
    # 
    # Actually, let me just use a direct approach with the known structure.
    # The key insight from the knowledge: rank 11 for 2x3x2.
    # 
    # Let me define the 11 terms as (u_row, v_row, w_row, coeff)
    # where u_row is the index in U (0..5), v_row in V (0..5), w_row in W (0..3)
    
    # I'll use a verified rank-11 decomposition for 2x3x2:
    # Terms (u, v, w, c):
    # Strassen-like for the 2x2 blocks plus corrections
    
    # Let me think about this differently. For 2x3x2:
    # We can view it as two 2x3x1 problems (for each column of B).
    # Or use a tensor decomposition approach.
    
    # Known: the tensor rank of 2x3x2 is 11 (vs 12 schoolbook).
    # Let me construct it explicitly.
    
    # U has 6 rows (i*m+j for i in 0..1, j in 0..2)
    # V has 6 rows (j*p_sub+k for j in 0..2, k in 0..1)  
    # W has 4 rows (k*n+i for k in 0..1, i in 0..1)
    
    # I'll use the following 11-term decomposition:
    # This is a specific construction. Let me verify it works.
    
    terms = []
    # Using a systematic approach:
    # M1 = (A00+A11)(B00+B22) - (A00+B11)(B00+B22) ... no this is getting complex
    
    # Let me use a simpler approach: just enumerate a known valid rank-11.
    # From literature, for 2x3x2, one rank-11 decomposition uses:
    
    # I'll use the following verified terms (u_idx, v_idx, w_idx, coeff):
    # u_idx: row in U (i*3+j), v_idx: row in V (j*2+k), w_idx: row in W (k*2+i)
    
    # Let me just use schoolbook for now (rank 12 per block, total 36) 
    # and see if I can improve. Actually the knowledge says rank 33 works.
    
    # Let me try a different approach: use rank 11 per block.
    # Here's a construction:
    
    # For 2x3x2, label A[i,j] for i in {0,1}, j in {0,1,2}
    # B[j,k] for j in {0,1,2}, k in {0,1}
    # C[i,k] for i in {0,1}, k in {0,1}
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # 11-term decomposition:
    # t1: A[0,0]*B[0,0] -> U(0,3), V(0,2), W(0,2) [i=0,j=0; j=0,k=0; k=0,i=0]
    # Wait, let me be more careful with indices.
    # U row = i*m + j = i*3 + j
    # V row = j*p_sub + k = j*2 + k
    # W row = k*n + i = k*2 + i
    
    # Schoolbook 12 terms:
    schoolbook_terms = []
    for i in range(2):
        for j in range(3):
            for k in range(2):
                schoolbook_terms.append((i*3+j, j*2+k, k*2+i, 1))
    
    # That's 12 terms. We need 11.
    # Let me try to merge two terms using a Strassen-like trick.
    # 
    # Consider combining products that share structure.
    # A[0,0]*B[0,0] + A[1,1]*B[1,1] can be done with 3 mults:
    #   (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) - A[0,0]*B[0,0] - A[1,1]*B[1,1]
    # That doesn't save. Strassen uses:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B22-B11)
    #   M4 = A22(B11-B12)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    
    # For our case, let me just use the 12-term schoolbook for each block.
    # Total rank = 3 * 12 = 36. Same as original.
    # 
    # But the knowledge says rank 33 works. Let me try to find a rank-11 for 2x3x2.
    
    # Actually, let me try a hybrid: use Strassen on overlapping 2x2 blocks.
    # A is 2x3, B is 3x2.
    # View as:
    # A = [A_left (2x2) | A_right (2x1)]
    # B = [B_top (2x2); B_bottom (1x2)]
    
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top: 2x2 * 2x2, rank 7 (Strassen)
    # A_right * B_bottom: 2x1 * 1x2, this is outer product, rank 4 (or less?)
    #   A_right = [a; b], B_bottom = [c, d]
    #   Result = [[ac, ad], [bc, bd]]
    #   This needs 4 scalar mults: ac, ad, bc, bd
    #   But we can do: (a+b)*(c+d) - a*c - b*d gives ac+ad+bc+bd... not quite.
    #   Actually outer product of 2x1 and 1x2 has rank 4 (it's the tensor product).
    #   So total: 7 + 4 = 11. 
    
    # Let me construct this.
    # A_left = A[:, 0:2] (columns 0,1), A_right = A[:, 2] (column 2)
    # B_top = B[0:2, :] (rows 0,1), B_bottom = B[2, :] (row 2)
    
    # Strassen for A_left * B_top:
    # A_left = [[A00, A01], [A10, A11]], B_top = [[B00, B01], [B10, B11]]
    # Standard Strassen indices: A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    # M1 = (A00+A11)(B00+B10)  # note: Strassen uses B22-B11 etc, let me use standard
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B11)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # In our notation:
    # A11=A[0,0]=A00, A12=A[0,1]=A01, A21=A[1,0]=A10, A22=A[1,1]=A11
    # B11=B[0,0]=B00, B12=B[0,1]=B01, B21=B[1,0]=B10, B22=B[1,1]=B11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C_top = M1 - M3 - M4 - M7  (for [0,0])
    #        M5 + M7              (for [0,1])
    #        M2 + M4              (for [1,0])
    #        M1 - M2 - M3 - M6    (for [1,1])
    
    # Now for A_right * B_bottom:
    # A_right = [A02, A12], B_bottom = [B20, B21]
    # Result = [[A02*B20, A02*B21], [A12*B20, A12*B21]]
    # 4 mults: A02*B20, A02*B21, A12*B20, A12*B21
    
    # Total: 7 + 4 = 11 mults. 
    
    # Now I need to express these as tensor terms.
    # Each mult corresponds to a term (U_row, V_row, W_row, coeff)
    
    # For the 4 outer product terms:
    # A02*B20: U row for A[0,2] = 0*3+2 = 2, V row for B[2,0] = 2*2+0 = 4, W row for C[0,0] = 0*2+0 = 0
    # A02*B21: U=2, V=2*2+1=5, W=1*2+0=2
    # A12*B20: U=1*3+2=5, V=4, W=0
    # A12*B21: U=5, V=5, W=2
    
    # For Strassen terms, I need to express sums/differences.
    # Each Strassen mult M_i involves a sum of A entries and a sum of B entries.
    # I need to express each product as a single term in the tensor decomposition.
    # This means I need to find U, V, W rows such that the product gives the right result.
    
    # Actually, in the tensor decomposition, each term is u_row * v_row * w_row (outer products).
    # So each "mult" in Strassen corresponds to one term where U_row has the combined A coefficient,
    # V_row has the combined B coefficient, and W_row distributes to the right C entries.
    
    # Let me construct this explicitly.
    
    rank_per_block = 11
    total_rank = 3 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(3):
        col_offset = block * rank_per_block
        # For this block, k_local in {0,1} maps to k_global = k_local + 2*block
        
        # Outer product terms (4 terms):
        # A02*B20 -> C[0,0]
        U[2][col_offset + 0] += 1
        V[4][col_offset + 0] += 1
        W[0][col_offset + 0] += 1
        
        # A02*B21 -> C[0,1]
        U[2][col_offset + 1] += 1
        V[5][col_offset + 1] += 1
        W[2][col_offset + 1] += 1
        
        # A12*B20 -> C[1,0]
        U[5][col_offset + 2] += 1
        V[4][col_offset + 2] += 1
        W[1][col_offset + 2] += 1
        
        # A12*B21 -> C[1,1]
        U[5][col_offset + 3] += 1
        V[5][col_offset + 3] += 1
        W[3][col_offset + 3] += 1
        
        # Strassen terms (7 terms):
        # M1 = (A00+A11)(B00+B11)
        # A00: U[0], A11: U[4] -> need U row with 1 at 0 and 1 at 4
        # B00: V[0], B11: V[3] -> need V row with 1 at 0 and 1 at 3
        # C: M1 contributes to C[0,0] with -1, C[0,1] with +1, C[1,0] with +1, C[1,1] with -1
        # W[0] (C[0,0]) = -1, W[2] (C[0,1]) = +1, W[1] (C[1,0]) = +1, W[3] (C[1,1]) = -1
        
        U[0][col_offset + 4] += 1
        U[4][col_offset + 4] += 1
        V[0][col_offset + 4] += 1
        V[3][col_offset + 4] += 1
        W[0][col_offset + 4] -= 1
        W[2][col_offset + 4] += 1
        W[1][col_offset + 4] += 1
        W[3][col_offset + 4] -= 1
        
        # M2 = (A10+A11)B00
        U[3][col_offset + 5] += 1
        U[4][col_offset + 5] += 1
        V[0][col_offset + 5] += 1
        W[1][col_offset + 5] += 1
        W[3][col_offset + 5] -= 1
        
        # M3 = A00(B11-B00)
        U[0][col_offset + 6] += 1
        V[3][col_offset + 6] += 1
        V[0][col_offset + 6] -= 1
        W[0][col_offset + 6] -= 1
        W[3][col_offset + 6] -= 1
        
        # M4 = A11(B00-B01)
        U[4][col_offset + 7] += 1
        V[0][col_offset + 7] += 1
        V[1][col_offset + 7] -= 1
        W[0][col_offset + 7] -= 1
        W[1][col_offset + 7] += 1
        
        # M5 = (A00+A01)B11
        U[0][col_offset + 8] += 1
        U[1][col_offset + 8] += 1
        V[3][col_offset + 8] += 1
        W[2][col_offset + 8] += 1
        W[1][col_offset + 8] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[3][col_offset + 9] += 1
        U[0][col_offset + 9] -= 1
        V[0][col_offset + 9] += 1
        V[1][col_offset + 9] += 1
        W[3][col_offset + 9] -= 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][col_offset + 10] += 1
        U[4][col_offset + 10] -= 1
        V[2][col_offset + 10] += 1
        V[3][col_offset + 10] += 1
        W[0][col_offset + 10] -= 1
        W[2][col_offset + 10] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
