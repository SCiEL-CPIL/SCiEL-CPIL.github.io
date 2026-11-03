"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition in two parts:
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x2 using schoolbook (rank 32)
    # Total rank = 81
    
    U_list = []  # list of (i, j, value) for U part
    V_list = []  # list of (j, k, value) for V part  
    W_list = []  # list of (k, i, value) for W part
    
    # --- Part 1: 4x4x4 Strassen decomposition ---
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we split into 2x2 blocks and apply Strassen recursively.
    # This gives 7*7 = 49 scalar multiplications.
    
    # Let me define the Strassen 2x2x2 decomposition explicitly.
    # Each "multiplication" in Strassen corresponds to a 2x2x2 sub-problem.
    # For the top level (4x4x4), we have 7 Strassen multiplications of 2x2 matrices.
    # Each 2x2x2 is further decomposed into 7 scalar multiplications.
    
    # I'll build this systematically.
    
    # Strassen 2x2x2 terms: each term is a product of a linear combo of A blocks 
    # and a linear combo of B blocks, contributing to a linear combo of C blocks.
    
    # Let me define the full 4x4x4 Strassen decomposition.
    
    # Top-level Strassen: 7 multiplications of 2x2 by 2x2.
    # For each, the A-factor and B-factor are 2x2 linear combinations.
    # Each 2x2x2 multiplication uses Strassen: 7 scalar mults.
    
    # This is getting complex. Let me use a cleaner approach.
    
    # Alternative: directly enumerate the 49 terms.
    
    # Strassen 2x2x2: define the 7 terms as (a_combo, b_combo, c_combo)
    # where a_combo is a 2x2 matrix of coefficients for A, etc.
    
    # For 2x2x2 Strassen:
    # Term 1: A*(1,1;1,0;0,1;0,0)... let me use a different notation.
    
    # Let me just hardcode the 49-term decomposition.
    
    # Actually, let me think of it as: the 4x4x4 tensor = sum of 49 rank-1 tensors.
    # Each rank-1 tensor is U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # I'll generate them programmatically using the Strassen recursion.
    
    # Strassen 2x2x2 decomposition (7 terms):
    # For 2x2x2, indices are (i,j,k) with i,j,k in {0,1}
    # Term t: U_t[i,j], V_t[j,k], W_t[k,i]
    
    strassen_2x2 = [
        # (U, V, W) where each is a 2x2 matrix (flattened as [i*2+j], [j*2+k], [k*2+i])
        # Term 1: (A11+A22)(B11+B22) -> C11+=1, C22+=1
        # A factor: [[1,0],[1,1]], B factor: [[1,1],[1,1]], C: C11+=1, C22+=1
        # U[i,j]: i is row of A, j is col of A (which is row of B)
        # For (A11+A22): U[0,0]=1, U[1,0]=1, U[1,1]=1, U[0,1]=0
        # Wait, I need to be more careful.
        
        # Let me use a cleaner formulation.
        # In 2x2x2: A is 2x2, B is 2x2, C = A*B is 2x2.
        # U[i,j] for i,j in {0,1}: coefficient of A[i,j]
        # V[j,k] for j,k in {0,1}: coefficient of B[j,k]  
        # W[k,i] for k,i in {0,1}: coefficient for C[i,k] (stored as W[k*2+i])
        
        # Strassen terms:
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        #   U: A[0,0]=1, A[1,1]=1 -> U[0,0]=1, U[1,1]=1
        #   V: B[0,0]=1, B[1,1]=1 -> V[0,0]=1, V[1,1]=1
        #   C: C[0,0]+=1, C[1,1]+=1 -> W[0,0]=1, W[1,1]=1 (W[k,i]: W[0*2+0]=W[0,0]=1, W[1*2+1]=W[1,1]=1)
        #
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        #   U: A[1,0]=1, A[1,1]=1
        #   V: B[0,0]=1
        #   C: C[1,0]+=1 -> W[0,1]=1 (W[k=0,i=1])
        #
        # M3 = A[0,0] * (B[1,0]-B[1,1])
        #   U: A[0,0]=1
        #   V: B[1,0]=1, B[1,1]=-1
        #   C: C[0,1]+=1 -> W[1,0]=1 (W[k=1,i=0])
        #
        # M4 = A[1,1] * (B[0,1]-B[0,0])
        #   U: A[1,1]=1
        #   V: B[0,1]=1, B[0,0]=-1
        #   C: C[0,0]+=1, C[1,0]+=1 -> W[0,0]=1, W[0,1]=1
        #
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        #   U: A[0,0]=1, A[0,1]=1
        #   V: B[1,1]=1
        #   C: C[0,1]+=1, C[1,1]+=1 -> W[1,0]=1, W[1,1]=1
        #
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        #   U: A[1,0]=1, A[0,0]=-1
        #   V: B[0,0]=1, B[0,1]=1
        #   C: C[1,1]+=1 -> W[1,1]=1
        #
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        #   U: A[0,1]=1, A[1,1]=-1
        #   V: B[1,0]=1, B[1,1]=1
        #   C: C[0,0]+=1 -> W[0,0]=1
    ]
    
    # Let me define each Strassen 2x2x2 term as (U_2x2, V_2x2, W_2x2)
    # U_2x2[i,j], V_2x2[j,k], W_2x2[k,i]
    
    def strassen_2x2_terms():
        """Return list of 7 tuples (U, V, W) for 2x2x2 Strassen."""
        terms = []
        # Each is (U[2][2], V[2][2], W[2][2]) where W[k][i]
        
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        terms.append(([[1,0],[0,1]], [[1,1],[1,1]], [[1,0],[0,1]]))
        # M2 = (A10+A11)B00 -> C10+=1
        terms.append(([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,0]]))
        # M3 = A00(B10-B11) -> C01+=1
        terms.append(([[1,0],[0,0]], [[0,0],[1,-1]], [[0,0],[1,0]]))
        # M4 = A11(B01-B00) -> C00+=1, C10+=1
        terms.append(([[0,0],[0,1]], [[-1,1],[0,0]], [[1,0],[1,0]]))
        # M5 = (A00+A01)B11 -> C01+=1, C11+=1
        terms.append(([[1,1],[0,0]], [[0,0],[0,1]], [[0,1],[0,1]]))
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        terms.append(([[1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]))
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        terms.append(([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]))
        
        return terms
    
    # Now for 4x4x4: split into 2x2 blocks.
    # A is 4x4, split into A00,A01,A10,A11 each 2x2.
    # B is 4x4, split into B00,B01,B10,B11 each 2x2.
    # C = A*B, split into C00,C01,C10,C11 each 2x2.
    
    # Top-level Strassen: 7 multiplications of 2x2 by 2x2.
    # Each 2x2x2 multiplication is further decomposed using Strassen (7 scalar mults).
    
    # For each top-level Strassen term, we have:
    #   A_factor = sum of 2x2 blocks of A with coefficients
    #   B_factor = sum of 2x2 blocks of B with coefficients
    #   C_factor = sum of 2x2 blocks of C with coefficients
    
    # Then each 2x2x2 (A_factor * B_factor -> C_factor) is decomposed into 7 scalar mults.
    
    # Total: 7 * 7 = 49 scalar mults.
    
    # Let me define the top-level Strassen for 4x4x4 in terms of 2x2 block operations.
    
    # Top-level: same as 2x2x2 Strassen but with 2x2 blocks.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mi, it's a 2x2x2 multiplication. Decompose each into 7 scalar mults.
    
    # I'll build the full 4x4x4 decomposition by expanding.
    
    # For a top-level term with A_factor coefficients a[bi][bj] (bi,bj in {0,1} for block indices)
    # and B_factor coefficients b[bj][bk] and C_factor coefficients c[bk][bi] (for C block [bi,bk])
    
    # Wait, I need to be careful with the block indexing.
    
    # A is 4x4. Block A[bi][bj] is the 2x2 block at row-block bi, col-block bj.
    # A[bi][bj] has entries A[2*bi + i', 2*bj + j'] for i',j' in {0,1}.
    
    # Similarly for B and C.
    
    # For a top-level Strassen term:
    # A_factor = sum_{bi,bj} a[bi][bj] * A[bi][bj]
    # B_factor = sum_{bj,bk} b[bj][bk] * B[bj][bk]
    # C_factor = sum_{bi,bk} c[bi][bk] * C[bi][bk]  (C[bi][bk] is block at row-block bi, col-block bk)
    
    # The 2x2x2 multiplication: (A_factor) * (B_factor) = C_factor (as 2x2 matrices)
    
    # Now decompose this 2x2x2 into 7 scalar mults using Strassen.
    
    # For the 2x2x2: A_factor is a 2x2 matrix, B_factor is 2x2, C_factor is 2x2.
    # A_factor[i',j'] = sum_{bi,bj} a[bi][bj] * A[2*bi+i', 2*bj+j']
    # B_factor[j',k'] = sum_{bj,bk} b[bj][bk] * B[2*bj+j', 2*bk+k']
    # C_factor[i',k'] = sum_{bi,bk} c[bi][bk] * C[2*bi+i', 2*bk+k']
    
    # Strassen on this 2x2x2: 7 terms.
    # For each Strassen sub-term t:
    #   U_sub[i',j'] (2x2), V_sub[j',k'] (2x2), W_sub[k',i'] (2x2)
    #   Contribution to C[i',k'] = sum_{j'} U_sub[i',j'] * A_factor[i',j'] * V_sub[j',k'] * W_sub[k',i']
    #   Wait, no. The decomposition is:
    #   C[i',k'] = sum_t (sum_{j'} U_t[i',j'] * A_factor[i',j']) * (sum_{j''} V_t[j'',k'] * B_factor[j'',k']) * W_t[k',i']
    #   
    #   Hmm, that's not right either. Let me go back to basics.
    
    # The tensor decomposition for 2x2x2:
    # T[i,j,k] = sum_t U_t[i,j] * V_t[j,k] * W_t[k,i]
    # where T[i,j,k] = 1 if this is the matrix multiplication tensor (i.e., always 1 for 2x2x2 MMT).
    
    # Actually no. The MMT tensor for 2x2x2: T[i,j,k] represents that A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k (it's the structure tensor).
    
    # The decomposition: 1 = sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] for all i,j,k.
    
    # This is what Strassen provides.
    
    # For the 4x4x4 case, the full tensor is:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    # Decomposition: T[i,j,k] = sum_{c=1}^{49} U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # I'll generate the 49 terms by expanding the two-level Strassen.
    
    s2 = strassen_2x2_terms()  # 7 terms for 2x2x2
    
    # Top-level Strassen terms for 4x4x4 (in terms of 2x2 blocks):
    # Each term: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # A_block_coeffs[bi][bj]: coefficient of A block (bi,bj)
    # B_block_coeffs[bj][bk]: coefficient of B block (bj,bk)  
    # C_block_coeffs[bi][bk]: coefficient of C block (bi,bk)
    
    top_strassen = [
        # M1 = (A00+A11)(B00+B11)
        # A: A00 coeff 1, A11 coeff 1 -> a = [[1,0],[0,1]]
        # B: B00 coeff 1, B11 coeff 1 -> b = [[1,0],[0,1]]
        # C: C00 coeff 1, C11 coeff 1 -> c = [[1,0],[0,1]]
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # M2 = (A10+A11)B00
        # A: A10 coeff 1, A11 coeff 1 -> a = [[0,0],[1,1]]
        # B: B00 coeff 1 -> b = [[1,0],[0,0]]
        # C: C10 coeff 1 -> c = [[0,0],[1,0]]
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        # M3 = A00(B10-B11)
        # A: A00 coeff 1 -> a = [[1,0],[0,0]]
        # B: B10 coeff 1, B11 coeff -1 -> b = [[0,0],[1,-1]]
        # C: C01 coeff 1 -> c = [[0,1],[0,0]]
        ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,1],[0,0]]),
        # M4 = A11(B01-B00)
        # A: A11 coeff 1 -> a = [[0,0],[0,1]]
        # B: B01 coeff 1, B00 coeff -1 -> b = [[-1,1],[0,0]]
        # C: C00 coeff 1, C10 coeff 1 -> c = [[1,0],[1,0]]
        ([[0,0],[0,1]], [[-1,1],[0,0]], [[1,0],[1,0]]),
        # M5 = (A00+A01)B11
        # A: A00 coeff 1, A01 coeff 1 -> a = [[1,1],[0,0]]
        # B: B11 coeff 1 -> b = [[0,0],[0,1]]
        # C: C01 coeff 1, C11 coeff 1 -> c = [[0,1],[0,1]]
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,1],[0,1]]),
        # M6 = (A10-A00)(B00+B01)
        # A: A10 coeff 1, A00 coeff -1 -> a = [[1,0],[1,0]]
        # B: B00 coeff 1, B01 coeff 1 -> b = [[1,1],[0,0]]
        # C: C11 coeff 1 -> c = [[0,0],[0,1]]
        ([[1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # M7 = (A01-A11)(B10+B11)
        # A: A01 coeff 1, A11 coeff -1 -> a = [[0,1],[0,-1]]
        # B: B10 coeff 1, B11 coeff 1 -> b = [[0,0],[1,1]]
        # C: C00 coeff 1 -> c = [[1,0],[0,0]]
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Now generate the 49 scalar-level terms.
    # For each top-level term (a, b, c) and each Strassen 2x2x2 sub-term (U_sub, V_sub, W_sub):
    
    U_4x4 = []  # list of 4x4 matrices (as lists of lists)
    V_4x4 = []
    W_4x4 = []
    
    for (a_blocks, b_blocks, c_blocks) in top_strassen:
        for (U_sub, V_sub, W_sub) in s2:
            # U_sub[i'][j'], V_sub[j'][k'], W_sub[k'][i']
            # Build U_4x4[4][4], V_4x4[4][4], W_4x4[4][4]
            
            U_full = [[0]*4 for _ in range(4)]
            V_full = [[0]*4 for _ in range(4)]
            W_full = [[0]*4 for _ in range(4)]  # W_full[k][i]
            
            for bi in range(2):
                for bj in range(2):
                    a_coeff = a_blocks[bi][bj]
                    if a_coeff == 0:
                        continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = 2*bi + i_prime
                            j = 2*bj + j_prime
                            U_full[i][j] += a_coeff * U_sub[i_prime][j_prime]
            
            for bj in range(2):
                for bk in range(2):
                    b_coeff = b_blocks[bj][bk]
                    if b_coeff == 0:
                        continue
                    for j_prime in range(2):
                        for k_prime in range(2):
                            j = 2*bj + j_prime
                            k = 2*bk + k_prime
                            V_full[j][k] += b_coeff * V_sub[j_prime][k_prime]
            
            for bi in range(2):
                for bk in range(2):
                    c_coeff = c_blocks[bi][bk]
                    if c_coeff == 0:
                        continue
                    for k_prime in range(2):
                        for i_prime in range(2):
                            k = 2*bk + k_prime
                            i = 2*bi + i_prime
                            W_full[k][i] += c_coeff * W_sub[k_prime][i_prime]
            
            U_4x4.append(U_full)
            V_4x4.append(V_full)
            W_4x4.append(W_full)
    
    # Now build the full U, V, W for 4x4x6.
    # Part 1: 4x4x4 (k=0,1,2,3) - 49 terms
    # Part 2: 4x4x2 (k=4,5) - 32 terms (schoolbook)
    
    total_rank = len(U_4x4) + 4*4*2  # 49 + 32 = 81
    
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 81
    V = [[0]*total_rank for _ in range(m*p)]  # 24 x 81
    W = [[0]*total_rank for _ in range(p*n)]  # 24 x 81
    
    # Fill in Part 1: 4x4x4 Strassen terms
    for c_idx in range(len(U_4x4)):
        U_c = U_4x4[c_idx]
        V_c = V_4x4[c_idx]
        W_c = W_4x4[c_idx]
        
        for i in range(4):
            for j in range(4):
                if U_c[i][j] != 0:
                    U[i*m + j][c_idx] = U_c[i][j]
        
        for j in range(4):
            for k in range(4):  # k = 0,1,2,3
                if V_c[j][k] != 0:
                    V[j*p + k][c_idx] = V_c[j][k]
        
        for k in range(4):  # k = 0,1,2,3
            for i in range(4):
                if W_c[k][i] != 0:
                    W[k*n + i][c_idx] = W_c[k][i]
    
    # Fill in Part 2: 4x4x2 schoolbook terms (k=4,5)
    col_offset = len(U_4x4)
    col = col_offset
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):  # k = 4, 5
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
