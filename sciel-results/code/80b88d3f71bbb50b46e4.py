"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # Block structure: split 4 into 2+2, split 6 into 3+3
    # A: 2x2 blocks of 2x2
    # B: 2x2 blocks of 2x3 (row-block x col-block)
    # C: 2x2 blocks of 2x3
    
    # Strassen: 7 multiplications, each is MM(2,2,3) with rank 11
    # Total rank: 7 * 11 = 77
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77
    
    col = 0
    
    # Helper: for a 2x2 A-block factor (linear combo) and 2x3 B-block factor,
    # add the rank-11 MM(2,2,3) decomposition
    # A_block_indices: dict mapping (bi, bj) -> list of (U_index, coeff)
    # B_block_indices: dict mapping (bk, bl) -> list of (V_index, coeff)
    # C_block_target: (ci, cl) -> W indices for the resulting 2x3 C-block
    
    # Strassen terms: (A_factor, B_factor, C_combination)
    # Each A_factor is a 2x2 matrix of (U_index, coeff) pairs
    # Each B_factor is a 2x3 matrix of (V_index, coeff) pairs
    # C_combination tells how to combine into C blocks
    
    # Let's define the 7 Strassen multiplications explicitly
    
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x3)
    
    # For each Strassen term, the A-factor and B-factor are sums/differences of blocks
    # Then we do MM(2,2,3) on them
    
    # MM(2,2,3) rank-11 decomposition:
    # 7 Strassen terms for the 2x2x2 part (cols 0,1 of the 2x3 B)
    # 4 schoolbook terms for the 2x2x1 part (col 2 of the 2x3 B)
    
    # For a given A-factor (2x2) and B-factor (2x3):
    # The 2x2x2 part uses Strassen on B[:,0:2]
    # The 2x2x1 part uses schoolbook on B[:,2]
    
    def add_mm223(A_factor, B_factor, C_target_row_block, C_target_col_block, start_col):
        """
        A_factor: 2x2 list of lists, each entry is (U_index, coeff)
        B_factor: 2x3 list of lists, each entry is (V_index, coeff)
        C_target_row_block: 0 or 1 (which row-block of C)
        C_target_col_block: 0 or 1 (which col-block of C)
        start_col: starting column index in U,V,W
        Returns: next column index
        """
        c = start_col
        
        # Strassen for the 2x2x2 part (B cols 0,1)
        # B2 = B_factor[:, 0:2] (2x2)
        B2_00 = B_factor[0][0]  # (V_idx, coeff)
        B2_01 = B_factor[0][1]
        B2_10 = B_factor[1][0]
        B2_11 = B_factor[1][1]
        
        # Strassen terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        A00 = A_factor[0][0]
        A01 = A_factor[0][1]
        A10 = A_factor[1][0]
        A11 = A_factor[1][1]
        
        # For each Strassen sub-term, we have a product of a 1x1 "A element" (which is actually a sum of A entries)
        # and a 1x1 "B element" (sum of B entries), producing a C element
        # But wait, in Strassen for 2x2x2, each "multiplication" is scalar*scalar
        # In our case, each "scalar" is a linear combination of A or B entries
        
        # Actually, I need to rethink. In the Strassen decomposition of MM(2,2,2),
        # each of the 7 terms is of the form (linear combo of A) * (linear combo of B) -> contributes to C
        # But here, the "A" and "B" are themselves 2x2 and 2x3 matrices, and the "multiplication"
        # is matrix multiplication.
        
        # I think I'm overcomplicating this. Let me step back.
        
        # The Strassen decomposition of MM(2,2,2) says:
        # C = A*B where A,B,C are 2x2
        # Can be computed with 7 scalar multiplications:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        # c00 = M1+M4-M5+M7
        # c01 = M3+M5
        # c10 = M2+M4
        # c11 = M1-M2+M3+M6
        
        # In our tensor decomposition, each "M" is a rank-1 tensor:
        # M1 corresponds to: U has (a00+a11) pattern, V has (b00+b11) pattern, W has contribution to c00
        # But M1 contributes to multiple C entries!
        
        # Actually, in the tensor decomposition, each rank-1 term is:
        # u ⊗ v ⊗ w where u is a vector in A-space, v in B-space, w in C-space
        # For M1: u = e_{a00} + e_{a11}, v = e_{b00} + e_{b11}, w = e_{c00} + e_{c11}
        # Wait no, that's not right either.
        
        # Let me think about this more carefully.
        # The MM(2,2,2) tensor is: sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * delta_{i,j,k}
        # where C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * [C[i,k] depends on A[i,j]B[j,k]]
        # More precisely: T = sum_{i,j,k} e_{(i,j)}^A ⊗ e_{(j,k)}^B ⊗ e_{(i,k)}^C
        # No wait, that's the schoolbook.
        
        # The tensor T for MM(2,2,2) has T[(i,j),(j,k),(i,k)] = 1 for all i,j,k, and 0 otherwise.
        # Wait, that's not right. The tensor is defined such that:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T is a 3-tensor with T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0.
        # But (i,j) and (j,k) share j, so:
        # T[(i,j),(j,k),(i,k)] = 1
        # This is a sum over i,j,k of e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
        # That's the schoolbook rank-8 decomposition.
        
        # Strassen shows this can be done with rank 7.
        # The Strassen decomposition is:
        # T = M1 + M2 + M3 + M4 + M5 + M6 + M7
        # where each Mx is a rank-1 tensor.
        
        # M1 = (e_{00}^A + e_{11}^A) ⊗ (e_{00}^B + e_{11}^B) ⊗ (e_{00}^C + e_{11}^C)
        # Wait, that doesn't seem right dimensionally.
        
        # Let me look at this differently. In the Strassen algorithm:
        # M1 = (A00+A11)(B00+B11)
        # This is a scalar multiplication. In the tensor decomposition, this corresponds to:
        # The term contributes to C00 and C11.
        # Specifically, M1 appears in c00 = M1+M4-M5+M7 and c11 = M1-M2+M3+M6
        # So M1 contributes +1 to C00 and +1 to C11.
        
        # As a rank-1 tensor: M1 = u1 ⊗ v1 ⊗ w1
        # where u1 = e_{A00} + e_{A11} (in A-space)
        # v1 = e_{B00} + e_{B11} (in B-space)  
        # w1 = e_{C00} + e_{C11} (in C-space)
        
        # Let's verify: u1 ⊗ v1 ⊗ w1 = (e_{A00}+e_{A11}) ⊗ (e_{B00}+e_{B11}) ⊗ (e_{C00}+e_{C11})
        # = e_{A00}⊗e_{B00}⊗e_{C00} + e_{A00}⊗e_{B00}⊗e_{C11} + e_{A00}⊗e_{B11}⊗e_{C00} + e_{A00}⊗e_{B11}⊗e_{C11}
        # + e_{A11}⊗e_{B00}⊗e_{C00} + e_{A11}⊗e_{B00}⊗e_{C11} + e_{A11}⊗e_{B11}⊗e_{C00} + e_{A11}⊗e_{B11}⊗e_{C11}
        
        # The original tensor has:
        # e_{A00}⊗e_{B00}⊗e_{C00} (from i=0,j=0,k=0)
        # e_{A00}⊗e_{B01}⊗e_{C01} (from i=0,j=0,k=1)
        # e_{A01}⊗e_{B10}⊗e_{C00} (from i=0,j=1,k=0)
        # e_{A01}⊗e_{B11}⊗e_{C01} (from i=0,j=1,k=1)
        # e_{A10}⊗e_{B00}⊗e_{C10} (from i=1,j=0,k=0)
        # e_{A10}⊗e_{B01}⊗e_{C11} (from i=1,j=0,k=1)
        # e_{A11}⊗e_{B10}⊗e_{C10} (from i=1,j=1,k=0)
        # e_{A11}⊗e_{B11}⊗e_{C11} (from i=1,j=1,k=1)
        
        # So M1 as I wrote it includes terms like e_{A00}⊗e_{B00}⊗e_{C11} which is NOT in the original tensor.
        # This means my understanding is wrong.
        
        # I think the issue is that in the Strassen decomposition, the rank-1 tensors don't directly correspond to the algorithm's intermediate values in the way I'm thinking.
        
        # Let me look up the actual Strassen tensor decomposition.
        # The Strassen decomposition of MM(2,2,2) as a sum of 7 rank-1 tensors:
        
        # Actually, I recall now. The 7 rank-1 tensors are:
        # T1 = (e00+e11)⊗(e00+e11)⊗(e00+e11)  -- NO, this is wrong
        
        # I think I need to be more careful. Let me just use the schoolbook for MM(2,2,3) with rank 12
        # and see if I can get a better overall rank.
        
        # Actually, let me try a completely different approach. Let me just use the known
        # rank-11 decomposition of MM(2,2,3) that I can construct explicitly.
        
        # For MM(2,2,3): A is 2x2, B is 2x3, C is 2x3
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # Rank 11 decomposition:
        # For k=0,1: use Strassen (7 terms)
        # For k=2: use schoolbook (4 terms)
        # Total: 11 terms
        
        # The 7 Strassen terms for k=0,1:
        # These are rank-1 tensors in the space A(2x2) x B(2x2 for cols 0,1) x C(2x2 for cols 0,1)
        # 
        # The 4 schoolbook terms for k=2:
        # T_{i,j} = e_{A[i,j]} ⊗ e_{B[j,2]} ⊗ e_{C[i,2]} for i,j in {0,1}
        
        # For the Strassen part, the 7 rank-1 tensors are (in the 2x2x2 subspace):
        # Let me denote A entries as a00,a01,a10,a11 and B entries as b00,b01,b10,b11 (for cols 0,1)
        # C entries as c00,c01,c10,c11
        
        # The Strassen rank-1 tensors (this is the standard decomposition):
        # I'll use the formulation where each term is a product of linear forms.
        
        # After more thought, let me just implement the schoolbook for everything and get rank 96,
        # then try to improve. Given time constraints, let me implement the 77-rank construction.
        
        # I'll use a different strategy: directly construct the 77 rank-1 tensors by
        # combining the Strassen block decomposition with the MM(2,2,3) rank-11 decomposition.
        
        # For each of the 7 Strassen block-multiplications, I need to expand the MM(2,2,3)
        # rank-11 decomposition. Each MM(2,2,3) decomposition has 11 rank-1 tensors.
        # Each rank-1 tensor in MM(2,2,3) is: u ⊗ v ⊗ w where u is in R^4 (2x2 A-block),
        # v is in R^6 (2x3 B-block), w is in R^6 (2x3 C-block).
        
        # When embedded in the full 4x4x4x6 problem, these become rank-1 tensors in the full space.
        
        # This is getting very complex to code up correctly. Let me just implement the schoolbook
        # with rank 96 as a safe fallback, and try to implement the 77-rank version.
        
        # Given the complexity and time, let me implement a clean schoolbook (rank 96) first,
        # then attempt the Strassen-based improvement.
        
        return None  # placeholder, will implement below
    
    # Let me just implement the schoolbook for now as a safe baseline
    # and structure the code to allow improvement
    
    rank = n * m * p  # 96
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
