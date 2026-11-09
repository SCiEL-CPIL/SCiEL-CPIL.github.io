"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition by splitting B into B1 (4x4) and B2 (4x2)
    # C = A*B1 + A*B2
    # Use Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x2 (rank 32)
    # Total rank = 81
    
    rank = 81
    U = [[0]*rank for _ in range(n*m)]   # 16 x 81
    V = [[0]*rank for _ in range(m*p)]   # 24 x 81
    W = [[0]*rank for _ in range(p*n)]   # 24 x 81
    
    col = 0
    
    # Part 1: 4x4x4 Strassen decomposition (rank 49)
    # A indices: i in 0..3, j in 0..3
    # B1 indices: j in 0..3, k in 0..3 (columns 0-3 of B)
    # C1 indices: k in 0..3, i in 0..3
    
    # Strassen for 4x4: split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B1 = [[B11, B12], [B21, B22]] each 2x2
    # C1 = [[C11, C12], [C21, C22]] each 2x2
    
    # 7 Strassen products, each is a 2x2x2 Strassen (7 rank-1 terms)
    # Total: 7*7 = 49 rank-1 terms
    
    # Define the 2x2 Strassen coefficients
    # For 2x2x2: M1..M7 with specific A and B block combinations
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix product, decomposed via Strassen into 7 rank-1 terms
    # For a 2x2 product P = X*Y where X,Y are 2x2:
    # P11 = S1 + S4 - S5 + S7
    # P12 = S3 + S5
    # P21 = S2 + S4
    # P22 = S1 - S2 + S3 + S6
    # where S1=(X11+X22)(Y11+Y22), S2=(X21+X22)Y11, S3=X11(Y12-Y22),
    #       S4=X22(Y21-Y11), S5=(X11+X12)Y22, S6=(X21-X11)(Y11+Y12), S7=(X12-X22)(Y21+Y22)
    
    # For the 4x4x4 Strassen, we have 7 "macro" products Mi = Xi * Yi
    # where Xi and Yi are 2x2 matrices formed from A and B1 blocks.
    
    # Let me define the 7 macro-products with their A-block and B-block expressions:
    # Each macro-product contributes 7 rank-1 terms (from inner Strassen)
    
    # Macro-product definitions (A-side, B-side) as 2x2 block expressions:
    # M1: A-side = A11+A22, B-side = B11+B22
    # M2: A-side = A21+A22, B-side = B11
    # M3: A-side = A11, B-side = B12-B22
    # M4: A-side = A22, B-side = B21-B11
    # M5: A-side = A11+A12, B-side = B22
    # M6: A-side = A21-A11, B-side = B11+B12
    # M7: A-side = A12-A22, B-side = B21+B22
    
    # For each macro-product, the A-side is a 2x2 matrix (sum/diff of A 2x2 blocks)
    # and B-side is a 2x2 matrix (sum/diff of B 2x2 blocks).
    
    # I'll represent each 2x2 block by its 4 entries.
    # A blocks: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3; A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    # B1 blocks: B11 = rows 0-1, cols 0-3... wait, B is 4x4 for this part.
    # B1: rows 0-3, cols 0-3. B11 = rows 0-1, cols 0-1; B12 = rows 0-1, cols 2-3; B21 = rows 2-3, cols 0-1; B22 = rows 2-3, cols 2-3
    
    # For a macro-product Mi = Xi * Yi where Xi is 2x2 (from A) and Yi is 2x2 (from B1):
    # The 7 inner Strassen terms S1..S7 each involve specific entries of Xi and Yi.
    
    # Let me just enumerate all 49 rank-1 terms directly.
    
    # For macro m (0..6), inner Strassen term s (0..6):
    # Each gives a rank-1 term in the 4x4x4 tensor.
    
    # I'll define the A-side and B-side for each (m, s) pair.
    
    # A-side blocks (2x2):
    # A11: entries at (0,0),(0,1),(1,0),(1,1) in A -> U indices: 0,1,4,5 (i*m+j = i*4+j)
    # A12: entries at (0,2),(0,3),(1,2),(1,3) -> U indices: 2,3,6,7
    # A21: entries at (2,0),(2,1),(3,0),(3,1) -> U indices: 8,9,12,13
    # A22: entries at (2,2),(2,3),(3,2),(3,3) -> U indices: 10,11,14,15
    
    # B1-side blocks (2x2):
    # B11: entries at (0,0),(0,1),(1,0),(1,1) in B1 -> V indices: j*p+k = j*6+k, j=0,1; k=0,1 -> 0,1,6,7
    # B12: entries at (0,2),(0,3),(1,2),(1,3) -> V indices: 2,3,8,9
    # B21: entries at (2,0),(2,1),(3,0),(3,1) -> V indices: 12,13,18,19
    # B22: entries at (2,2),(2,3),(3,2),(3,3) -> V indices: 14,15,20,21
    
    # For each macro-product, define which A-blocks and B-blocks are used with what signs.
    
    # Macro m: (A_blocks, B_blocks) where each is a list of (block, sign)
    macros = [
        # M1: (A11+A22) * (B11+B22)
        ([(0,1),(2,1)], [(0,1),(2,1)]),
        # M2: (A21+A22) * B11
        ([(1,1),(2,1)], [(0,1)]),
        # M3: A11 * (B12-B22)
        ([(0,1)], [(1,1),(2,-1)]),
        # M4: A22 * (B21-B11)
        ([(2,1)], [(2,1),(0,-1)]),
        # M5: (A11+A12) * B22
        ([(0,1),(1,1)], [(2,1)]),
        # M6: (A21-A11) * (B11+B12)
        ([(1,1),(0,-1)], [(0,1),(1,1)]),
        # M7: (A12-A22) * (B21+B22)
        ([(1,1),(2,-1)], [(2,1),(2,1)]),  # Wait, B21+B22, that's blocks 2 and 2? No.
    ]
    
    # Let me fix: block indices 0=A11/B11, 1=A12/B12, 2=A21/B21, 3=A22/B22
    # M7: (A12-A22) * (B21+B22) -> A: [(1,1),(3,-1)], B: [(2,1),(3,1)]
    
    macros = [
        # M1: (A11+A22) * (B11+B22)
        ([(0,1),(3,1)], [(0,1),(3,1)]),
        # M2: (A21+A22) * B11
        ([(2,1),(3,1)], [(0,1)]),
        # M3: A11 * (B12-B22)
        ([(0,1)], [(1,1),(3,-1)]),
        # M4: A22 * (B21-B11)
        ([(3,1)], [(2,1),(0,-1)]),
        # M5: (A11+A12) * B22
        ([(0,1),(1,1)], [(3,1)]),
        # M6: (A21-A11) * (B11+B12)
        ([(2,1),(0,-1)], [(0,1),(1,1)]),
        # M7: (A12-A22) * (B21+B22)
        ([(1,1),(3,-1)], [(2,1),(3,1)]),
    ]
    
    # For each macro m and inner Strassen term s, we need to figure out
    # which specific entries of A and B are involved.
    
    # Inner Strassen for 2x2 product P = X*Y:
    # S1 = (X11+X22)(Y11+Y22)
    # S2 = (X21+X22)Y11
    # S3 = X11(Y12-Y22)
    # S4 = X22(Y21-Y11)
    # S5 = (X11+X12)Y22
    # S6 = (X21-X11)(Y11+Y12)
    # S7 = (X12-X22)(Y21+Y22)
    
    # Each Ss is a product of two linear combinations of entries.
    # For the rank-1 decomposition, each Ss contributes to specific C entries.
    
    # Actually, for the tensor decomposition, each rank-1 term is:
    # U[i,j] * V[j,k] * W[k,i] for specific (i,j,k)
    
    # This is getting very complex to do by hand. Let me use a different approach.
    
    # I'll construct the decomposition by building the actual multiplication
    # and extracting rank-1 terms.
    
    # Actually, let me just use a simpler but still improved approach.
    # Use schoolbook for everything but with a smarter block structure.
    
    # Alternative: just use schoolbook rank 96. Score = 96/96 = 1.0.
    # That's the same as current.
    
    # Let me try to implement the Strassen 4x4x4 properly.
    
    # I'll take a different approach: build the decomposition programmatically
    # by simulating the Strassen algorithm and recording each scalar multiplication.
    
    # Reset
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, u_coeff=1, v_coeff=1, w_coeff=1):
        nonlocal rank
        U_list.append((u_idx, u_coeff))
        V_list.append((v_idx, v_coeff))
        W_list.append((w_idx, w_coeff))
        rank += 1
    
    # For 4x4x4 Strassen:
    # I'll implement it by tracking each scalar multiplication.
    
    # Let me define helper to add a 2x2 Strassen decomposition for given
    # A-block entries and B-block entries.
    
    # A 2x2 block of A: entries at positions (r0+c0, c0+c) for r,c in 0..1
    # U index for A[i,j] = i*4+j
    
    # B 2x2 block of B: entries at positions (r0+c0... no, B[j,k], V index = j*6+k
    
    # C 2x2 block: C[i,k], W index = k*4+i
    
    # For a 2x2x2 Strassen decomposition of A_block * B_block = C_block:
    # A_block[i,j] for i,j in 0..1, mapped to global A[ai+i, aj+j]
    # B_block[i,j] for i,j in 0..1, mapped to global B[bi+i, bj+j]  
    # C_block[i,j] for i,j in 0..1, mapped to global C[ci+i, cj+j]
    
    def strassen_2x2(ai, aj, bi, bj, ci, cj):
        """Add 7 rank-1 terms for 2x2x2 Strassen.
        A block at top-left (ai,aj), B block at top-left (bi,bj), C block at top-left (ci,cj)
        """
        # A entries: A[ai,aj]=A11, A[ai,aj+1]=A12, A[ai+1,aj]=A21, A[ai+1,aj+1]=A22
        # U indices:
        A11 = (ai, aj)
        A12 = (ai, aj+1)
        A21 = (ai+1, aj)
        A22 = (ai+1, aj+1)
        
        # B entries: B[bi,bj]=B11, B[bi,bj+1]=B12, B[bi+1,bj]=B21, B[bi+1,bj+1]=B22
        # V indices:
        B11 = (bi, bj)
        B12 = (bi, bj+1)
        B21 = (bi+1, bj)
        B22 = (bi+1, bj+1)
        
        # C entries: C[ci,cj]=C11, C[ci,cj+1]=C12, C[ci+1,cj]=C21, C[ci+1,cj+1]=C22
        # W indices: W[k*n+i] = W[cj+ci*4]... wait, W index = k*n+i where k is col, i is row
        # C[i,k] -> W[k*4+i]
        C11 = (cj, ci)      # C[ci,cj], W index = cj*4+ci
        C12 = (cj+1, ci)    # C[ci,cj+1], W index = (cj+1)*4+ci
        C21 = (cj, ci+1)    # C[ci+1,cj], W index = cj*4+(ci+1)
        C22 = (cj+1, ci+1)  # C[ci+1,cj+1], W index = (cj+1)*4+(ci+1)
        
        # S1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        # This is a product of sums, so it's not a single rank-1 term.
        # (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # That's 4 rank-1 terms!
        
        # Hmm, I think I'm overcomplicating this. The Strassen algorithm uses 7 multiplications
        # of linear combinations, but each "multiplication" of two linear combinations
        # is itself multiple rank-1 terms.
        
        # Wait no. In the tensor decomposition context, a "multiplication" of two linear
        # combinations IS a rank-1 term if we allow the U, V, W entries to be linear combinations.
        # But U, V, W must have specific entries, and the product U[i,j,c]*V[j,k,c]*W[k,i,c]
        # must equal the contribution to C[i,k] from that term.
        
        # Actually, in a rank-r decomposition, each term c has:
        # C[i,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # summed over j.
        
        # For a rank-1 term, we fix a specific j, so:
        # C[i,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # This means each rank-1 term involves specific (i,j,k) entries.
        
        # So Strassen's "M1 = (A11+A22)(B11+B22)" doesn't directly give a rank-1 term.
        # Instead, we need to express the final C entries as sums of products of individual
        # A and B entries.
        
        # I think the cleanest way is to just do schoolbook and accept rank 96,
        # or find a different approach.
        
        # Let me just do schoolbook for now.
        pass
    
    # Given the complexity, let me just implement schoolbook with rank 96.
    # This matches the current score of 1.0.
    
    # Actually, let me try a different structural approach:
    # Use the fact that 4x4x6 can be seen as computing 4 rows of output.
    # Each row i of C: C[i,:] = A[i,:] * B (1x4 times 4x6 = 1x6)
    # This is a 1x4x6 tensor, rank = 4*6 = 24 per row, total 4*24 = 96.
    
    # No improvement.
    
    # Let me try: split A into 2x2 blocks and use a different factorization.
    
    # I'll just go with schoolbook rank 96.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
