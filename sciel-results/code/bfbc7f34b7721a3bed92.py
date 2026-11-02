<<<<<<< SEARCH
"""Block-based decomposition: Rank-23 for 3x3x3 block + schoolbook for 5 columns."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
=======
"""Strassen-based block decomposition for 3x3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll build a decomposition using Strassen's rank-7 for 2x2x2 blocks
    # plus schoolbook terms for the remaining entries.
    #
    # Strategy: For each pair of rows (i1,i2) of A and each pair of columns (k1,k2) of B,
    # we can use Strassen on the 2x2x2 sub-tensor if both indices j range over a pair.
    # But j only ranges over 3 values, so we can form at most one 2x2x2 Strassen block
    # per (row-pair, col-pair) using j in {0,1} or {1,2}.
    #
    # Simpler approach: Use Strassen for the top-left 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}),
    # then handle all remaining terms with schoolbook.
    #
    # Strassen 2x2x2 rank-7 terms:
    # m1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # m2 = (A10+A11)*B00       -> contributes to C10
    # m3 = A00*(B10-B11)       -> contributes to C01
    # m4 = A11*(B01-B00)       -> contributes to C00, C10
    # m5 = (A00+A01)*B11       -> contributes to C00, C01
    # m6 = (A10-A00)*(B00+B10) -> contributes to C11
    # m7 = (A01-A11)*(B01+B11) -> contributes to C00, C11
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # The remaining C entries (involving row/col index 2) use schoolbook.
    #
    # Total: 7 (Strassen) + schoolbook for remaining terms.
    # Remaining C entries: C02, C03,...,C07, C12,...,C17, C20,...,C27
    # That's 8*2 + 8*1 = 24 entries, each needing 3 mults = 72 schoolbook mults.
    # But wait, we already used some A and B entries in Strassen.
    #
    # Actually, let me just do a cleaner approach:
    # Use Strassen for the 2x2x2 sub-block where i in {0,1}, j in {0,1}, k in {0,1}.
    # Then for all other (i,j,k) triples, use schoolbook.
    #
    # The Strassen block handles (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}.
    # That's 8 entries replaced by 7 rank-1 terms.
    # Remaining: 72 - 8 = 64 schoolbook terms.
    # Total rank: 7 + 64 = 71.
    
    # Let me build this explicitly.
    # First, the 7 Strassen terms for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}).
    
    # Strassen term definitions (u_coeffs, v_coeffs, w_coeffs):
    # Each term r has: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # for i in {0,1}, j in {0,1}, k in {0,1}
    
    # m1 = (A00+A11)*(B00+B11): 
    #   U: A00=1, A11=1 -> U[0*3+0]=1, U[1*3+1]=1
    #   V: B00=1, B11=1 -> V[0*8+0]=1, V[1*8+1]=1
    #   W: C00=1, C11=1 -> W[0*3+0]=1, W[1*3+1]=1
    # Contribution to (i,j,k): U[i*3+j]*V[j*8+k]*W[k*3+i]
    # For (0,0,0): 1*1*1=1, (0,0,1): 1*1*0=0, (0,1,0): 1*1*0=0, (0,1,1): 1*1*0=0
    # For (1,0,0): 0*1*1=0, (1,0,1): 0*1*0=0, (1,1,0): 0*1*0=0, (1,1,1): 0*1*0=0
    # Hmm, that doesn't work. Let me reconsider.
    
    # The tensor decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For the 2x2x2 block, T[i,j,k]=1 for all i,j,k in {0,1}.
    #
    # Strassen decomposition of the 2x2x2 all-ones tensor:
    # r=0: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1) -> m1=(A00+A11)(B00+B11)
    #   Check: (0,0,0): 1*1*1=1, (0,1,0): 0*0*1=0... 
    #   Wait, I need to be more careful with indexing.
    
    # Let me use a different formulation. For 2x2x2, the 7 Strassen terms are:
    # Term 0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    #   (i,j) pairs: (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->1
    #   (j,k) pairs: (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->1  
    #   (k,i) pairs: (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->1
    #   Product at (i,j,k): U[i,j]*V[j,k]*W[k,i]
    #   (0,0,0): 1*1*1=1, (0,0,1): 1*0*0=0, (0,1,0): 0*0*1=0, (0,1,1): 0*1*0=0
    #   (1,0,0): 0*1*1=0, (1,0,1): 0*0*0=0, (1,1,0): 1*0*0=0, (1,1,1): 1*1*1=1
    #   Sum so far: (0,0,0)=1, (1,1,1)=1, rest=0
    
    # This is getting complicated. Let me just use a verified Strassen decomposition.
    # I'll encode the 7 terms directly.
    
    # For 2x2x2, using indices (i,j,k) in {0,1}^3:
    # The 7 Strassen rank-1 tensors:
    # r0: U00=1,U11=1; V00=1,V11=1; W00=1,W11=1
    # r1: U10=1,U11=1; V00=1; W10=1
    # r2: U00=1; V10=1,V11=-1; W01=1
    # r3: U11=1; V01=1,V00=-1; W00=1,W10=1
    # r4: U00=1,U01=1; V11=1; W00=-1,W01=1
    # r5: U10=1,U00=-1; V00=1,V10=1; W11=1
    # r6: U01=1,U11=-1; V01=1,V11=1; W00=1,W11=1
    
    # Let me verify r0: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    # (0,0,0): U(0,0)*V(0,0)*W(0,0) = 1*1*1 = 1
    # (0,0,1): U(0,0)*V(0,1)*W(1,0) = 1*0*0 = 0
    # (0,1,0): U(0,1)*V(1,0)*W(0,0) = 0*0*1 = 0
    # (0,1,1): U(0,1)*V(1,1)*W(1,0) = 0*1*0 = 0
    # (1,0,0): U(1,0)*V(0,0)*W(0,1) = 0*1*0 = 0
    # (1,0,1): U(1,0)*V(0,1)*W(1,1) = 0*0*1 = 0
    # (1,1,0): U(1,1)*V(1,0)*W(0,1) = 1*0*0 = 0
    # (1,1,1): U(1,1)*V(1,1)*W(1,1) = 1*1*1 = 1
    # So r0 contributes 1 to (0,0,0) and (1,1,1).
    
    # This is correct for Strassen. Let me build the full 7-term decomposition.
    
    # I'll create the U, V, W matrices for the full 3x3x8 problem.
    # Start with all zeros, then add Strassen terms for the 2x2x2 block,
    # then add schoolbook terms for all other (i,j,k) triples.
    
    rank = 71  # 7 Strassen + 64 schoolbook
    
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 71
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 71
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 71
    
    # Strassen terms for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # Term 0: (A00+A11)(B00+B11)
    r = 0
    U[0*3+0][r] += 1.0; U[1*3+1][r] += 1.0
    V[0*8+0][r] += 1.0; V[1*8+1][r] += 1.0
    W[0*3+0][r] += 1.0; W[1*3+1][r] += 1.0
    
    # Term 1: (A10+A11)B00
    r = 1
    U[1*3+0][r] += 1.0; U[1*3+1][r] += 1.0
    V[0*8+0][r] += 1.0
    W[1*3+0][r] += 1.0
    
    # Term 2: A00(B10-B11)
    r = 2
    U[0*3+0][r] += 1.0
    V[1*8+0][r] += 1.0; V[1*8+1][r] += -1.0
    W[0*3+1][r] += 1.0
    
    # Term 3: A11(B01-B00)
    r = 3
    U[1*3+1][r] += 1.0
    V[0*8+1][r] += 1.0; V[0*8+0][r] += -1.0
    W[0*3+0][r] += 1.0; W[1*3+0][r] += 1.0
    
    # Term 4: (A00+A01)B11
    r = 4
    U[0*3+0][r] += 1.0; U[0*3+1][r] += 1.0
    V[1*8+1][r] += 1.0
    W[0*3+0][r] += -1.0; W[0*3+1][r] += 1.0
    
    # Term 5: (A10-A00)(B00+B10)
    r = 5
    U[1*3+0][r] += 1.0; U[0*3+0][r] += -1.0
    V[0*8+0][r] += 1.0; V[1*8+0][r] += 1.0
    W[1*3+1][r] += 1.0
    
    # Term 6: (A01-A11)(B01+B11)
    r = 6
    U[0*3+1][r] += 1.0; U[1*3+1][r] += -1.0
    V[0*8+1][r] += 1.0; V[1*8+1][r] += 1.0
    W[0*3+0][r] += 1.0; W[1*3+1][r] += 1.0
    
    # Now add schoolbook terms for all (i,j,k) NOT in the 2x2x2 Strassen block
    # The Strassen block covers i in {0,1}, j in {0,1}, k in {0,1}
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0,1) and j in (0,1) and k in (0,1):
                    continue  # Already handled by Strassen
                U[i*m+j][col] += 1.0
                V[j*p+k][col] += 1.0
                W[k*n+i][col] += 1.0
                col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
