"""Program for matrix multiplication 2x3 by 3x5 tensor decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    # Schoolbook rank 30 baseline is always valid
    rank_schoolbook = n * m * p  # 30
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Try to find rank 28 decomposition using structured search
    # The idea: express the tensor as sum of 28 rank-1 terms
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where (u_c)_{ij} (v_c)_{jk} (w_c)_{ki}
    # T_{ij, jk, ki} = 1 if all indices valid, else 0
    # Actually T_{ij, jk, ki} = delta(i,j,k) for the matmul tensor
    # The tensor entry T[(i,j),(j,k),(k,i)] = 1 for valid indices
    
    # We need: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for invalid combinations (where the j in (i,j) != j in (j,k)), it should be 0
    # But actually the tensor only has non-zero entries at valid (i,j,k) triples.
    # The tensor T is defined on the space C^{n*m} x C^{m*p} x C^{p*n}
    # T[a, b, c] = sum over valid paths... 
    # Actually for matmul: T is such that contracting A with T along first factor 
    # and B along second gives C.
    # T_{ij, jk, ki} = 1 means: (A ⊗ T) gives (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # For invalid entries (where the middle index doesn't match), the tensor is 0.
    # But wait - in the standard formulation, the tensor only lives on valid indices.
    # The "invalid" entries in U, V, W don't directly appear in the sum because
    # the tensor is only defined on the valid subgrid.
    
    # Actually, re-reading the problem: the tensor is the matmul tensor which has
    # T[(i,j),(j,k),(k,i)] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # All other entries are 0. So we need:
    # For valid (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For invalid entries, we don't care (they're not part of the tensor).
    # But the decomposition must exactly match the tensor, meaning:
    # The tensor T has entries T[(i,j),(j,k),(k,i)] for valid indices only.
    
    # So the constraint is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # That's n*m*p = 30 constraints. With rank 28, we have 28*(6+15+10) = 28*31 = 868 unknowns.
    # This is very overdetermined in a good way - should be feasible.
    
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    # Try a simple algebraic approach
    # Split the 30 terms into groups and try to merge some
    
    # Strategy: use rank 28 by finding a representation
    # Let's try: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For each c, we assign values to the 30 entries of U (well, 6 entries), 15 entries of V, 10 entries of W
    
    # Simple approach: try rank r from 28 upward, use randomized search
    # But this might be slow. Let's think of a structured approach.
    
    # Alternative: the Strassen-like approach for 2x3 and 3x5
    # Actually, let's just try to verify if rank 28 is achievable with a simple pattern.
    
    # Let me try a different approach: use the fact that we can write
    # some products as combinations.
    
    # For 2x3 * 3x5:
    # C_00 = A_00*B_00 + A_01*B_10 + A_02*B_20
    # C_01 = A_00*B_01 + A_01*B_11 + A_02*B_21
    # C_02 = A_00*B_02 + A_01*B_12 + A_02*B_22
    # C_03 = A_00*B_03 + A_01*B_13 + A_02*B_23
    # C_04 = A_00*B_04 + A_01*B_14 + A_02*B_24
    # C_10 = A_10*B_00 + A_11*B_10 + A_12*B_20
    # C_11 = A_10*B_01 + A_11*B_11 + A_12*B_21
    # C_12 = A_10*B_02 + A_11*B_12 + A_12*B_22
    # C_13 = A_10*B_03 + A_11*B_13 + A_12*B_23
    # C_14 = A_10*B_04 + A_11*B_14 + A_12*B_24
    
    # Each C_ik is a dot product of row i of A and column k of B.
    # Standard: 2*3*5 = 30 multiplications.
    # To reduce: use tricks like (A_00+A_01)*(B_00+B_10) = A_00*B_00 + A_00*B_10 + A_01*B_00 + A_01*B_10
    # This replaces 2 multiplications with 1 but introduces cross terms.
    
    # Let me try a systematic search with numpy for rank 28.
    # We have 30 equations (one per (i,j,k) triple) and rank*31 unknowns.
    # For rank 28: 28*31 = 868 unknowns, 30 equations. Very underdetermined.
    # So a solution should exist. Let's find one.
    
    # Actually, I realize the constraint is that the entries should be "integer/half-integer".
    # Let me try to construct one explicitly.
    
    # Simple idea: take the schoolbook 30 terms and merge pairs.
    # For example, terms (i,j,k) and (i,j',k') can sometimes be merged.
    # Term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1, others 0.
    # Term (i,j',k'): U[i*m+j']=1, V[j'*p+k']=1, W[k'*n+i]=1, others 0.
    # Merged: need u,v,w such that:
    #   u_{ij}*v_{jk}*w_{ki} = 1
    #   u_{ij'}*v_{j'k'}*w_{k'i} = 1
    #   and all other products = 0.
    # This is tricky.
    
    # Let me just do a numerical search for rank 28 with rational entries.
    # Since the problem is very underdetermined, we can set most entries to 0
    # and solve for the rest.
    
    # Approach: start with 30 schoolbook terms, then try to eliminate 2 of them
    # by expressing them as linear combinations of the other 28.
    
    # Actually, let me try a cleaner approach: use the fact that we only need
    # the sum to equal 1 for each (i,j,k). Let me set up a system.
    
    # For rank r, we have r columns. Let's try r=28.
    # Initialize with schoolbook 30, then zero out 2 columns and adjust.
    
    # Better: let's just try to find any valid rank-28 decomposition numerically.
    
    # Since we have 30 constraints and 868 free variables, we can:
    # 1. Set 868-30 = 838 variables to 0
    # 2. Solve the remaining 30x30 system (but it's nonlinear!)
    
    # The equations are cubic: sum_c U[a,c]*V[b,c]*W[d,c] = 1
    # This is nonlinear. Let me use a different approach.
    
    # Greedy: start with 28 random rank-1 terms, try to fit.
    # Or: use the schoolbook and merge.
    
    # Let me try: take schoolbook, pick 2 terms to remove, and see if I can
    # express their contribution using the remaining 28 terms by modifying them.
    
    # This is getting complicated. Let me just try a random search with verification.
    
    def verify(U, V, W):
        """Check if the decomposition is correct."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(len(U[0])):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find rank 28 by starting from schoolbook and perturbing
    # Actually, let me try a smarter approach.
    
    # Key insight: For each output entry C_ik, we need:
    # sum_j (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]) = 1 for each j
    # Wait no, it's: for each (i,j,k), the sum over c must equal 1.
    
    # Let me try: for each c, define the "pattern" of which (i,j,k) it contributes to.
    # In schoolbook, each c contributes to exactly one (i,j,k).
    # For rank 28, some c's must contribute to multiple (i,j,k)'s.
    
    # A term c with U[i1*m+j1,c]=a, V[j1*p+k1,c]=b, W[k1*n+i1,c]=c1
    # contributes a*b*c1 to term (i1,j1,k1).
    # If we also have U[i2*m+j2,c]=a2, V[j2*p+k2,c]=b2, W[k2*n+i2,c]=c2
    # it also contributes a2*b2*c2 to term (i2,j2,k2).
    
    # For this to work without affecting other terms, we need all other
    # products to be 0. This means the support of U_c, V_c, W_c must be
    # such that U_c[a] * V_c[b] * W_c[d] is nonzero only for the desired (i,j,k) pairs.
    
    # This is very restrictive. Let me think differently.
    
    # Alternative: allow the decomposition to have "cross terms" that cancel.
    # E.g., one term contributes +x to (i1,j1,k1) and +y to (i2,j2,k2),
    # another contributes -x+1 to (i1,j1,k1) and -y+1 to (i2,j2,k2).
    # Net: (i1,j1,k1) gets 1, (i2,j2,k2) gets 1.
    # But this requires 2 terms to handle what was 1 term, which is worse.
    
    # Actually for going from 30 to 28, we need to save 2 terms.
    # The simplest: find 2 terms (i1,j1,k1) and (i2,j2,k2) that can be
    # "absorbed" into existing terms.
    
    # Let me try a concrete construction.
    # Consider terms (0,0,0) and (0,0,1) - same i=0, j=0, different k.
    # Schoolbook: 
    #   Term A: U[0,0]=1, V[0,0]=1, W[0,0]=1 (for (0,0,0))
    #   Term B: U[0,0]=1, V[0,1]=1, W[1,0]=1 (for (0,0,1))
    # 
    # Can we merge? Need u,v,w such that:
    #   u[0]*v[0]*w[0] = 1  (for (0,0,0): i=0,j=0,k=0)
    #   u[0]*v[1]*w[n+0] = 1  (for (0,0,1): i=0,j=0,k=1, W index = 1*2+0=2)
    #   all other products = 0
    # 
    # u[0]=1, v[0]=1, w[0]=1, v[1]=1, w[2]=1
    # Check other products: u[a]*v[b]*w[d] for other (a,b,d)
    # u is nonzero only at position 0 (i=0,j=0)
    # v is nonzero at positions 0 (j=0,k=0) and 1 (j=0,k=1)  
    # w is nonzero at positions 0 (k=0,i=0) and 2 (k=1,i=0)
    # Products:
    #   (i=0,j=0,k=0): u[0]*v[0]*w[0] = 1*1*1 = 1 ✓
    #   (i=0,j=0,k=1): u[0]*v[1]*w[2] = 1*1*1 = 1 ✓
    #   (i=0,j=0,k=2): u[0]*v[2]*w[4] = 1*0*0 = 0 ✓ (v[2] is for j=0,k=2)
    #   etc.
    # But wait, we also need to check that this doesn't affect OTHER (i,j,k) entries.
    # For (i'=0,j'=0,k'=2): u[0]*v[2]*w[4] = 1*0*0 = 0 ✓
    # For (i'=1,j'=0,k'=0): u[3]*v[0]*w[0] = 0*1*1 = 0 ✓ (u[3] is i=1,j=0)
    # Great! So this single term handles both (0,0,0) and (0,0,1).
    
    # Similarly, we can merge (0,0,2) and (0,0,3):
    #   u[0]=1, v[2]=1, v[3]=1, w[4]=1, w[6]=1
    #   (0,0,2): u[0]*v[2]*w[4] = 1 ✓
    #   (0,0,3): u[0]*v[3]*w[6] = 1 ✓
    
    # And (0,0,4) stays alone: u[0]=1, v[4]=1, w[8]=1
    
    # Wait, but I used u[0]=1 in multiple terms. That's fine because each
    # "term" c has its own set of U, V, W values.
    
    # So:
    # Term 0: handles (0,0,0) and (0,0,1): U[0]=1, V[0]=1, V[1]=1, W[0]=1, W[2]=1
    # Term 1: handles (0,0,2) and (0,0,3): U[0]=1, V[2]=1, V[3]=1, W[4]=1, W[6]=1
    # Term 2: handles (0,0,4): U[0]=1, V[4]=1, W[8]=1
    # That's 3 terms instead of 5 for the (0,0,k) entries. Saved 2!
    
    # Now let me construct the full rank-28 decomposition.
    # Original 30 terms grouped by (i,j):
    # (0,0,k) for k=0..4: 5 terms -> 3 terms (saved 2)
    # (0,1,k) for k=0..4: 5 terms -> keep as 5
    # (0,2,k) for k=0..4: 5 terms -> keep as 5
    # (1,0,k) for k=0..4: 5 terms -> keep as 5
    # (1,1,k) for k=0..4: 5 terms -> keep as 5
    # (1,2,k) for k=0..4: 5 terms -> keep as 5
    # Total: 3 + 5*5 = 28. 
    
    # Let me build this.
    
    U = [[0] * 28 for _ in range(6)]  # n*m = 6
    V = [[0] * 28 for _ in range(15)]  # m*p = 15
    W = [[0] * 28 for _ in range(10)]  # p*n = 10
    
    # Term 0: (0,0,0) and (0,0,1)
    U[0][0] = 1  # i=0, j=0
    V[0][0] = 1  # j=0, k=0
    V[1][0] = 1  # j=0, k=1
    W[0][0] = 1  # k=0, i=0
    W[2][0] = 1  # k=1, i=0
    
    # Term 1: (0,0,2) and (0,0,3)
    U[0][1] = 1
    V[2][1] = 1  # j=0, k=2
    V[3][1] = 1  # j=0, k=3
    W[4][1] = 1  # k=2, i=0
    W[6][1] = 1  # k=3, i=0
    
    # Term 2: (0,0,4)
    U[0][2] = 1
    V[4][2] = 1  # j=0, k=4
    W[8][2] = 1  # k=4, i=0
    
    # Terms 3-7: (0,1,k) for k=0..4
    for k in range(5):
        c = 3 + k
        U[1][c] = 1  # i=0, j=1
        V[5 + k][c] = 1  # j=1, k=k
        W[k * 2 + 0][c] = 1  # k=k, i=0
    
    # Terms 8-12: (0,2,k) for k=0..4
    for k in range(5):
        c = 8 + k
        U[2][c] = 1  # i=0, j=2
        V[10 + k][c] = 1  # j=2, k=k
        W[k * 2 + 0][c] = 1  # k=k, i=0
    
    # Terms 13-17: (1,0,k) for k=0..4
    for k in range(5):
        c = 13 + k
        U[3][c] = 1  # i=1, j=0
        V[k][c] = 1  # j=0, k=k
        W[k * 2 + 1][c] = 1  # k=k, i=1
    
    # Terms 18-22: (1,1,k) for k=0..4
    for k in range(5):
        c = 18 + k
        U[4][c] = 1  # i=1, j=1
        V[5 + k][c] = 1  # j=1, k=k
        W[k * 2 + 1][c] = 1  # k=k, i=1
    
    # Terms 23-27: (1,2,k) for k=0..4
    for k in range(5):
        c = 23 + k
        U[5][c] = 1  # i=1, j=2
        V[10 + k][c] = 1  # j=2, k=k
        W[k * 2 + 1][c] = 1  # k=k, i=1
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, fall back to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
