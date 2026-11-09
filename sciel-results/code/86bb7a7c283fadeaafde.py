import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition by breaking down the problem into smaller blocks.
    # A is 3x4, B is 4x6.
    # Split A into A1 (2x4) and A2 (1x4).
    # Split B into B1 (4x4) and B2 (4x2).
    
    # Blocks:
    # C11 = A1 B1 (2x4x4)
    # C12 = A1 B2 (2x4x2)
    # C21 = A2 B1 (1x4x4) -> Rank 1
    # C22 = A2 B2 (1x4x2) -> Rank 1
    
    # Helper to add a rank-1 term to U, V, W
    # u: vector of length n_block (part of n)
    # v: vector of length m_block (part of m)
    # w: vector of length p_block (part of p)
    # offsets: i_off, k_off, j_off
    
    U = np.zeros((n * m, 100)) # Pre-allocate large enough, will trim
    V = np.zeros((m * p, 100))
    W = np.zeros((p * n, 100))
    
    rank = 0
    
    def add_term(u, v, w, i_off, k_off, j_off):
        nonlocal rank
        # u is length n_sub, v is length m_sub, w is length p_sub
        # In the global tensor T[i, k, j], this term contributes u[i'] v[k'] w[j']
        # to T[i_off+i', k_off+k', j_off+j']
        
        # U index: i*m + k? No.
        # U shape (n*m, rank). U[i*m+k, r] corresponds to coefficient of A[i,k] in X_r.
        # V shape (m*p, rank). V[k*p+j, r] corresponds to coefficient of B[k,j] in Y_r.
        # W shape (p*n, rank). W[j*n+i, r] corresponds to coefficient of P_r in C[i,j].
        # Note: W index is j*n + i based on "W[k*n+i,r]" in prompt? 
        # Prompt: "W[k*n+i,r]". Here k is the index for p? 
        # "output factor follows the official trace convention W[k*n+i,r]".
        # Usually C is n x p. Index is i (row), j (col).
        # If W is (p*n, rank), index is j*n + i.
        # The prompt says W[k*n+i]. So k is the column index of C (which is p), i is row index (n).
        
        for ii in range(len(u)):
            for kk in range(len(v)):
                # U[i_off+ii, m + k_off+kk, r] = u[ii] * v[kk]?
                # No, the term is u_ii v_kk w_jj.
                # X_r = sum u_ii v_kk A_ii,kk?
                # If we define X_r = sum_{ii,kk} u_ii v_kk A_ii,kk,
                # and Y_r = sum_{kk,jj} ... wait.
                # The standard rank-1 term in tensor decomposition T = sum u_r v_r w_r
                # means T[i,k,j] = sum_r u_r[i] v_r[k] w_r[j].
                # Then C[i,j] = sum_{i,k,j} T[i,k,j] A[i,k] B[k,j]
                # = sum_r sum_{i,k,j} u_r[i] v_r[k] w_r[j] A[i,k] B[k,j]
                # This is NOT separable into X_r Y_r.
                
                # HOWEVER, the prompt asks for U, V, W such that they "exactly decompose the matrix multiplication tensor".
                # And the score is the rank.
                # The standard interpretation in these coding challenges (like SCiEL) is that 
                # U, V, W define the bilinear algorithm.
                # X_r = sum_{i,k} U[i*m+k, r] A[i,k]
                # Y_r = sum_{k,j} V[k*p+j, r] B[k,j]
                # C[i,j] = sum_r W[j*n+i, r] X_r Y_r
                
                # This corresponds to a tensor decomposition where each term r is 
                # NOT a simple u_r v_r w_r, but rather a "bilinear form" term.
                # Specifically, the tensor T is decomposed into a sum of terms 
                # that are products of a linear form in A, a linear form in B, 
                # and a linear form in C (output).
                
                # So for each r, we have:
                # u_r in R^(n*m) (coeffs for A)
                # v_r in R^(m*p) (coeffs for B)
                # w_r in R^(p*n) (coeffs for C)
                
                # We need to populate U[:, r], V[:, r], W[:, r].
                
                pass

    # We will manually construct the terms for Strassen and the rank-1 blocks.
    
    # Strassen M_{2,2,2} terms (7 terms)
    # A is 2x2, B is 2x2.
    # Indices i in 0..1, k in 0..1, j in 0..1.
    
    def get_strassen_terms():
        terms = []
        # M1: (A00+A11)(B00+B11)
        u = [1, 0, 0, 1] # A00, A01, A10, A11
        v = [1, 0, 0, 1] # B00, B01, B10, B11
        w = [1, 0, 0, 1] # C00, C01, C10, C11 (j*n+i: 0*2+0=0, 1*2+0=2, 0*2+1=1, 1*2+1=3)
        # W indices: j*n+i.
        # C00 (i=0,j=0) -> idx 0
        # C01 (i=0,j=1) -> idx 2
        # C10 (i=1,j=0) -> idx 1
        # C11 (i=1,j=1) -> idx 3
        # M1 contributes to C00 and C11.
        w = [1, 0, 0, 1] # C00=1, C10=0, C01=0, C11=1.
        # Wait, w vector order in W is j*n+i.
        # w[0] is coeff for C00. w[1] for C10. w[2] for C01. w[3] for C11.
        # M1 = (A00+A11)(B00+B11).
        # C00 += M1. C11 += M1.
        # So w = [1, 0, 0, 1].
        terms.append((u, v, w))
        
        # M2: (A10+A11)B00
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 1, 0, 0] # C10 += M2
        terms.append((u, v, w))
        
        # M3: A00(B01-B11)
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 0, 1, 0] # C01 += M3
        terms.append((u, v, w))
        
        # M4: A11(B10-B00)
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 0, 0] # C00 += M4
        terms.append((u, v, w))
        
        # M5: (A00+A01)B11
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [0, 0, 1, -1] # C01 += M5, C11 -= M5
        terms.append((u, v, w))
        
        # M6: (A10-A00)(B00+B01)
        u = [-1, 0, 1, 0]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C11 += M6
        terms.append((u, v, w))
        
        # M7: (A01-A11)(B10+B11)
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # C00 += M7
        terms.append((u, v, w))
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # Block 1: C11 = A1 B1 (2x4x4)
    # A1 is 2x4 (rows 0,1 of A). B1 is 4x4 (cols 0..3 of B).
    # Split B1 into B1a (4x2, cols 0,1) and B1b (4x2, cols 2,3).
    # C11a = A1 B1a (2x4x2). C11b = A1 B1b (2x4x2).
    
    # Sub-block 1.1: A1 B1a (2x4x2)
    # Split A1 into A1a (2x2, cols 0,1) and A1b (2x2, cols 2,3).
    # B1a into B1a_top (2x2, rows 0,1) and B1a_bot (2x2, rows 2,3).
    # C11a = A1a B1a_top + A1b B1a_bot.
    
    # Term 1.1.1: A1a B1a_top (2x2x2)
    # A1a is A[0:2, 0:2]. B1a_top is B[0:2, 0:2].
    # i_off=0, k_off=0, j_off=0.
    for u, v, w in strassen_terms:
        U[0*4+0:2*4+2, rank] = 0 # Init
        V[0*6+0:2*6+2, rank] = 0
        W[0*3+0:2*3+2, rank] = 0
        # Map u (len 4) to U indices i*4+k
        # u[0]=A00, u[1]=A01, u[2]=A10, u[3]=A11
        U[0*4+0, rank] = u[0]
        U[0*4+1, rank] = u[1]
        U[1*4+0, rank] = u[2]
        U[1*4+1, rank] = u[3]
        
        # Map v (len 4) to V indices k*6+j
        # v[0]=B00, v[1]=B01, v[2]=B10, v[3]=B11
        V[0*6+0, rank] = v[0]
        V[0*6+1, rank] = v[1]
        V[1*6+0, rank] = v[2]
        V[1*6+1, rank] = v[3]
        
        # Map w (len 4) to W indices j*3+i
        # w[0]=C00, w[1]=C10, w[2]=C01, w[3]=C11
        W[0*3+0, rank] = w[0]
        W[0*3+1, rank] = w[1]
        W[1*3+0, rank] = w[2]
        W[1*3+1, rank] = w[3]
        rank += 1
        
    # Term 1.1.2: A1b B1a_bot (2x2x2)
    # A1b is A[0:2, 2:4]. B1a_bot is B[2:4, 0:2].
    # i_off=0, k_off=2, j_off=0.
    for u, v, w in strassen_terms:
        U[0*4+2:2*4+4, rank] = 0
        V[2*6+0:4*6+2, rank] = 0
        W[0*3+0:2*3+2, rank] = 0
        U[0*4+2, rank] = u[0]
        U[0*4+3, rank] = u[1]
        U[1*4+2, rank] = u[2]
        U[1*4+3, rank] = u[3]
        
        V[2*6+0, rank] = v[0]
        V[2*6+1, rank] = v[1]
        V[3*6+0, rank] = v[2]
        V[3*6+1, rank] = v[3]
        
        W[0*3+0, rank] = w[0]
        W[0*3+1, rank] = w[1]
        W[1*3+0, rank] = w[2]
        W[1*3+1, rank] = w[3]
        rank += 1
        
    # Sub-block 1.2: A1 B1b (2x4x2)
    # B1b is B[0:4, 2:4].
    # Split A1 into A1a (2x2, cols 0,1) and A1b (2x2, cols 2,3).
    # B1b into B1b_top (2x2, rows 0,1) and B1b_bot (2x2, rows 2,3).
    # C11b = A1a B1b_top + A1b B1b_bot.
    
    # Term 1.2.1: A1a B1b_top (2x2x2)
    # A[0:2, 0:2], B[0:2, 2:4].
    # i_off=0, k_off=0, j_off=2.
    for u, v, w in strassen_terms:
        U[0*4+0:2*4+2, rank] = 0
        V[0*6+2:2*6+4, rank] = 0
        W[2*3+0:4*3+2, rank] = 0 # j=2,3
        U[0*4+0, rank] = u[0]
        U[0*4+1, rank] = u[1]
        U[1*4+0, rank] = u[2]
        U[1*4+1, rank] = u[3]
        
        V[0*6+2, rank] = v[0]
        V[0*6+3, rank] = v[1]
        V[1*6+2, rank] = v[2]
        V[1*6+3, rank] = v[3]
        
        W[2*3+0, rank] = w[0]
        W[2*3+1, rank] = w[1]
        W[3*3+0, rank] = w[2]
        W[3*3+1, rank] = w[3]
        rank += 1
        
    # Term 1.2.2: A1b B1b_bot (2x2x2)
    # A[0:2, 2:4], B[2:4, 2:4].
    # i_off=0, k_off=2, j_off=2.
    for u, v, w in strassen_terms:
        U[0*4+2:2*4+4, rank] = 0
        V[2*6+2:4*6+4, rank] = 0
        W[2*3+0:4*3+2, rank] = 0
        U[0*4+2, rank] = u[0]
        U[0*4+3, rank] = u[1]
        U[1*4+2, rank] = u[2]
        U[1*4+3, rank] = u[3]
        
        V[2*6+2, rank] = v[0]
        V[2*6+3, rank] = v[1]
        V[3*6+2, rank] = v[2]
        V[3*6+3, rank] = v[3]
        
        W[2*3+0, rank] = w[0]
        W[2*3+1, rank] = w[1]
        W[3*3+0, rank] = w[2]
        W[3*3+1, rank] = w[3]
        rank += 1
        
    # Block 2: C12 = A1 B2 (2x4x2)
    # B2 is B[0:4, 4:6].
    # Split A1 into A1a (2x2, cols 0,1) and A1b (2x2, cols 2,3).
    # B2 into B2_top (2x2, rows 0,1) and B2_bot (2x2, rows 2,3).
    # C12 = A1a B2_top + A1b B2_bot.
    
    # Term 2.1: A1a B2_top (2x2x2)
    # A[0:2, 0:2], B[0:2, 4:6].
    # i_off=0, k_off=0, j_off=4.
    for u, v, w in strassen_terms:
        U[0*4+0:2*4+2, rank] = 0
        V[0*6+4:2*6+6, rank] = 0
        W[4*3+0:6*3+2, rank] = 0
        U[0*4+0, rank] = u[0]
        U[0*4+1, rank] = u[1]
        U[1*4+0, rank] = u[2]
        U[1*4+1, rank] = u[3]
        
        V[0*6+4, rank] = v[0]
        V[0*6+5, rank] = v[1]
        V[1*6+4, rank] = v[2]
        V[1*6+5, rank] = v[3]
        
        W[4*3+0, rank] = w[0]
        W[4*3+1, rank] = w[1]
        W[5*3+0, rank] = w[2]
        W[5*3+1, rank] = w[3]
        rank += 1
        
    # Term 2.2: A1b B2_bot (2x2x2)
    # A[0:2, 2:4], B[2:4, 4:6].
    # i_off=0, k_off=2, j_off=4.
    for u, v, w in strassen_terms:
        U[0*4+2:2*4+4, rank] = 0
        V[2*6+4:4*6+6, rank] = 0
        W[4*3+0:6*3+2, rank] = 0
        U[0*4+2, rank] = u[0]
        U[0*4+3, rank] = u[1]
        U[1*4+2, rank] = u[2]
        U[1*4+3, rank] = u[3]
        
        V[2*6+4, rank] = v[0]
        V[2*6+5, rank] = v[1]
        V[3*6+4, rank] = v[2]
        V[3*6+5, rank] = v[3]
        
        W[4*3+0, rank] = w[0]
        W[4*3+1, rank] = w[1]
        W[5*3+0, rank] = w[2]
        W[5*3+1, rank] = w[3]
        rank += 1
        
    # Block 3: C21 = A2 B1 (1x4x4)
    # A2 is A[2:3, 0:4]. B1 is B[0:4, 0:4].
    # Rank 1.
    # T[i,k,j] = 1 for i=2, all k, all j.
    # u = [1,1,1,1] (for A2)
    # v = [1,1,1,1] (for B1)
    # w = [1,1,1,1,1,1,1,1,1,1,1,1] (for C21, i=2, j=0..3)
    # W index j*3+2.
    U[2*4+0:3*4+4, rank] = 1
    V[0*6+0:4*6+4, rank] = 1
    W[0*3+2:4*3+2:3, rank] = 1 # j=0..3, i=2. Indices 2, 5, 8, 11.
    # Correct indices: 0*3+2=2, 1*3+2=5, 2*3+2=8, 3*3+2=11.
    rank += 1
    
    # Block 4: C22 = A2 B2 (1x4x2)
    # A2 is A[2:3, 0:4]. B2 is B[0:4, 4:6].
    # Rank 1.
    U[2*4+0:3*4+4, rank] = 1
    V[0*6+4:4*6+6, rank] = 1
    W[4*3+2:6*3+2:3, rank] = 1 # j=4,5, i=2. Indices 14, 17.
    rank += 1
    
    # Trim and return
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
