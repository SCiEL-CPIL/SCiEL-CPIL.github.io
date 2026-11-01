"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomposition():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c] for input A[i,j], V[j*2+k, c] for input B[j,k], W[k*2+i, c] for output C[k*2+i]... 
    Wait, the convention is W[k*n+i, r] where n=m=p=2.
    So W index is k*2+i where k is column of B (which is row of C) and i is row of A (which is column of C).
    Actually let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    For matrix mult C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    The tensor is T[i,j,k] = delta[i,k]... no.
    
    Let me think about the tensor decomposition convention more carefully.
    
    We want: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] ... no that's not right either.
    
    Actually the standard tensor decomposition for matrix multiplication:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where T[i,j,k] = 1 if this corresponds to a valid (i,j,k) triple for the multiplication.
    
    Wait, I think the convention is:
    The multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    
    No, that's not right. The matrix multiplication tensor is defined such that:
    For matrices A (n x m) and B (m x p), the product C (n x p) has:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The tensor T has T[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor of shape n x m x p).
    
    A rank-r decomposition is: T[i,j,k] = sum_{r=0}^{r-1} U[i,r] * V[j,r] * W[k,r]
    
    Wait, but the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    So U is indexed by (i,j) flattened as i*m+j, V by (j,k) flattened as j*p+k, W by (k,i) flattened as k*n+i.
    
    And the decomposition is: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is the standard matrix multiplication tensor decomposition.
    
    For schoolbook: rank = n*m*p, with U[i*m+j, i*m*p + j*m*p + k] = 1, etc.
    
    OK so for Strassen 2x2:
    n=m=p=2, rank=7.
    
    Strassen's algorithm computes:
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    where:
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1]-B[1,1])
    M4 = A[1,1] * (B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1]) * B[1,1]
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    Each Mr is a scalar product of two linear combinations.
    
    So for each Mr, we have a rank-1 term in the decomposition.
    
    For M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    This contributes to C[0,0] with coefficient +1 and to C[1,1] with coefficient +1.
    
    In the tensor decomposition, for the term corresponding to M1:
    U values: U[0*2+0, r1] = 1, U[1*2+1, r1] = 1 (for A[0,0] and A[1,1])
    V values: V[0*2+0, r1] = 1, V[1*2+1, r1] = 1 (for B[0,0] and B[1,1])
    W values: W[0*2+0, r1] = 1, W[1*2+1, r1] = 1 (for C[0,0] and C[1,1])
    
    Wait, W[k*n+i, r] where n=2. So W[k*2+i, r].
    C[0,0] corresponds to k=0, i=0, so W[0*2+0, r] = W[0, r]
    C[1,1] corresponds to k=1, i=1, so W[1*2+1, r] = W[3, r]
    
    Let me verify: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For C[0,0]: sum_j sum_r U[0*2+j, r] * V[j*2+0, r] * W[0*2+0, r]
    = sum_j sum_r U[j, r] * V[2j, r] * W[0, r]
    
    Hmm, this is getting complex. Let me just construct it programmatically.
    
    For each of the 7 Strassen multiplications, I'll determine the coefficients in U, V, W.
    """
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen multiplications:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # For each multiplication r, we have:
    # U coefficients for A elements
    # V coefficients for B elements  
    # W coefficients for C elements (with signs from the combination)
    
    # M1: A00+A11 times B00+B11, contributes +1 to C00, +1 to C11
    # U: A00 (idx 0) = 1, A11 (idx 3) = 1
    # V: B00 (idx 0) = 1, B11 (idx 3) = 1
    # W: C00 (k=0,i=0, idx 0) = 1, C11 (k=1,i=1, idx 3) = 1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (A10+A11) * B00, contributes +1 to C10
    # U: A10 (idx 2) = 1, A11 (idx 3) = 1
    # V: B00 (idx 0) = 1
    # W: C10 (k=0,i=1, idx 1) = 1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1
    
    # M3: A00 * (B01-B11), contributes +1 to C01, -1 to C11
    # U: A00 (idx 0) = 1
    # V: B01 (idx 1) = 1, B11 (idx 3) = -1
    # W: C01 (k=1,i=0, idx 2) = 1, C11 (k=1,i=1, idx 3) = -1
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[2][2] += 1; W[3][2] += -1
    
    # M4: A11 * (B10-B00), contributes +1 to C00, +1 to C10
    # U: A11 (idx 3) = 1
    # V: B10 (idx 2) = 1, B00 (idx 0) = -1
    # W: C00 (k=0,i=0, idx 0) = 1, C10 (k=0,i=1, idx 1) = 1
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # M5: (A00+A01) * B11, contributes -1 to C00, +1 to C01
    # U: A00 (idx 0) = 1, A01 (idx 1) = 1
    # V: B11 (idx 3) = 1
    # W: C00 (k=0,i=0, idx 0) = -1, C01 (k=1,i=0, idx 2) = 1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    # M6: (A10-A00) * (B00+B01), contributes +1 to C11
    # U: A10 (idx 2) = 1, A00 (idx 0) = -1
    # V: B00 (idx 0) = 1, B01 (idx 1) = 1
    # W: C11 (k=1,i=1, idx 3) = 1
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11) * (B10+B11), contributes +1 to C00, -1 to C11
    # U: A01 (idx 1) = 1, A11 (idx 3) = -1
    # V: B10 (idx 2) = 1, B11 (idx 3) = 1
    # W: C00 (k=0,i=0, idx 0) = 1, C11 (k=1,i=1, idx 3) = -1
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += -1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4_decomposition():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using two levels of Strassen.
    
    Level 1: Partition 4x4 into 2x2 blocks, use 7 block multiplications.
    Level 2: Each 2x2 block multiplication uses Strassen's 7 scalar multiplications.
    Total: 7 * 7 = 49 scalar multiplications.
    """
    n = m = p = 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # 4x4 matrices partitioned into 2x2 blocks:
    # A = [[A00, A01],     B = [[B00, B01],
    #      [A10, A11]]           [B10, B11]]
    # where Aij is 2x2, with Aij[a,b] = A[2*i+a, 2*j+b]
    # Similarly for B and C.
    
    # Strassen for 4x4 (block level):
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mx is a 2x2 matrix product, which we decompose using Strassen (7 scalar mults).
    
    # For each block multiplication Mx = P * Q where P and Q are 2x2 (linear combos of A/B blocks),
    # we need to:
    # 1. Determine the linear combination coefficients for P (from A blocks)
    # 2. Determine the linear combination coefficients for Q (from B blocks)
    # 3. For each of the 7 Strassen scalar mults in P*Q, add to U, V, W with appropriate signs
    
    # Block indices: block (bi, bj) of A corresponds to elements A[2*bi+a, 2*bj+b] for a,b in {0,1}
    # Flattened index in U: (2*bi+a)*4 + (2*bj+b)
    
    # Let me define the block-level Strassen terms:
    # Each term r_block (0..6) has:
    #   - P coefficients: dict mapping (bi,bj) -> coeff for A block (bi,bj)
    #   - Q coefficients: dict mapping (bi,bj) -> coeff for B block (bi,bj)
    #   - C contributions: dict mapping (ci,cj) -> coeff for C block (ci,cj)
    
    block_terms = [
        # M1: P = A00+A11, Q = B00+B11, C: C00+=1, C11+=1
        {'P': {(0,0): 1, (1,1): 1}, 'Q': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M2: P = A10+A11, Q = B00, C: C10+=1
        {'P': {(1,0): 1, (1,1): 1}, 'Q': {(0,0): 1}, 'C': {(1,0): 1}},
        # M3: P = A00, Q = B01-B11, C: C01+=1, C11+=-1
        {'P': {(0,0): 1}, 'Q': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): -1}},
        # M4: P = A11, Q = B10-B00, C: C00+=1, C10+=1
        {'P': {(1,1): 1}, 'Q': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # M5: P = A00+A01, Q = B11, C: C00+=-1, C01+=1
        {'P': {(0,0): 1, (0,1): 1}, 'Q': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # M6: P = A10-A00, Q = B00+B01, C: C11+=1
        {'P': {(1,0): 1, (0,0): -1}, 'Q': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # M7: P = A01-A11, Q = B10+B11, C: C00+=1, C11+=-1
        {'P': {(0,1): 1, (1,1): -1}, 'Q': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): -1}},
    ]
    
    for rb, term in enumerate(block_terms):
        P_coeffs = term['P']  # (bi,bj) -> coeff
        Q_coeffs = term['Q']  # (bi,bj) -> coeff
        C_coeffs = term['C']  # (ci,cj) -> coeff
        
        for rs in range(7):
            r = rb * 7 + rs  # global rank index
            
            # For the rs-th Strassen scalar multiplication in the 2x2 product:
            # U2 coefficients tell us which elements of P (2x2) are involved
            # V2 coefficients tell us which elements of Q (2x2) are involved
            # W2 coefficients tell us which elements of the result (2x2) get which sign
            
            # P is a 2x2 matrix where P[a,b] = sum over (bi,bj) in P_coeffs: coeff * A[2*bi+a, 2*bj+b]
            # Q is a 2x2 matrix where Q[a,b] = sum over (bi,bj) in Q_coeffs: coeff * B[2*bi+a, 2*bj+b]
            # Result R is 2x2 where R[a,b] = P[a,b'] * Q[b',b] summed... 
            # Actually R = P * Q (2x2 matrix product)
            
            # The rs-th Strassen term computes: (linear combo of P elements) * (linear combo of Q elements)
            # This contributes to R with coefficients given by W2.
            
            # For each element of P that appears in the rs-th Strassen U2 term:
            # U2[row_in_P][rs] gives the coefficient
            # row_in_P = a*2+b where P[a,b] is the element
            # P[a,b] = sum over (bi,bj): P_coeffs[(bi,bj)] * A[2*bi+a, 2*bj+b]
            # So U[flattened_index_of_A[2*bi+a, 2*bj+b]][r] += P_coeffs[(bi,bj)] * U2[a*2+b][rs]
            
            for (bi, bj), pc in P_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        # P[a,b] includes A[2*bi+a, 2*bj+b] with coefficient pc
                        u2_coeff = U2[a * 2 + b][rs]
                        if u2_coeff != 0:
                            # A element index: (2*bi+a) * 4 + (2*bj+b)
                            ai = 2 * bi + a
                            aj = 2 * bj + b
                            U[ai * 4 + aj][r] += pc * u2_coeff
            
            # Similarly for Q
            for (bi, bj), qc in Q_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        # Q[a,b] includes B[2*bi+a, 2*bj+b] with coefficient qc
                        v2_coeff = V2[a * 2 + b][rs]
                        if v2_coeff != 0:
                            bi_idx = 2 * bi + a
                            bj_idx = 2 * bj + b
                            V[bi_idx * 4 + bj_idx][r] += qc * v2_coeff
            
            # For W: the result R[a,b] contributes to C blocks
            # R is the 2x2 product P*Q
            # R[a,b] = sum_c P[a,c] * Q[c,b]
            # The rs-th Strassen term contributes to R with W2 coefficients
            # W2[k*2+i][rs] where k is row of R, i is column of R? 
            # Wait, W2 index is k*n+i where n=2, so k*2+i.
            # R[k,i] gets coefficient W2[k*2+i][rs] from the rs-th term.
            
            # R[a,b] contributes to C blocks:
            # R is a 2x2 block result. If this is for C block (ci,cj), then
            # R[a,b] = C[2*ci+a, 2*cj+b]
            # So W[flattened_index_of_C[2*ci+a, 2*cj+b]][r] += C_coeffs[(ci,cj)] * W2[a*2+b][rs]
            # Wait, W index is k*n+i = k*4+i where k is column index, i is row index of C.
            # C[row, col] -> W[col*4+row][r]
            
            for (ci, cj), cc in C_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        # R[a,b] = C[2*ci+a, 2*cj+b]
                        w2_coeff = W2[a * 2 + b][rs]
                        if w2_coeff != 0:
                            crow = 2 * ci + a
                            ccol = 2 * cj + b
                            # W index: ccol * 4 + crow
                            W[ccol * 4 + crow][r] += cc * w2_coeff
    
    return U, V, W

def construct():
    # Try rank-49 Strassen decomposition
    U, V, W = _strassen_4x4_decomposition()
    n, m, p = DIMENSIONS
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
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
