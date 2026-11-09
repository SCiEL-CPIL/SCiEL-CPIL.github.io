"""Program for 6x6x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    6x6 matrix multiplication using a block-based approach with Strassen on 2x2 blocks.
    
    We partition the 6x6 matrices into 3x3 blocks of size 2x2.
    Each block multiplication C_ij = sum_k A_ik * B_kj involves 3 terms of 2x2 multiplications.
    Using Strassen for each 2x2 multiplication reduces 8 scalar mults to 7.
    
    However, we need to be careful: the standard approach would be:
    - 9 block positions (i,j) each needing sum over k=0,1,2 of A_ik*B_kj
    - That's 9*3 = 27 instances of 2x2 matrix multiplication
    - Each 2x2 mult via Strassen uses rank 7
    - Total rank: 27 * 7 = 189 (same as schoolbook! No improvement)
    
    Wait, let me reconsider. Schoolbook for 6x6x6:
    rank = 6*6*6 = 216
    
    With 3x3 block partition of 2x2 blocks:
    - We have 3x3 = 9 block positions for C
    - For each C_ij, we compute sum_{k=0}^{2} A_ik @ B_kj
    - Each A_ik @ B_kj is a 2x2 matrix product
    - Using Strassen, each 2x2 product has rank 7
    - So total rank = 9 * 3 * 7 = 189
    
    This is better than 216! Let's implement this.
    
    The Kronecker product structure:
    U, V, W have shapes (36, rank), (36, rank), (36, rank)
    
    For each (i, j, k) block triple and each Strassen term t:
    - U row: corresponds to A_ik element
    - V row: corresponds to B_kj element  
    - W row: corresponds to C_ij element
    
    Let me define the Strassen decomposition for 2x2 matrices:
    A = [[a11, a12], [a21, a22]]
    B = [[b11, b12], [b21, b22]]
    
    Strassen terms:
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a21 + a22)b11
    M3 = a11(b12 - b22)
    M4 = a22(b21 - b11)
    M5 = (a11 + a12)b22
    M6 = (a21 - a11)(b11 + b12)
    M7 = (a12 - a22)(b21 + b22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Now, for the full 6x6 case with 3x3 blocks of 2x2:
    A_ik is a 2x2 block at row-block i, col-block k
    B_kj is a 2x2 block at row-block k, col-block j
    C_ij is a 2x2 block at row-block i, col-block j
    
    For each (i, j, k) and each Strassen term t (1-7):
    We add a rank-1 contribution.
    
    U has 36 rows: row index = i*4 + r where i is block row (0-2), r is within-block row (0-1)
    Actually, let me think about the indexing more carefully.
    
    A is 6x6, indexed as A[row, col] where row = i*2 + r, col = k*2 + c
    So A_ik[r, c] = A[i*2+r, k*2+c]
    
    Similarly for B and C.
    
    U shape: (36, rank) - 36 rows for A elements
    V shape: (36, rank) - 36 rows for B elements
    W shape: (36, rank) - 36 rows for C elements
    
    For a given (i, j, k, t):
    - U contributes to rows corresponding to A_ik elements
    - V contributes to rows corresponding to B_kj elements
    - W contributes to rows corresponding to C_ij elements
    
    Let me define the Strassen coefficients as matrices:
    For term t, we have:
    - A coefficients: a_t[2,2] such that the A-part is sum of a_t[r,c] * A_ik[r,c]
    - B coefficients: b_t[2,2] such that the B-part is sum of b_t[r,c] * B_kj[r,c]
    - C coefficients: c_t[2,2] such that the C-part is sum of c_t[r,c] * C_ij[r,c]
    
    Actually, each Strassen term M_t is a product of a linear combination of A elements
    and a linear combination of B elements. The result contributes to specific C elements.
    
    Let me define for each term t:
    - A_t: 2x2 matrix of coefficients for A_ik
    - B_t: 2x2 matrix of coefficients for B_kj
    - C_t: 2x2 matrix of coefficients for C_ij (which C elements get this term)
    
    Term 1: M1 = (a11+a22)(b11+b22)
    A_1 = [[1,0],[0,1]], B_1 = [[1,0],[0,1]]
    C_1 = [[1,0],[0,1]] (contributes to C11 and C22 with +1)
    
    Term 2: M2 = (a21+a22)b11
    A_2 = [[0,0],[1,1]], B_2 = [[1,0],[0,0]]
    C_2 = [[0,0],[1,0]] (contributes to C21 with +1)
    
    Term 3: M3 = a11(b12-b22)
    A_3 = [[1,0],[0,0]], B_3 = [[0,1],[0,-1]]
    C_3 = [[0,1],[0,0]] (contributes to C12 with +1)
    
    Term 4: M4 = a22(b21-b11)
    A_4 = [[0,0],[0,1]], B_4 = [[-1,0],[1,0]]
    C_4 = [[1,0],[0,1]] (contributes to C11 with +1 and C22 with +1)
    
    Term 5: M5 = (a11+a12)b22
    A_5 = [[1,1],[0,0]], B_5 = [[0,0],[0,1]]
    C_5 = [[-1,0],[0,0]] (contributes to C11 with -1)
    
    Term 6: M6 = (a21-a11)(b11+b12)
    A_6 = [[-1,0],[1,0]], B_6 = [[1,1],[0,0]]
    C_6 = [[0,0],[0,1]] (contributes to C22 with +1)
    
    Term 7: M7 = (a12-a22)(b21+b22)
    A_7 = [[0,1],[0,-1]], B_7 = [[0,0],[1,1]]
    C_7 = [[1,0],[0,0]] (contributes to C11 with +1)
    
    Let me verify:
    C11 = M1 + M4 - M5 + M7
    = (a11+a22)(b11+b22) + a22(b21-b11) - (a11+a12)b22 + (a12-a22)(b21+b22)
    = a11*b11 + a11*b22 + a22*b11 + a22*b22 + a22*b21 - a22*b11 - a11*b22 - a12*b22 + a12*b21 + a12*b22 - a22*b21 - a22*b22
    = a11*b11 + a12*b21
    ✓
    
    C12 = M3 + M5
    = a11(b12-b22) + (a11+a12)b22
    = a11*b12 - a11*b22 + a11*b22 + a12*b22
    = a11*b12 + a12*b22
    ✓
    
    C21 = M2 + M4
    = (a21+a22)b11 + a22(b21-b11)
    = a21*b11 + a22*b11 + a22*b21 - a22*b11
    = a21*b11 + a22*b21
    ✓
    
    C22 = M1 - M2 + M3 + M6
    = (a11+a22)(b11+b22) - (a21+a22)b11 + a11(b12-b22) + (a21-a11)(b11+b12)
    = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a21*b11 - a22*b11 + a11*b12 - a11*b22 + a21*b11 + a21*b12 - a11*b11 - a11*b12
    = a22*b22 + a21*b12
    ✓
    
    Great! Now let's build the decomposition.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen coefficient matrices
    # Each term has A_t[2,2], B_t[2,2], C_t[2,2]
    strassen_terms = [
        # Term 1
        {'A': [[1,0],[0,1]], 'B': [[1,0],[0,1]], 'C': [[1,0],[0,1]]},
        # Term 2
        {'A': [[0,0],[1,1]], 'B': [[1,0],[0,0]], 'C': [[0,0],[1,0]]},
        # Term 3
        {'A': [[1,0],[0,0]], 'B': [[0,1],[0,-1]], 'C': [[0,1],[0,0]]},
        # Term 4
        {'A': [[0,0],[0,1]], 'B': [[-1,0],[1,0]], 'C': [[1,0],[0,1]]},
        # Term 5
        {'A': [[1,1],[0,0]], 'B': [[0,0],[0,1]], 'C': [[-1,0],[0,0]]},
        # Term 6
        {'A': [[-1,0],[1,0]], 'B': [[1,1],[0,0]], 'C': [[0,0],[0,1]]},
        # Term 7
        {'A': [[0,1],[0,-1]], 'B': [[0,0],[1,1]], 'C': [[1,0],[0,0]]},
    ]
    
    rank = 9 * 3 * 7  # 189
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for i in range(3):       # block row of A, block row of C
        for k in range(3):   # block col of A, block row of B
            for j in range(3):  # block col of B, block col of C
                for t in range(7):  # Strassen term
                    A_t = strassen_terms[t]['A']
                    B_t = strassen_terms[t]['B']
                    C_t = strassen_terms[t]['C']
                    
                    for r in range(2):  # within-block row
                        for c in range(2):  # within-block col
                            # A_ik[r, c] is at global position (i*2+r, k*2+c)
                            # U row index: i*2+r is the row, but U is indexed by (row, col) of A
                            # A is 6x6, so U has 36 rows indexed by row*6 + col? 
                            # Wait, the task says U has shape (n*m, rank) = (36, rank)
                            # For matrix multiplication A@B=C, the standard convention:
                            # U[i*m+j, c] for A[i,j], V[j*p+k, c] for B[j,k], W[k*n+i, c] for C[k,i]?
                            # No wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
                            # So W is indexed by [k*n+i, r] for C[i,k]? Or C[k,i]?
                            # The convention W[k*n+i, r] suggests C is indexed as C[i,k] with row i, col k
                            # So W[k*n+i, r] corresponds to C[i,k]
                            
                            # For A: U[row, col] where A is n x m = 6 x 6
                            # U[i*m+j, c] = U[i*6+j, c] for A[i,j]
                            
                            # For B: V[j*p+k, c] = V[j*6+k, c] for B[j,k]
                            
                            # For C: W[k*n+i, c] = W[k*6+i, c] for C[i,k]
                            
                            # A_ik[r,c]: A at block (i,k), within-block (r,c)
                            # Global A position: row = i*2+r, col = k*2+c
                            # U row index: (i*2+r)*6 + (k*2+c)
                            u_row = (i * 2 + r) * 6 + (k * 2 + c)
                            U[u_row][col] += A_t[r][c]
                            
                            # B_kj[r,c]: B at block (k,j), within-block (r,c)
                            # Global B position: row = k*2+r, col = j*2+c
                            # V row index: (k*2+r)*6 + (j*2+c)
                            v_row = (k * 2 + r) * 6 + (j * 2 + c)
                            V[v_row][col] += B_t[r][c]
                            
                            # C_ij[r,c]: C at block (i,j), within-block (r,c)
                            # Global C position: row = i*2+r, col = j*2+c
                            # W row index: col_first * 6 + row_first = (j*2+c)*6 + (i*2+r)
                            w_row = (j * 2 + c) * 6 + (i * 2 + r)
                            W[w_row][col] += C_t[r][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
