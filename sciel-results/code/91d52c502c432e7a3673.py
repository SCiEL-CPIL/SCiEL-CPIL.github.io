import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with rank 7.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # Strassen's 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)

    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # u vectors (A11, A12, A21, A22) indices 0,1,2,3.
    U[0, 0] = 1; U[3, 0] = 1    # M1: A11+A22
    U[2, 1] = 1; U[3, 1] = 1    # M2: A21+A22
    U[0, 2] = 1                 # M3: A11
    U[3, 3] = 1                 # M4: A22
    U[0, 4] = 1; U[1, 4] = 1    # M5: A11+A12
    U[0, 5] = -1; U[2, 5] = 1   # M6: -A11+A21
    U[1, 6] = 1; U[3, 6] = -1   # M7: A12-A22

    # v vectors (B11, B12, B21, B22) indices 0,1,2,3.
    V[0, 0] = 1; V[3, 0] = 1    # M1: B11+B22
    V[0, 1] = 1                 # M2: B11
    V[1, 2] = 1; V[3, 2] = -1   # M3: B12-B22
    V[2, 3] = 1; V[3, 3] = -1   # M4: B21-B11
    V[3, 4] = 1                 # M5: B22
    V[0, 5] = 1; V[1, 5] = 1    # M6: B11+B12
    V[2, 6] = 1; V[3, 6] = 1    # M7: B21+B22

    # w vectors (C11, C12, C21, C22) indices 0,1,2,3.
    W[0, 0] = 1; W[3, 0] = 1    # M1: C11+C22
    W[2, 1] = 1; W[3, 1] = -1   # M2: C21-C22
    W[1, 2] = 1; W[3, 2] = 1    # M3: C12+C22
    W[0, 3] = 1; W[2, 3] = 1    # M4: C11+C21
    W[0, 4] = -1; W[1, 4] = 1   # M5: -C11+C12
    W[3, 5] = 1                 # M6: C22
    W[0, 6] = 1                 # M7: C11

    return U, V, W


def _strassen_4x4x4():
    """
    Return U, V, W for 4x4x4 matrix multiplication with rank 49.
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    # Strassen 4x4: 7 block-level multiplications, each 2x2 Strassen (rank 7)
    # Block A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # Block B indices: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # Block C indices: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)

    # u vectors for A blocks (A00, A01, A10, A11)
    u_blocks = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: -A00+A10
        [0, 1, 0, -1],  # M7: A01-A11
    ]

    # v vectors for B blocks (B00, B01, B10, B11)
    v_blocks = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]

    # w vectors for C blocks (C00, C01, C10, C11)
    w_blocks = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]

    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            # For each element in the 2x2 blocks
            for a in range(2):       # block row of A
                for b in range(2):   # block col of A
                    for c in range(2):  # block row of B
                        for d in range(2):  # block col of B
                            for i in range(2):  # element row within block
                                for j in range(2):  # element col within block
                                    for k in range(2):  # element row within B block
                                        # U index: (a*2+i, b*2+j)
                                        u_idx = (a * 2 + i) * 4 + (b * 2 + j)
                                        # V index: (c*2+k, d*2+j) -- wait, need to be careful
                                        # V[j,k] where j is col of A = row of B, k is col of B
                                        # Actually V index is (j_block*2+j_elem, k_block*2+k_elem)
                                        # j_block = b (col of A = row of B), k_block = d (col of B)
                                        v_idx = (b * 2 + i) * 4 + (d * 2 + k)  # hmm, need to think
                                        # Actually: V[j,k] where j is from A's column, k is from B's column
                                        # j = b*2 + i_elem_in_block, but i is row of A...
                                        # Let me reconsider the indexing.

    # Let me redo this more carefully.
    # U[i,j] where i is row of A (0..3), j is col of A (0..3)
    # V[j,k] where j is col of A = row of B (0..3), k is col of B (0..3)
    # W[k,i] where k is col of B (0..3), i is row of A (0..3)

    # Block structure:
    # A = [[A00, A01], [A10, A11]] where A00 is rows 0-1, cols 0-1
    # B = [[B00, B01], [B10, B11]] where B00 is rows 0-1, cols 0-1
    # C = [[C00, C01], [C10, C11]] where C00 is rows 0-1, cols 0-1

    # For block (a,b) of A: rows a*2 to a*2+1, cols b*2 to b*2+1
    # For block (c,d) of B: rows c*2 to c*2+1, cols d*2 to d*2+1
    # For block (e,f) of C: rows e*2 to e*2+1, cols f*2 to f*2+1

    # U[i,j] = sum over terms
    # V[j,k] = sum over terms
    # W[k,i] = sum over terms

    # Term (r,s) where r is block-level Strassen index, s is 2x2 Strassen index
    # Block A index for term r: u_blocks[r] gives coefficients for A00,A01,A10,A11
    # Block B index for term r: v_blocks[r] gives coefficients for B00,B01,B10,B11
    # Block C index for term r: w_blocks[r] gives coefficients for C00,C01,C10,C11

    # For the 2x2 Strassen within each block:
    # U2[i',j',s] where i',j' are 0 or 1 (element within 2x2 block)
    # V2[j',k',s]
    # W2[k',i',s]

    # Full U[i,j] for i = a*2+i', j = b*2+j':
    # U[i,j] += u_blocks[r][a*4+b] * U2[i',j',s]
    # But only if the block coefficient is nonzero.

    # Full V[j,k] for j = b*2+j', k = d*2+k':
    # V[j,k] += v_blocks[r][b*4+d] * V2[j',k',s]

    # Full W[k,i] for k = d*2+k', i = a*2+i':
    # W[k,i] += w_blocks[r][a*4+d] * W2[k',i',s]

    # Wait, this doesn't quite work because the block indices for U, V, W are different.
    # Let me think again.

    # Actually, the standard Strassen 4x4 decomposition works as follows:
    # We have 7 "big" multiplications M1..M7, each of which is a 2x2 matrix multiply.
    # Each 2x2 multiply uses 7 Strassen terms, giving 49 total terms.

    # For big multiplication r (r=0..6):
    #   A_r = u_blocks[r][0]*A00 + u_blocks[r][1]*A01 + u_blocks[r][2]*A10 + u_blocks[r][3]*A11
    #   B_r = v_blocks[r][0]*B00 + v_blocks[r][1]*B01 + v_blocks[r][2]*B10 + v_blocks[r][3]*B11
    #   M_r = A_r * B_r  (2x2 matrix multiply)
    #   C += w_blocks[r][0]*M_r[0,0] to C00, w_blocks[r][1]*M_r[0,0] to C01, etc.

    # Wait, that's not right either. Let me look up the standard Strassen 4x4.

    # Actually, I think the correct approach is:
    # The 4x4 matrix multiply can be decomposed using 7 Strassen-like formulas at the block level,
    # where each "product" is a 2x2 matrix multiply, which itself uses 7 Strassen terms.

    # Let me just verify by construction. I'll build it and validate.

    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            # For each element position
            for i in range(4):
                for j in range(4):
                    a = i // 2  # block row of A
                    b = j // 2  # block col of A
                    i_prime = i % 2  # element row within block
                    j_prime = j % 2  # element col within block
                    u_coeff = u_blocks[r][a * 2 + b]
                    if u_coeff != 0:
                        U4[i * 4 + j, col] = u_coeff * U2[i_prime * 2 + j_prime, s]

            for j in range(4):
                for k in range(4):
                    b = j // 2  # block col of A = block row of B
                    d = k // 2  # block col of B
                    j_prime = j % 2
                    k_prime = k % 2
                    v_coeff = v_blocks[r][b * 2 + d]
                    if v_coeff != 0:
                        V4[j * 4 + k, col] = v_coeff * V2[j_prime * 2 + k_prime, s]

            for k in range(4):
                for i in range(4):
                    d = k // 2  # block col of B
                    a = i // 2  # block row of A
                    k_prime = k % 2
                    i_prime = i % 2
                    w_coeff = w_blocks[r][a * 2 + d]
                    if w_coeff != 0:
                        W4[k * 4 + i, col] = w_coeff * W2[k_prime * 2 + i_prime, s]

    return U4, V4, W4


def construct():
    t0 = time.time()
    n, m, p = N, M, P

    # Rank 85: 49 (Strassen 4x4x4) + 36 (schoolbook for remaining)
    R = 85
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Part 1: Strassen 4x4x4 (rank 49) for i<4, j<4, k<4
    U4, V4, W4 = _strassen_4x4x4()
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4

    c = 49

    # Part 2: Schoolbook for remaining terms
    # Terms: (i,j,k) where NOT (i<4 AND j<4 AND k<4)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 4 and j < 4 and k < 4:
                    continue
                if c >= R:
                    break
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
            if c >= R:
                break
        if c >= R:
            break

    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        if not _validate(U, V, W):
            raise ValueError("All decompositions failed")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
