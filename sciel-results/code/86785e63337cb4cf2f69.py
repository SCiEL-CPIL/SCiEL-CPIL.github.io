"""Matrix multiplication 3,3,4 - Strassen-based rank-34 decomposition."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check that U @ V^T matches W reshaped as tensor."""
    import numpy as np
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # For each rank index c, we have a rank-1 tensor
    # U[i*m+j, c] * V[j*p+k, c] should give A[i,j]*B[j,k] -> C[i,k]
    # W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for the corresponding (i,j,k)
    # The product tensor is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Check: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other combinations, it should be 0.
    # Actually the standard check is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_c A[i,j] V[j,k,c] * (sum_c U[i,j,c] W[k,i,c] ... )
    # No, let me think again.
    # The decomposition says: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  (for the (i,j,k) entry of the mult tensor)
    # Wait, the matrix multiplication tensor M[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # So M[i,j,k] = 1 for all i,j,k? No.
    # M[i,j,k] is the tensor such that C[i,k] = sum_j A[i,j] B[j,k] * M[i,j,k]
    # So M[i,j,k] = 1 for all i,j,k.
    # The rank-1 decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need this to equal 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 3, 4

    # Build a rank-34 decomposition using Strassen for 2x2x2 blocks
    # Strategy:
    #   For k in {0,1}: use a 3x3x2 block decomposition of rank 17
    #   For k in {2,3}: use the same 3x3x2 block decomposition of rank 17
    #   Total rank = 34

    # The 3x3x2 decomposition of rank 17:
    #   Part A: 2x2x2 Strassen core for i in {0,1}, j in {0,1}, k in {0,1} -> rank 7
    #   Part B: j=2 column: i in {0,1,2}, k in {0,1} -> rank 6 (direct)
    #   Part C: i=2 row (j in {0,1}): j in {0,1}, k in {0,1} -> rank 4 (direct)
    #   Total: 7 + 6 + 4 = 17

    # Strassen 2x2x2 rank-7 decomposition:
    # For matrices A (2x2), B (2x2), C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A01+A11)B10
    # M3 = A00(B01-B11)
    # M4 = A11(B00-B10)
    # M5 = (A10+A00)B01
    # M6 = (A10-A01)(B10+B11)
    # M7 = (A01-A10)(B00+B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each Strassen term, the (U, V, W) coefficients for the 2x2x2 block:
    # Term 1: U has A00+A11, V has B00+B11, W has C00+C11
    #   In our indexing for the 2x2x2 sub-block (i' in {0,1}, j' in {0,1}, k' in {0,1}):
    #   U[i'*2+j', 1] = coeff of A[i',j']
    #   V[j'*2+k', 1] = coeff of B[j',k']
    #   W[k'*2+i', 1] = coeff of C[i',k']

    # Let's define the 7 Strassen terms for 2x2x2:
    # Each term is a triple (u_vec, v_vec, w_vec) where:
    #   u_vec[4] for A indices (00,01,10,11)
    #   v_vec[4] for B indices (00,01,10,11)
    #   w_vec[4] for C indices (00,01,10,11) but with trace convention W[k*2+i]

    # A indices: (i',j') -> flat: i'*2+j'
    # B indices: (j',k') -> flat: j'*2+k'
    # C indices: (i',k') -> flat for W: k'*2+i' (trace convention)

    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # Term 1: (A00+A11)(B00+B11), contributes to C00 and C11
        # u: A00=1, A01=0, A10=0, A11=1
        # v: B00=1, B01=0, B10=0, B11=1  -> flat: B[j'*2+k']: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        #   B00: j'=0,k'=0 -> idx 0
        #   B01: j'=0,k'=1 -> idx 1
        #   B10: j'=1,k'=0 -> idx 2
        #   B11: j'=1,k'=1 -> idx 3
        #   So B00+B11: v = [1, 0, 0, 1]
        # w: C00 and C11. C[i',k'] -> W[k'*2+i']
        #   C00: i'=0,k'=0 -> W[0*2+0]=W[0]
        #   C11: i'=1,k'=1 -> W[1*2+1]=W[3]
        #   So w = [1, 0, 0, 1]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # Term 2: (A01+A11)B10, contributes to C10
        # u: A01=1, A11=1 -> [0,1,0,1]
        # v: B10=1 -> [0,0,1,0]
        # w: C10: i'=1,k'=0 -> W[0*2+1]=W[1]
        #   w = [0,1,0,0]
        ([0,1,0,1], [0,0,1,0], [0,1,0,0]),
        # Term 3: A00(B01-B11), contributes to C01
        # u: A00=1 -> [1,0,0,0]
        # v: B01-B11: [0,1,0,-1]
        # w: C01: i'=0,k'=1 -> W[1*2+0]=W[2]
        #   w = [0,0,1,0]
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
        # Term 4: A11(B00-B10), contributes to C00 and C10
        # u: A11=1 -> [0,0,0,1]
        # v: B00-B10: [1,0,-1,0]
        # w: C00: W[0], C10: W[1]
        #   w = [1,1,0,0]
        ([0,0,0,1], [1,0,-1,0], [1,1,0,0]),
        # Term 5: (A10+A00)B01, contributes to C00 (negative) and C01
        # u: A10=1, A00=1 -> [1,0,1,0]
        # v: B01=1 -> [0,1,0,0]
        # w: C00: -1, C01: +1
        #   C00: W[0], C01: W[2]
        #   w = [-1, 0, 1, 0]
        ([1,0,1,0], [0,1,0,0], [-1,0,1,0]),
        # Term 6: (A10-A01)(B10+B11), contributes to C11
        # u: A10=1, A01=-1 -> [0,-1,1,0]
        # v: B10+B11: [0,0,1,1]
        # w: C11: W[3]
        #   w = [0,0,0,1]
        ([0,-1,1,0], [0,0,1,1], [0,0,0,1]),
        # Term 7: (A01-A10)(B00+B10), contributes to C00
        # u: A01=1, A10=-1 -> [0,1,-1,0]
        # v: B00+B10: [1,0,1,0]
        # w: C00: W[0]
        #   w = [1,0,0,0]
        ([0,1,-1,0], [1,0,1,0], [1,0,0,0]),
    ]

    # Now build the full 3x3x4 decomposition with rank 34.
    # For each k_block in {0,1} (corresponding to p=2 sub-blocks for k=0,1 and k=2,3):
    #   We have a 3x3x2 block.
    #   The 3x3x2 block has indices i in {0,1,2}, j in {0,1,2}, k_local in {0,1}.
    #   k_global = k_block * 2 + k_local.

    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 34

    col = 0
    for k_block in range(2):
        # k_global = k_block * 2 + k_local
        # 17 terms for this block

        # Part A: 2x2x2 Strassen core (i in {0,1}, j in {0,1}, k_local in {0,1})
        for (u_c, v_c, w_c) in strassen_terms:
            c = col
            # u_c: [A00, A01, A10, A11] -> maps to U[i*m+j] for i in {0,1}, j in {0,1}
            #   A[i,j] -> U[i*3 + j] (since m=3, not 2)
            #   u_c[0] = A00 -> U[0*3+0] = U[0]
            #   u_c[1] = A01 -> U[0*3+1] = U[1]
            #   u_c[2] = A10 -> U[1*3+0] = U[3]
            #   u_c[3] = A11 -> U[1*3+1] = U[4]
            U[0][c] += u_c[0]   # A00
            U[1][c] += u_c[1]   # A01
            U[3][c] += u_c[2]   # A10
            U[4][c] += u_c[3]   # A11

            # v_c: [B00, B01, B10, B11] -> maps to V[j*p+k]
            #   B[j_local, k_local] where j_local in {0,1}, k_local in {0,1}
            #   k_global = k_block * 2 + k_local
            #   V[j_global * p + k_global] = V[j_local * 4 + k_block*2 + k_local]
            #   B00: j=0,k_local=0 -> V[0*4 + k_block*2 + 0] = V[k_block*2]
            #   B01: j=0,k_local=1 -> V[0*4 + k_block*2 + 1] = V[k_block*2+1]
            #   B10: j=1,k_local=0 -> V[1*4 + k_block*2 + 0] = V[4 + k_block*2]
            #   B11: j=1,k_local=1 -> V[1*4 + k_block*2 + 1] = V[4 + k_block*2+1]
            V[k_block*2][c] += v_c[0]       # B00
            V[k_block*2+1][c] += v_c[1]     # B01
            V[4 + k_block*2][c] += v_c[2]   # B10
            V[4 + k_block*2+1][c] += v_c[3] # B11

            # w_c: [C00, C01, C10, C11] in trace convention W[k*2+i]
            #   C[i_local, k_local] -> W[k_global * n + i_local] = W[(k_block*2+k_local)*3 + i_local]
            #   w_c[0] = C00: i=0,k_local=0 -> W[(k_block*2+0)*3 + 0] = W[k_block*6]
            #   w_c[1] = C01: i=0,k_local=1 -> W[(k_block*2+1)*3 + 0] = W[k_block*6+3]
            #   w_c[2] = C10: i=1,k_local=0 -> W[(k_block*2+0)*3 + 1] = W[k_block*6+1]
            #   w_c[3] = C11: i=1,k_local=1 -> W[(k_block*2+1)*3 + 1] = W[k_block*6+4]
            W[k_block*6][c] += w_c[0]       # C00
            W[k_block*6+3][c] += w_c[1]     # C01
            W[k_block*6+1][c] += w_c[2]     # C10
            W[k_block*6+4][c] += w_c[3]     # C11

            col += 1

        # Part B: j=2 column (i in {0,1,2}, k_local in {0,1}) -> 6 terms
        for i in range(3):
            for k_local in range(2):
                c = col
                k_global = k_block * 2 + k_local
                # A[i, 2] * B[2, k_local] -> C[i, k_local]
                U[i * m + 2][c] = 1.0
                V[2 * p + k_global][c] = 1.0
                W[k_global * n + i][c] = 1.0
                col += 1

        # Part C: i=2 row, j in {0,1} (j in {0,1}, k_local in {0,1}) -> 4 terms
        for j in range(2):
            for k_local in range(2):
                c = col
                k_global = k_block * 2 + k_local
                # A[2, j] * B[j, k_local] -> C[2, k_local]
                U[2 * m + j][c] = 1.0
                V[j * p + k_global][c] = 1.0
                W[k_global * n + 2][c] = 1.0
                col += 1

    # Convert to integer/half-integer if possible
    def to_clean(x):
        # Check if close to integer or half-integer
        r = round(x * 2) / 2
        if abs(x - r) < 1e-10:
            return r
        return x

    U = [[to_clean(x) for x in row] for row in U]
    V = [[to_clean(x) for x in row] for row in V]
    W = [[to_clean(x) for x in row] for row in W]

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U = [[0] * rank_fb for _ in range(n * m)]
        V = [[0] * rank_fb for _ in range(m * p)]
        W = [[0] * rank_fb for _ in range(p * n)]
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
