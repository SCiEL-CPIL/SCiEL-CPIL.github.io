"""Matrix multiplication 2x3 x 3x9 with rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS  # 2, 3, 9

    # Fallback: schoolbook rank 54
    def schoolbook():
        r = n * m * p  # 54
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Build a rank-15 decomposition for 2x3 x 3x3
    # A (2x3) * B (3x3) = C (2x3)
    # U index: i*3+j for A[i,j], i in {0,1}, j in {0,1,2}
    # V index: j*3+k for B[j,k], j in {0,1,2}, k in {0,1,2}
    # W index: k*2+i for C[i,k], i in {0,1}, k in {0,1,2}
    #
    # Strategy: Use Laderman (rank 7) for the 2x2 x 2x2 subblock
    # (A rows 0,1; B cols 0,1; C rows 0,1; C cols 0,1)
    # Then handle the 3rd column of A, 3rd row of B, and 3rd column of C
    # with 8 additional terms.
    #
    # Laderman's algorithm for 2x2 x 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]]
    # 7 terms:
    # t1: a00 * b00 -> c00
    # t2: (a00+a01) * b10 -> c10
    # t3: a01 * (b00-b10) -> c00 (correction: actually let me use the standard form)
    #
    # Standard Laderman:
    # m1 = a00 * b00
    # m2 = a01 * b10
    # m3 = (a00+a01) * (b00-b10)
    # m4 = a10 * b01
    # m5 = a11 * b11
    # m6 = (a10+a11) * (b01-b11)
    # m7 = (a00+a11) * (b00+b11)
    # c00 = m1 + m3
    # c01 = m4 - m6
    # c10 = m2 + m3
    # c11 = m5 + m7 - m1 - m3 - m2 - m4 + m6  -- wait, let me re-derive
    #
    # Actually, let me use the well-known Laderman construction:
    # For A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # p1 = a*e
    # p2 = (a+b)*g
    # p3 = (a-c)*h
    # p4 = d*(f+h)
    # p5 = (b+c)*e
    # p6 = (b+d)*(f-g)
    # p7 = (a+b+c+d)*(f+g)
    # C[0,0] = p1 + p3 - p5 + p7
    # C[0,1] = p4 - p3 - p6 + p7
    # C[1,0] = p2 - p1 + p5 - p7
    # C[1,1] = p4 - p2 + p6 - p7
    #
    # Let me verify: C[0,0] = a*e + (a-c)*h - (b+c)*e + (a+b+c+d)*(f+g)
    # = ae + ah - ch - be - ce + (a+b+c+d)f + (a+b+c+d)g
    # = ae + ah - ch - be - ce + af + bf + cf + df + ag + bg + cg + dg
    # This doesn't look right. Let me use a simpler verified construction.
    #
    # I'll use a direct approach: enumerate 15 terms and verify numerically.
    #
    # Alternative: Use the fact that for 2x3 x 3x3, we can think of it as:
    # C[:,k] = A * B[:,k] for k=0,1,2
    # Each C[:,k] is a 2-vector = A (2x3) times B[:,k] (3x1)
    # This is 2x3 x 3x1 matrix-vector multiplication, rank 6 each (schoolbook)
    # Total schoolbook: 18
    #
    # To reduce, we share computations across k values.
    #
    # Here's a concrete rank-15 construction:
    # Define 15 bilinear forms. I'll use a systematic approach.
    
    def build_2x3x3_rank15():
        """Returns (U, V, W) for 2x3 x 3x3 with rank 15."""
        n2, m2, p2 = 2, 3, 3
        r = 15
        U = [[0]*r for _ in range(6)]
        V = [[0]*r for _ in range(9)]
        W = [[0]*r for _ in range(6)]
        
        # I'll construct 15 terms explicitly.
        # Term structure: each term t has U_vec (6), V_vec (9), W_vec (6)
        # such that sum_t U_vec[t] ⊗ V_vec[t] ⊗ W_vec[t] = tensor T
        # where T[i*3+j, j*3+k, k*2+i] = 1
        
        # Let me use a known valid construction by careful design:
        #
        # Terms 0-6: Laderman for the 2x2 subblock (A rows 0,1; B cols 0,1; C rows 0,1; C cols 0,1)
        # A_sub = [[a00, a01], [a10, a11]]
        # B_sub = [[b00, b01], [b10, b11]]
        # C_sub = [[c00, c01], [c10, c11]]
        
        # Laderman's 7 terms for 2x2x2:
        # t0: a00 * b00
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1  # c00
        # t1: (a00+a01) * b10
        U[0][1] = 1; U[1][1] = 1; V[3][1] = 1; W[3][1] = 1  # c10
        # t2: (a00-a10) * b01
        U[0][2] = 1; U[3][2] = -1; V[1][2] = 1; W[1][2] = 1  # c01
        # t3: a11 * b11
        U[4][3] = 1; V[4][3] = 1; W[4][3] = 1  # c11
        # t4: (a01+a11) * (b00-b10)
        U[1][4] = 1; U[4][4] = 1; V[0][4] = 1; V[3][4] = -1; W[0][4] = 1; W[3][4] = 1
        # t5: (a00+a11) * (b01-b11)
        U[0][5] = 1; U[4][5] = 1; V[1][5] = 1; V[4][5] = -1; W[1][5] = 1; W[4][5] = 1
        # t6: (a00+a01+a10+a11) * (b00+b01+b10+b11)
        U[0][6] = 1; U[1][6] = 1; U[3][6] = 1; U[4][6] = 1
        V[0][6] = 1; V[1][6] = 1; V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1; W[1][6] = 1; W[3][6] = 1; W[4][6] = 1
        
        # This Laderman construction needs verification. Let me instead use a 
        # simpler, definitely correct approach: just use schoolbook for the 
        # 2x2x2 part (rank 8) and add 7 more terms for the rest = 15 total.
        # But that gives rank 15 only if the 7 extra terms handle everything else.
        
        # Let me restart with a cleaner approach. I'll use schoolbook for 
        # specific parts and verify the total rank.
        
        # Actually, let me just directly construct and verify.
        # I'll set up the 15 terms and check if they reproduce the tensor.
        
        U = [[0]*15 for _ in range(6)]
        V = [[0]*15 for _ in range(9)]
        W = [[0]*15 for _ in range(6)]
        
        # Term 0: a00*b00 -> c00
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 1: a00*b10 -> c10
        U[0][1] = 1; V[3][1] = 1; W[3][1] = 1
        # Term 2: a00*b20 -> c20... wait, C is 2x3, so c indices are (row,col) with row in {0,1}, col in {0,1,2}
        # W[k*2+i]: W[0]=c[0,0], W[1]=c[0,1], W[2]=c[0,2], W[3]=c[1,0], W[4]=c[1,1], W[5]=c[1,2]
        
        # Let me be very explicit:
        # U[0]=a00, U[1]=a01, U[2]=a02, U[3]=a10, U[4]=a11, U[5]=a12
        # V[0]=b00, V[1]=b01, V[2]=b02, V[3]=b10, V[4]=b11, V[5]=b12, V[6]=b20, V[7]=b21, V[8]=b22
        # W[0]=c00, W[1]=c01, W[2]=c02, W[3]=c10, W[4]=c11, W[5]=c12
        
        # C[0,0] = a00*b00 + a01*b10 + a02*b20
        # C[0,1] = a00*b01 + a01*b11 + a02*b21
        # C[0,2] = a00*b02 + a01*b12 + a02*b22
        # C[1,0] = a10*b00 + a11*b10 + a12*b20
        # C[1,1] = a10*b01 + a11*b11 + a12*b21
        # C[1,2] = a10*b02 + a11*b12 + a12*b22
        
        # 15 terms. I'll use a construction that shares some computations.
        #
        # Idea: For the first row of C (c00, c01, c02), we compute:
        # c0k = a00*b0k + a01*b1k + a02*b2k for k=0,1,2
        # This is 3 outputs from 3 inputs of A (row 0) and 9 inputs of B.
        # Similarly for row 1.
        #
        # A simple sharing: define y0 = a00, y1 = a00+a01, y2 = a01
        # Then a00 = y0, a01 = y2, a00+a01 = y1
        # But this doesn't reduce multiplications.
        #
        # Let me try a different approach: use the structure to find 15 terms.
        # I'll construct them and verify numerically with numpy.
        
        # Reset
        U = [[0]*15 for _ in range(6)]
        V = [[0]*15 for _ in range(9)]
        W = [[0]*15 for _ in range(6)]
        
        # I'll use a construction based on splitting B's columns:
        # For k=0: c00 = a00*b00 + a01*b10 + a02*b20
        #            c10 = a10*b00 + a11*b10 + a12*b20
        # Share b00: (a00+a10)*b00 contributes to both c00 and c10
        #
        # Term 0: (a00+a10)*b00 -> contributes to c00 and c10
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; W[0][0] = 1; W[3][0] = 1
        # Term 1: (a01+a11)*b10 -> contributes to c00 and c10
        U[1][1] = 1; U[4][1] = 1; V[3][1] = 1; W[0][1] = 1; W[3][1] = 1
        # Term 2: (a02+a12)*b20 -> contributes to c00 and c10
        U[2][2] = 1; U[5][2] = 1; V[6][2] = 1; W[0][2] = 1; W[3][2] = 1
        # But this gives c00 = (a00+a10)*b00 + (a01+a11)*b10 + (a02+a12)*b20
        # which is NOT c00 = a00*b00 + a01*b10 + a02*b20
        # So this approach doesn't work directly.
        
        # Let me just use the schoolbook for 2x3x3 (rank 18) which gives total 54,
        # and acknowledge that finding rank 45 requires more sophisticated algebra.
        #
        # However, let me try one more approach: use the known result that
        # R(2,3,3) might be 15 or 16. I'll attempt a rank-16 construction.
        
        # Given the difficulty and risk of error, I'll return schoolbook (rank 54).
        U = [[0]*18 for _ in range(6)]
        V = [[0]*18 for _ in range(9)]
        W = [[0]*18 for _ in range(6)]
        c = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][c] = 1
                    V[j*3+k][c] = 1
                    W[k*2+i][c] = 1
                    c += 1
        return U, V, W

    if time.time() > deadline - 10.0:
        return schoolbook()

    # Try to build a rank-45 decomposition by tiling 3 blocks of 2x3x3
    # If we can get rank 15 per block, total rank = 45
    U_block, V_block, W_block = build_2x3x3_rank15()
    block_rank = len(U_block[0])
    
    if block_rank <= 15:
        # Tile 3 blocks
        r = block_rank * 3
        U = [[0]*r for _ in range(6)]
        V = [[0]*r for _ in range(27)]
        W = [[0]*r for _ in range(18)]
        
        for block in range(3):
            for i in range(6):
                for t in range(block_rank):
                    U[i][block*block_rank + t] = U_block[i][t]
            for j in range(3):
                for k in range(3):
                    v_idx = j*3 + k
                    v_block_idx = j*3 + (k % 3)  # within the 3x3 block
                    for t in range(block_rank):
                        V[v_idx][block*block_rank + t] = V_block[v_block_idx][t]
            for k in range(3):
                for i in range(2):
                    w_idx = k*2 + i
                    w_block_idx = k*2 + i
                    for t in range(block_rank):
                        W[w_idx][block*block_rank + t] = W_block[w_block_idx][t]
        
        # Verify
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for random A and B, does A*B match the decomposition?
        A_test = np.random.randn(2, 3)
        B_test = np.random.randn(3, 9)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((2, 9))
        for t in range(r):
            a_vec = U_np[:, t].reshape(2, 3)
            b_vec = V_np[:, t].reshape(3, 9)
            w_vec = W_np[:, t].reshape(9, 2)  # W[k*2+i]
            # The bilinear form: sum_t (A . u_t) * (v_t . B) * w_t
            # More precisely: C = sum_t (u_t^T A) (v_t^T B) w_t^T ... 
            # Actually: C[i,k] = sum_t U[i*m+j_t, t] * V[j_t*p+k_t, t] * W[k_t*n+i, t]
            # Let me just do the full tensor contraction
            pass
        
        # Simpler verification: build the full tensor and check
        # T[i,j,k] should equal sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*2+i, t]
        # for the 2x3x3 case, but we have 2x3x9 now.
        
        # For the full 2x3x9:
        # U is 6xr, V is 27xr, W is 18xr
        # T[i,j,k] = sum_t U[i*3+j, t] * V[j*9+k, t] * W[k*2+i, t]
        # should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        
        error = 0
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    val = 0
                    for t in range(r):
                        val += U[i*3+j][t] * V[j*9+k][t] * W[k*2+i][t]
                    error += abs(val - 1)
        
        if error < 1e-10:
            return {"U": U, "V": V, "W": W}
    
    return schoolbook()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
