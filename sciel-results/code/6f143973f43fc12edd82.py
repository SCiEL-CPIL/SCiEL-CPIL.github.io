"""Matrix multiplication 4,4,8 — rank-minimal decomposition via Strassen-Kronecker."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    Strategy: Use Kronecker product structure.
    - M_{4,4,4} has a known rank-49 decomposition (Strassen 2x2x2 applied twice).
    - M_{1,1,2} has rank 2 (trivial: just two scalar multiplications).
    - M_{4,4,8} = M_{4,4,4} ⊗ M_{1,1,2} has rank 49 * 2 = 98.
    
    The 4x8 output is viewed as two 4x4 blocks, each computed by the 4x4x4 decomposition.
    """
    deadline = time.time() + 100  # safety margin before 120s limit
    
    # Step 1: Build Strassen rank-7 decomposition for 2x2x2
    # U: (2*2, 7) = (4, 7), V: (2*2, 7) = (4, 7), W: (2*2, 7) = (4, 7)
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4+m5+m7, m3+m5], [m2+m4, m1+m2+m3+m6]]
    
    # U[i*m+j, t] for i,j in {0,1}, t in {0..6}
    # V[j*p+k, t] for j,k in {0,1}, t in {0..6}
    # W[k*n+i, t] for k,i in {0,1}, t in {0..6}
    
    # U rows: (i,j) -> index i*2+j
    # V rows: (j,k) -> index j*2+k
    # W rows: (k,i) -> index k*2+i
    
    # m1 = (a+d)(e+h): A contributes a (i=0,j=0) and d (i=1,j=1); B contributes e (j=0,k=0) and h (j=1,k=1)
    # m2 = (c+d)e: A contributes c (i=1,j=0) and d (i=1,j=1); B contributes e (j=0,k=0)
    # m3 = a(f-h): A contributes a (i=0,j=0); B contributes f (j=0,k=1) and -h (j=1,k=1)
    # m4 = d(g-e): A contributes d (i=1,j=1); B contributes g (j=1,k=0) and -e (j=0,k=0)
    # m5 = (a+b)h: A contributes a (i=0,j=0) and b (i=0,j=1); B contributes h (j=1,k=1)
    # m6 = (c-a)(e+f): A contributes c (i=1,j=0) and -a (i=0,j=0); B contributes e (j=0,k=0) and f (j=0,k=1)
    # m7 = (b-d)(g+h): A contributes b (i=0,j=1) and -d (i=1,j=1); B contributes g (j=1,k=0) and h (j=1,k=1)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # t=0: m1
    U2[0][0] = 1  # a (i=0,j=0)
    U2[3][0] = 1  # d (i=1,j=1)
    V2[0][0] = 1  # e (j=0,k=0)
    V2[3][0] = 1  # h (j=1,k=1)
    W2[0][0] = 1  # C[0,0] gets m1
    W2[3][0] = 1  # C[1,1] gets m1
    
    # t=1: m2
    U2[2][1] = 1  # c (i=1,j=0)
    U2[3][1] = 1  # d (i=1,j=1)
    V2[0][1] = 1  # e (j=0,k=0)
    W2[3][1] = 1  # C[1,1] gets m2
    
    # t=2: m3
    U2[0][2] = 1  # a (i=0,j=0)
    V2[1][2] = 1  # f (j=0,k=1)
    V2[3][2] = -1 # -h (j=1,k=1)
    W2[0][2] = 1  # C[0,1] gets m3
    W2[3][2] = 1  # C[1,1] gets m3
    
    # t=3: m4
    U2[3][3] = 1  # d (i=1,j=1)
    V2[2][3] = 1  # g (j=1,k=0)
    V2[0][3] = -1 # -e (j=0,k=0)
    W2[0][3] = 1  # C[0,0] gets m4
    W2[3][3] = 1  # C[1,1] gets m4
    
    # t=4: m5
    U2[0][4] = 1  # a (i=0,j=0)
    U2[1][4] = 1  # b (i=0,j=1)
    V2[3][4] = 1  # h (j=1,k=1)
    W2[0][4] = 1  # C[0,0] gets m5
    W2[1][4] = 1  # C[0,1] gets m5
    
    # t=5: m6
    U2[2][5] = 1  # c (i=1,j=0)
    U2[0][5] = -1 # -a (i=0,j=0)
    V2[0][5] = 1  # e (j=0,k=0)
    V2[1][5] = 1  # f (j=0,k=1)
    W2[1][5] = 1  # C[0,1] gets m6
    W2[2][5] = 1  # C[1,0] gets m6
    
    # t=6: m7
    U2[1][6] = 1  # b (i=0,j=1)
    U2[3][6] = -1 # -d (i=1,j=1)
    V2[2][6] = 1  # g (j=1,k=0)
    V2[3][6] = 1  # h (j=1,k=1)
    W2[0][6] = 1  # C[0,0] gets m7
    W2[2][6] = 1  # C[1,0] gets m7
    
    # Verify 2x2x2 decomposition
    # For each (i,j,k) in {0,1}^3, sum over t of U[i*2+j,t]*V[j*2+k,t]*W[k*2+i,t] should be 1 iff k corresponds to output position
    # Actually, the tensor T_{i,j,k} = delta where C[k,i] = sum_j A[i,j]*B[j,k]
    # So T_{i,j,k} = 1 for all valid (i,j,k) combinations, and the decomposition should reconstruct this.
    
    # Step 2: Build 4x4x4 decomposition by Kronecker product of two 2x2x2 decompositions
    # U4: (4*4, 49) = (16, 49), V4: (4*4, 49) = (16, 49), W4: (4*4, 49) = (16, 49)
    # Kronecker: U4[(i1*2+i2, j1*2+j2), t1*7+t2] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i4 = i1*2 + i2
                    j4 = j1*2 + j2
                    for t1 in range(7):
                        for t2 in range(7):
                            t4 = t1*7 + t2
                            val = U2[i1*2+j1][t1] * U2[i2*2+j2][t2]
                            if val != 0:
                                U4[i4*4+j4][t4] = val
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j4 = j1*2 + j2
                    k4 = k1*2 + k2
                    for t1 in range(7):
                        for t2 in range(7):
                            t4 = t1*7 + t2
                            val = V2[j1*2+k1][t1] * V2[j2*2+k2][t2]
                            if val != 0:
                                V4[j4*4+k4][t4] = val
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k4 = k1*2 + k2
                    i4 = i1*2 + i2
                    for t1 in range(7):
                        for t2 in range(7):
                            t4 = t1*7 + t2
                            val = W2[k1*2+i1][t1] * W2[k2*2+i2][t2]
                            if val != 0:
                                W4[k4*4+i4][t4] = val
    
    if time.time() > deadline - 20:
        # Fallback: return schoolbook
        n, m, p = DIMENSIONS
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Step 3: Build 4x4x8 decomposition
    # M_{4,4,8} = M_{4,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2} has rank 2: U=(1,2)=[1,1], V=(1,2)=[1,1], W=(2,1)=[[1],[1]]
    # Wait, for 1x1x2: A is 1x1, B is 1x2, C is 2x1
    # C[k,0] = A[0,0]*B[0,k] for k=0,1
    # Tensor T_{i,j,k} with i=0, j=0, k in {0,1}
    # U[i*m+j, t] = U[0, t], V[j*p+k, t] = V[0, t] for k in {0,1}, W[k*n+i, t] = W[k, 0, t]
    
    # Simple rank-2 decomposition:
    # t=0: U=1, V[0,0]=1, V[0,1]=0, W[0,0]=1, W[1,0]=0
    # t=1: U=1, V[0,0]=0, V[0,1]=1, W[0,0]=0, W[1,0]=1
    
    U1 = [[1, 1]]  # shape (1, 2)
    V1 = [[1, 0],  # shape (1*2, 2) = (2, 2)
          [0, 1]]
    W1 = [[1, 0],  # shape (2*1, 2) = (2, 2)
          [0, 1]]
    
    # Kronecker product for 4x4x8:
    # U8: (4*4, 98) = (16, 98)
    # V8: (4*8, 98) = (32, 98)
    # W8: (8*4, 98) = (32, 98)
    
    # U8[i4*4+j4, t4*2+t1] = U4[i4*4+j4, t4] * U1[0, t1] = U4[i4*4+j4, t4] * 1
    # Actually, the Kronecker structure:
    # The 4x8 matrix B has columns indexed by (k1, k2) where k1 in {0,1,2,3} (for 4x4 part) and k2 in {0,1}
    # Similarly for output C indexed by (i1, i2) with i1 in {0,1,2,3} and i2 in {0,1}
    
    # Let me reconsider. The 4x4x4 multiplication handles A (4x4) * B_4 (4x4) -> C_4 (4x4)
    # We want A (4x4) * B (4x8) -> C (8x4) where B = [B_4_left | B_4_right]
    
    # Using Kronecker: view 4x4 as 2x2 blocks of 2x2, and 4x8 as...
    # Actually, let's use a different factorization:
    # 4x4x8 = (2x2x4) ⊗ (2x2x2)
    # Or: 4x4x8 = (4x4x4) ⊗ (1x1x2)
    
    # For (4x4x4) ⊗ (1x1x2):
    # A is 4x4, B is 4x8, C is 8x4
    # View B as B_4 ⊗ B_2 where B_4 is 4x4 and B_2 is 1x2? No, that doesn't work dimensionally.
    
    # Let me think differently. The multiplication tensor for (n,m,p) has indices (i,j,k) with:
    # i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # T[i,j,k] = 1 (it's the structure tensor)
    
    # For 4x4x8: i in 0..3, j in 0..3, k in 0..7
    # We can write this as a direct sum:
    # T_{4,4,8}[i,j,k] = T_{4,4,4}[i,j,k//2] * delta_{k%2, 0} + ... no, that's not right.
    
    # Actually, 4x4x8 multiplication: C[k,i] = sum_j A[i,j] * B[j,k]
    # This is equivalent to two independent 4x4x4 multiplications if we split B into two 4x4 matrices:
    # B = [B_a | B_b] where B_a and B_b are both 4x4
    # C = [C_a; C_b] where C_a = A*B_a and C_b = A*B_b, both 4x4
    
    # So T_{4,4,8} = T_{4,4,4} ⊕ T_{4,4,4} (direct sum)
    # Rank = 49 + 49 = 98
    
    # This is simpler! Just use two copies of the 4x4x4 decomposition.
    
    rank8 = 98
    U8 = [[0]*rank8 for _ in range(16)]
    V8 = [[0]*rank8 for _ in range(32)]
    W8 = [[0]*rank8 for _ in range(32)]
    
    # Copy 1: for k in 0..3 (first 4 columns of B, first 4 rows of C)
    for i in range(4):
        for j in range(4):
            for t in range(49):
                U8[i*4+j][t] = U4[i*4+j][t]
    
    for j in range(4):
        for k in range(4):
            for t in range(49):
                V8[j*8+k][t] = V4[j*4+k][t]
    
    for k in range(4):
        for i in range(4):
            for t in range(49):
                W8[k*4+i][t] = W4[k*4+i][t]
    
    # Copy 2: for k in 4..7 (last 4 columns of B, last 4 rows of C)
    for i in range(4):
        for j in range(4):
            for t in range(49):
                U8[i*4+j][t+49] = U4[i*4+j][t]
    
    for j in range(4):
        for k in range(4):
            for t in range(49):
                V8[j*8+(k+4)][t+49] = V4[j*4+k][t]
    
    for k in range(4):
        for i in range(4):
            for t in range(49):
                W8[(k+4)*4+i][t+49] = W4[k*4+i][t]
    
    # Verify the decomposition (quick spot check)
    # For a random test: A[i,j]*B[j,k] should equal sum_t U8[i*4+j,t]*V8[j*8+k,t]*W8[k*4+i,t]
    # Let's verify for a few entries
    
    # Check: sum_t U8[i*4+j,t] * V8[j*8+k,t] * W8[k*4+i,t] should equal 1 for all valid (i,j,k)
    # since T[i,j,k] = 1 for the multiplication tensor
    
    import random
    random.seed(42)
    tests = [(0,0,0), (0,0,3), (0,0,4), (0,0,7), (3,3,0), (3,3,7), (1,2,5)]
    for i, j, k in tests:
        s = 0
        for t in range(rank8):
            s += U8[i*4+j][t] * V8[j*8+k][t] * W8[k*4+i][t]
        if s != 1:
            # Fallback to schoolbook
            n, m, p = DIMENSIONS
            rank = n*m*p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U[ii*m+jj][col] = 1
                        V[jj*p+kk][col] = 1
                        W[kk*n+ii][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U8, "V": V8, "W": W8}
# EVOLVE-BLOCK-END
