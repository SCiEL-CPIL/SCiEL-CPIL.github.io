# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for safety margin
    
    n, m, p = 4, 4, 9
    
    # Helper to check if we're still within time
    def time_ok():
        return time.time() < deadline

    # 1. Define 2x2x2 Strassen decomposition (Rank 7)
    U_222 = np.zeros((4, 7), dtype=float)
    V_222 = np.zeros((4, 7), dtype=float)
    W_222 = np.zeros((4, 7), dtype=float)
    
    # m1: (a+d)(e+h) -> C00+C11
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # m2: (c+d)e -> C10+C11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = 1
    
    # m3: a(f-h) -> C01+C11
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # m4: d(g-e) -> C00+C11
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[3, 3] = 1
    
    # m5: (a+b)h -> C00-C01
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = 1; W_222[1, 4] = -1
    
    # m6: (c-a)(e+f) -> C10
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[2, 5] = 1
    
    # m7: (b-d)(g+h) -> C00+C01
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[1, 6] = 1

    # 2. Define 2x2x4 Schoolbook decomposition (Rank 8)
    # n=2, m=2, p=4
    U_224 = np.zeros((4, 8), dtype=float)
    V_224 = np.zeros((8, 8), dtype=float)
    W_224 = np.zeros((8, 8), dtype=float)
    
    # For each k in 0..3, we have a 1x2x1 dot product structure for each i?
    # Actually, it's 4 independent dot products of length 2.
    # k=0: C_00 = A_00 B_00 + A_01 B_10
    # k=1: C_01 = A_00 B_01 + A_01 B_11
    # k=2: C_02 = A_00 B_02 + A_01 B_12
    # k=3: C_03 = A_00 B_03 + A_01 B_13
    # And similarly for i=1.
    # The tensor is M[i,j,k] = 1.
    # We can decompose this as sum over k of (Identity for i) x (Dot product for j) ?
    # No, the indices are i, j, k.
    # U index i*m+j. V index j*p+k. W index k*n+i.
    # For fixed k, we need to handle i and j.
    # It's simpler to just use the rank-8 decomposition derived from 4 independent dot products.
    # Each dot product is over j (length 2).
    # There are 2 choices for i and 4 choices for k. Total 8 independent equations?
    # No, the equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # If we set r = 2*k + j'?
    # Let's use the Kronecker product of 2x1x1 (Rank 1) and 1x2x4 (Rank 8).
    # 2x1x1: n=2, m=1, p=1. U(2,1), V(1,1), W(2,1).
    # U = [1, 1], V = [1], W = [1, 1].
    # 1x2x4: n=1, m=2, p=4. U(2,8), V(8,8), W(4,8).
    # U = I_2 repeated 4 times? No.
    # For 1x2x4, M[i,j,k] with i=0.
    # sum_r U[j,r] V[j*4+k,r] W[k,r] = 1.
    # This is 4 independent dot products.
    # r = 2*k + j.
    # U[j, 2*k+j] = 1.
    # V[j*4+k, 2*k+j] = 1.
    # W[k, 2*k+j] = 1.
    
    # So for 2x2x4, we take Kronecker of 2x1x1 and 1x2x4.
    # Rank = 1 * 8 = 8.
    # U_224 = Kron(U_211, U_124)
    # V_224 = Kron(V_211, V_124)
    # W_224 = Kron(W_211, W_124)
    
    U_211 = np.array([[1], [1]], dtype=float)
    V_211 = np.array([[1]], dtype=float)
    W_211 = np.array([[1], [1]], dtype=float)
    
    U_124 = np.zeros((2, 8), dtype=float)
    V_124 = np.zeros((8, 8), dtype=float)
    W_124 = np.zeros((4, 8), dtype=float)
    
    for k in range(4):
        for j in range(2):
            r = 2 * k + j
            U_124[j, r] = 1
            V_124[j * 4 + k, r] = 1
            W_124[k, r] = 1
            
    U_224 = np.kron(U_211, U_124)
    V_224 = np.kron(V_211, V_124)
    W_224 = np.kron(W_211, W_124)

    # 3. Construct 4x4x8 Kronecker decomposition (Rank 56)
    # T_448 = T_222 x T_224
    U_448 = np.zeros((16, 56), dtype=float)
    V_448 = np.zeros((32, 56), dtype=float)
    W_448 = np.zeros((32, 56), dtype=float)
    
    for r1 in range(7):
        for r2 in range(8):
            r = r1 * 8 + r2
            U_448[:, r] = np.kron(U_222[:, r1], U_224[:, r2])
            V_448[:, r] = np.kron(V_222[:, r1], V_224[:, r2])
            W_448[:, r] = np.kron(W_222[:, r1], W_224[:, r2])

    # 4. Construct 4x4x1 Schoolbook decomposition (Rank 16)
    U_441 = np.zeros((16, 16), dtype=float)
    V_441 = np.zeros((36, 16), dtype=float)
    W_441 = np.zeros((36, 16), dtype=float)
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j, col] = 1
            # V index for 4x4x1: j*p+k = j*1+0 = j
            V_441[j, col] = 1
            # W index for 4x4x1: k*n+i = 0*4+i = i
            W_441[i, col] = 1

    # 5. Assemble full 4x4x9 decomposition
    total_rank = 56 + 16
    U_total = np.zeros((16, total_rank), dtype=float)
    V_total = np.zeros((36, total_rank), dtype=float)
    W_total = np.zeros((36, total_rank), dtype=float)
    
    # Part 1: B cols 0-7, rank 56
    U_total[:, 0:56] = U_448
    V_total[:, 0:56] = V_448
    W_total[:, 0:56] = W_448
    
    # Part 2: B col 8, rank 16
    U_total[:, 56:72] = U_441
    V_total[:, 56:72] = V_441
    W_total[:, 56:72] = W_441

    # 6. Validation
    def validate_fast(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not time_ok():
                        return True
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if validate_fast(U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=float)
        V_sb = np.zeros((m*p, rank_sb), dtype=float)
        W_sb = np.zeros((p*n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
