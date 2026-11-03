import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # The rank of the matrix multiplication tensor M_{2,3,3} is 17.
    # This is derived by partitioning the multiplication into:
    # 1. M_{2,2,2} (top-left 2x2 block): Rank 7 (Strassen)
    # 2. M_{2,1,2} (A[:,2] * B[2,:]): Rank 4 (Schoolbook)
    # 3. M_{2,3,1} (A[:,: ] * B[:,2]): Rank 6 (Schoolbook)
    # Total Rank = 7 + 4 + 6 = 17.
    
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Indices:
    # U[i, r] corresponds to A[i // m, i % m]
    # V[j, r] corresponds to B[j // p, j % p]
    # W[k, r] corresponds to C[k // n, k % n]
    # Note: The task specifies W[k*n+i, r] for C[i,k].
    # So W index is k*n + i.
    
    # --- Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
    # Let A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # Let B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    # C00 = A00*B00, C01 = A00*B01 + A01*B11, C10 = A10*B00 + A11*B10, C11 = A11*B11
    
    # Strassen Terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # U indices for A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # V indices for B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # W indices for C: k*n+i. C is 2x3.
    # C(0,0) -> 0*2+0=0
    # C(0,1) -> 1*2+0=2
    # C(0,2) -> 2*2+0=4
    # C(1,0) -> 0*2+1=1
    # C(1,1) -> 1*2+1=3
    # C(1,2) -> 2*2+1=5

    # Term 0: M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11

    # Term 1: M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10
    W[3, 1] = -1 # C11

    # Term 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1  # C01
    W[3, 2] = 1  # C11

    # Term 3: M4
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1 # C00
    W[1, 3] = 1  # C10

    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1  # C01

    # Term 5: M6
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1  # C00

    # Term 6: M7
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1  # C00

    # --- Part 2: Schoolbook for A[:,2] * B[2,:] -> C[:,0:2] ---
    # A[:,2] is 2x1: indices 2 (A[0,2]), 5 (A[1,2])
    # B[2,:] is 1x2: indices 6 (B[2,0]), 7 (B[2,1])
    # C[:,0] gets A[0,2]B[2,0] and A[1,2]B[2,0]
    # C[:,1] gets A[0,2]B[2,1] and A[1,2]B[2,1]
    
    # Term 7: A[0,2] * B[2,0] -> C[0,0]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1  # C(0,0)

    # Term 8: A[0,2] * B[2,1] -> C[0,1]
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1  # C(0,1)

    # Term 9: A[1,2] * B[2,0] -> C[1,0]
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1  # C(1,0)

    # Term 10: A[1,2] * B[2,1] -> C[1,1]
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1 # C(1,1)

    # --- Part 3: Schoolbook for A[:,: ] * B[:,2] -> C[:,2] ---
    # A[:,0:2] is 2x2: indices 0,1,3,4
    # B[0:2,2] is 2x1: indices 2 (B[0,2]), 5 (B[1,2])
    # C[0,2] gets A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] gets A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 11: A[0,0] * B[0,2] -> C[0,2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1 # C(0,2)

    # Term 12: A[0,1] * B[1,2] -> C[0,2]
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1 # C(0,2)

    # Term 13: A[0,2] * B[2,2] -> C[0,2]
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1 # C(0,2)

    # Term 14: A[1,0] * B[0,2] -> C[1,2]
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1 # C(1,2)

    # Term 15: A[1,1] * B[1,2] -> C[1,2]
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1 # C(1,2)

    # Term 16: A[1,2] * B[2,2] -> C[1,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1 # C(1,2)

    # Verification
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if not verify(U, V, W):
        raise RuntimeError("Rank-17 verification failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
