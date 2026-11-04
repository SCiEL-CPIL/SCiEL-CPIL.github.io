import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    
    Uses Strassen's algorithm for 4x4x4 blocks (rank 49) applied to each of two
    4x4x4 sub-problems (left and right halves of the 4x4x8 output).
    Rank 98 is the minimal known rank for this tensor.
    
    Indexing convention:
    - U: (n*m, rank) = (16, 98), U[i*m+j, r] is coefficient of A[i,j] in product r
    - V: (m*p, rank) = (32, 98), V[j*p+k, r] is coefficient of B[j,k] in product r
    - W: (p*n, rank) = (32, 98), W[k*n+i, r] is coefficient of product r in C[i,k]
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 += M1, C11 += M1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1  # C01 += M3, C11 += M3
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1  # C00 += M4, C10 += M4
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1  # C00 -= M5, C01 += M5
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1  # C11 += M6
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1  # C00 += M7

    # Build 4x4x4 decomposition from two-level Strassen (rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Vectorized construction using Kronecker products
    # For U: U4[i*4+j, ro*7+ri] = U2[I*2+ip, ro] * U2[J*2+jp, ri] where i=2I+ip, j=2J+jp
    # This is a Kronecker-like product but with the specific index mapping
    
    # Build U4 using the tensor product structure
    # U2 has shape (4,7) where index = I*2+ip for I in {0,1}, ip in {0,1}
    # U4 index = i*4+j = (2I+ip)*4 + (2J+jp) = 8I + 4ip + 2J + jp
    # r = ro*7 + ri
    
    # We can build U4 as follows:
    # For each (ro, ri), U4[:, ro*7+ri] is determined by U2[:,ro] and U2[:,ri]
    # The mapping from (I,ip,J,jp) to i*4+j is: i=2I+ip, j=2J+jp
    # So index = (2I+ip)*4 + (2J+jp) = 8I + 4ip + 2J + jp
    
    # Let's create the index mapping arrays
    I_vals = np.array([0,0,1,1])  # for uo indices 0,1,2,3
    ip_vals = np.array([0,1,0,1])
    J_vals = np.array([0,1,0,1])
    jp_vals = np.array([0,1,0,1])
    
    # For U: uo -> (I,J), ui -> (ip,jp), i=2I+ip, j=2J+jp, idx=i*4+j
    # U4[idx, ro*7+ri] = U2[uo,ro] * U2[ui,ri]
    # idx = (2*I[uo] + ip[ui])*4 + (2*J[uo] + jp[ui])
    
    # Precompute the index for each (uo, ui) pair
    U_idx = np.zeros((4,4), dtype=int)
    for uo in range(4):
        for ui in range(4):
            I = uo // 2
            J = uo % 2
            ip = ui // 2
            jp = ui % 2
            i = 2 * I + ip
            j = 2 * J + jp
            U_idx[uo, ui] = i * 4 + j
    
    # For V: vo -> (J,K), vi -> (jp,kp), j=2J+jp, k=2K+kp, idx=j*4+k
    V_idx = np.zeros((4,4), dtype=int)
    for vo in range(4):
        for vi in range(4):
            J = vo // 2
            K = vo % 2
            jp = vi // 2
            kp = vi % 2
            j = 2 * J + jp
            k = 2 * K + kp
            V_idx[vo, vi] = j * 4 + k
    
    # For W: wo -> (K,I), wi -> (kp,ip), k=2K+kp, i=2I+ip, idx=k*4+i
    W_idx = np.zeros((4,4), dtype=int)
    for wo in range(4):
        for wi in range(4):
            K = wo // 2
            I = wo % 2
            kp = wi // 2
            ip = wi % 2
            k = 2 * K + kp
            i = 2 * I + ip
            W_idx[wo, wi] = k * 4 + i
    
    # Now build U4, V4, W4 efficiently
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            
            # U4: for each (uo,ui), add U2[uo,ro]*U2[ui,ri] to U4[U_idx[uo,ui], r]
            for uo in range(4):
                if U2[uo, ro] != 0:
                    for ui in range(4):
                        if U2[ui, ri] != 0:
                            U4[U_idx[uo, ui], r] += U2[uo, ro] * U2[ui, ri]
            
            # V4
            for vo in range(4):
                if V2[vo, ro] != 0:
                    for vi in range(4):
                        if V2[vi, ri] != 0:
                            V4[V_idx[vo, vi], r] += V2[vo, ro] * V2[vi, ri]
            
            # W4
            for wo in range(4):
                if W2[wo, ro] != 0:
                    for wi in range(4):
                        if W2[wi, ri] != 0:
                            W4[W_idx[wo, wi], r] += W2[wo, ro] * W2[wi, ri]

    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    if time.time() > deadline:
        # Return best result found so far (which is the full result if we reached here)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
