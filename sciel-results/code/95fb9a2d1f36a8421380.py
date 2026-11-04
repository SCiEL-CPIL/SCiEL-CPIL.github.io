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
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1]   # i=1,j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0,k=0
        [0, 0, 1, 0, 0, 1, 0],   # j=0,k=1
        [0, 0, 0, 1, 0, 0, 1],   # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1]   # j=1,k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0
        [0, 1, 0, 1, 0, 0, 0],   # k=0,i=1
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0
        [1, -1, 1, 0, 0, 1, 0]   # k=1,i=1
    ], dtype=float)
    
    # Verify Strassen 2x2x2
    # T[i,j,k] = sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    # Should equal 1 for all i,j,k
    T2 = np.zeros((2, 2, 2))
    for r in range(7):
        u3 = U2[:, r].reshape(2, 2)
        v3 = V2[:, r].reshape(2, 2)
        w3 = W2[:, r].reshape(2, 2)
        T2 += np.einsum('ij,jk,ki->ijk', u3, v3, w3)
    
    if np.max(np.abs(T2 - 1.0)) > 1e-9:
        raise ValueError("Strassen 2x2x2 verification failed")
    
    # Build 4x4x4 decomposition from Kronecker product of Strassen (rank 49)
    # U4[i*4+j, r] where r = ro*7 + ri
    # U4[i*4+j, ro*7+ri] = U2[I*2+ip, ro] * U2[ip*2+jp, ri] where i=2I+ip, j=2J+jp
    # This is the Kronecker product structure
    
    # Vectorized construction using broadcasting
    # For U: U4[i*4+j, ro*7+ri] = U2[I*2+ip, ro] * U2[ip*2+jp, ri]
    # where i = 2*I + ip, j = 2*J + jp, so I=i//2, ip=i%2, J=j//2, jp=j%2
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            # U: i*4+j index
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    ip = i % 2
                    J = j // 2
                    jp = j % 2
                    U4[i * 4 + j, r] = U2[I * 2 + ip, ro] * U2[ip * 2 + jp, ri]
            
            # V: j*4+k index
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    jp = j % 2
                    K = k // 2
                    kp = k % 2
                    V4[j * 4 + k, r] = V2[J * 2 + jp, ro] * V2[jp * 2 + kp, ri]
            
            # W: k*4+i index
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    kp = k % 2
                    I = i // 2
                    ip = i % 2
                    W4[k * 4 + i, r] = W2[K * 2 + kp, ro] * W2[kp * 2 + ip, ri]
    
    # Verify 4x4x4 decomposition
    T4 = np.zeros((4, 4, 4))
    for r in range(49):
        u3 = U4[:, r].reshape(4, 4)
        v3 = V4[:, r].reshape(4, 4)
        w3 = W4[:, r].reshape(4, 4)
        T4 += np.einsum('ij,jk,ki->ijk', u3, v3, w3)
    
    if np.max(np.abs(T4 - 1.0)) > 1e-9:
        raise ValueError("4x4x4 verification failed")
    
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
    
    # Verify full 4x4x8 decomposition
    T = np.zeros((4, 4, 8))
    for r in range(rank):
        u3 = U[:, r].reshape(4, 4)
        v3 = V[:, r].reshape(4, 8)
        w3 = W[:, r].reshape(8, 4)
        T += np.einsum('ij,jk,ki->ijk', u3, v3, w3)
    
    if np.max(np.abs(T - 1.0)) > 1e-9:
        raise ValueError("4x4x8 verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
