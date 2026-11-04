"""Matrix multiplication 4x5x6: explicit low-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook(n, m, p):
        """Returns U, V, W for n x m x p matrix multiplication."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_222():
        """Returns U, V, W for 2 x 2 x 2 matrix multiplication (Strassen)."""
        n, m, p = 2, 2, 2
        rank = 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen factors
        # Indices: i in 0..1, j in 0..1, k in 0..1
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11) B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00 (B01 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11 (B10 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5 = (A00 + A01) B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def build_decomposition():
        """
        Partition 4x5x6 into blocks:
        n=4 = 2+2
        m=5 = 2+3
        p=6 = 3+3
        
        Blocks:
        1. n0(2), m0(2), p0(3) -> 2x2x3 (Schoolbook)
        2. n0(2), m0(2), p1(3) -> 2x2x3 (Schoolbook)
        3. n0(2), m1(3), p0(3) -> 2x3x3 (Schoolbook)
        4. n0(2), m1(3), p1(3) -> 2x3x3 (Schoolbook)
        5. n1(2), m0(2), p0(3) -> 2x2x3 (Schoolbook)
        6. n1(2), m0(2), p1(3) -> 2x2x3 (Schoolbook)
        7. n1(2), m1(3), p0(3) -> 2x3x3 (Schoolbook)
        8. n1(2), m1(3), p1(3) -> 2x3x3 (Schoolbook)
        
        Wait, the inner dimension m is split into 2+3.
        The tensor is T[i, j, k].
        i in 0..3, j in 0..4, k in 0..5.
        
        Split i: 0..1 (I0), 2..3 (I1)
        Split j: 0..1 (J0), 2..4 (J1)
        Split k: 0..2 (K0), 3..5 (K1)
        
        We need to handle each combination of (I, J, K).
        However, Strassen works on 2x2x2.
        We can further split the 3s into 2+1.
        
        Let's use a finer partition to utilize Strassen on 2x2x2 cores.
        n=4 = 2+2
        m=5 = 2+2+1
        p=6 = 2+2+2
        
        This creates many small blocks.
        Block (2,2,2) -> Strassen (Rank 7)
        Block (2,2,1) -> Schoolbook (Rank 4)
        Block (2,1,2) -> Schoolbook (Rank 4)
        Block (2,1,1) -> Schoolbook (Rank 2)
        Block (2,2,2) -> Strassen (Rank 7)
        ...
        
        Total Rank Calculation:
        i splits: 2, 2
        j splits: 2, 2, 1
        k splits: 2, 2, 2
        
        Combinations:
        i=2, j=2, k=2: 7
        i=2, j=2, k=2: 7
        i=2, j=2, k=2: 7
        i=2, j=2, k=2: 7  (4 blocks of 2x2x2)
        i=2, j=1, k=2: 4
        i=2, j=1, k=2: 4
        i=2, j=1, k=2: 4  (3 blocks of 2x1x2)
        i=2, j=2, k=2: 7 (Wait, k is always 2? No, k splits are 2,2,2)
        
        Let's list them:
        i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 -> 7
        i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 -> 7
        i in {0,1}, j in {0,1}, k in {4,5}: 2x2x2 -> 7
        i in {0,1}, j in {2,3}, k in {0,1}: 2x2x2 -> 7
        i in {0,1}, j in {2,3}, k in {2,3}: 2x2x2 -> 7
        i in {0,1}, j in {2,3}, k in {4,5}: 2x2x2 -> 7
        i in {0,1}, j in {4},   k in {0,1}: 2x1x2 -> 4
        i in {0,1}, j in {4},   k in {2,3}: 2x1x2 -> 4
        i in {0,1}, j in {4},   k in {4,5}: 2x1x2 -> 4
        
        i in {2,3}, j in {0,1}, k in {0,1}: 2x2x2 -> 7
        i in {2,3}, j in {0,1}, k in {2,3}: 2x2x2 -> 7
        i in {2,3}, j in {0,1}, k in {4,5}: 2x2x2 -> 7
        i in {2,3}, j in {2,3}, k in {0,1}: 2x2x2 -> 7
        i in {2,3}, j in {2,3}, k in {2,3}: 2x2x2 -> 7
        i in {2,3}, j in {2,3}, k in {4,5}: 2x2x2 -> 7
        i in {2,3}, j in {4},   k in {0,1}: 2x1x2 -> 4
        i in {2,3}, j in {4},   k in {2,3}: 2x1x2 -> 4
        i in {2,3}, j in {4},   k in {4,5}: 2x1x2 -> 4
        
        Total Rank = 12 * 7 + 6 * 4 = 84 + 24 = 108.
        
        This is better than 120.
        """
        
        # Define partitions
        # i: [0,1], [2,3]
        # j: [0,1], [2,3], [4]
        # k: [0,1], [2,3], [4,5]
        
        i_splits = [(0, 2), (2, 4)]
        j_splits = [(0, 2), (2, 4), (4, 5)]
        k_splits = [(0, 2), (2, 4), (4, 6)]
        
        U_list = []
        V_list = []
        W_list = []
        
        for i_start, i_end in i_splits:
            for j_start, j_end in j_splits:
                for k_start, k_end in k_splits:
                    ni = i_end - i_start
                    mj = j_end - j_start
                    pk = k_end - k_start
                    
                    if ni == 2 and mj == 2 and pk == 2:
                        u, v, w = strassen_222()
                    else:
                        u, v, w = schoolbook(ni, mj, pk)
                    
                    # Shift indices
                    u_shifted = np.zeros((n * m, u.shape[1]))
                    v_shifted = np.zeros((m * p, v.shape[1]))
                    w_shifted = np.zeros((p * n, w.shape[1]))
                    
                    for r in range(u.shape[1]):
                        # U indices: i*m + j
                        for ii in range(ni):
                            for jj in range(mj):
                                u_shifted[(i_start + ii) * m + (j_start + jj), r] = u[ii * mj + jj, r]
                        
                        # V indices: j*p + k
                        for jj in range(mj):
                            for kk in range(pk):
                                v_shifted[(j_start + jj) * p + (k_start + kk), r] = v[jj * pk + kk, r]
                        
                        # W indices: k*n + i
                        for kk in range(pk):
                            for ii in range(ni):
                                w_shifted[(k_start + kk) * n + (i_start + ii), r] = w[kk * ni + ii, r]
                    
                    U_list.append(u_shifted)
                    V_list.append(v_shifted)
                    W_list.append(w_shifted)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        return {"U": U, "V": V, "W": W}

    def verify(result, atol=1e-8):
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build the decomposition
    result = build_decomposition()
    
    # Verify
    if not verify(result):
        # Fallback to schoolbook if verification fails
        U, V, W = schoolbook(n, m, p)
        result = {"U": U, "V": V, "W": W}

    return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
# EVOLVE-BLOCK-END
