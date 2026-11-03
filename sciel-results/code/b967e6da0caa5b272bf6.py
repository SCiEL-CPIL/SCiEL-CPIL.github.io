"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product (2x2x2) ⊗ (2x2x3)
    - 2x2x2: Strassen (rank 7)
    - 2x2x3: Strassen for cols 0-1 (rank 7) + schoolbook for col 2 (rank 4) = rank 11
    - Total initial rank: 7 * 11 = 77
    - Then attempt rank reduction via linear dependency detection
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # ===== Build Strassen 2x2x2 base (rank 7) =====
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # U rows indexed by i*m+j = i*2+j, V rows by j*p+k = j*2+k, W rows by k*n+i = k*2+i
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # M2 = (a10+a11)b00 -> C10 += 1, C11 += -1
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # M3 = a00(b01-b11) -> C01 += 1, C11 += 1
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    
    # M4 = a11(b10-b00) -> C00 += 1, C10 += 1
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    
    # M5 = (a00+a01)b11 -> C00 += -1, C01 += 1
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # M6 = (a10-a00)(b00+b01) -> C11 += 1
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11) -> C00 += 1
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    
    # ===== Build 2x2x3 base (rank 11) =====
    # A(2x2) * B(2x3) -> C(2x3)
    # U rows: i*2+j (4 rows), V rows: j*3+k (6 rows), W rows: k*2+i (6 rows)
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Terms 0-6: Strassen for columns 0,1
    # V1 row j*2+k maps to V2 row j*3+k
    # W1 row k*2+i maps to W2 row k*2+i (k only 0,1 for these terms)
    for c in range(7):
        U2[:, c] = U1[:, c]
        # V mapping: V1[j*2+k] -> V2[j*3+k]
        V2[0, c] = V1[0, c]  # j=0,k=0
        V2[1, c] = V1[1, c]  # j=0,k=1
        V2[3, c] = V1[2, c]  # j=1,k=0
        V2[4, c] = V1[3, c]  # j=1,k=1
        # W mapping: W1[k*2+i] -> W2[k*2+i]
        W2[0, c] = W1[0, c]  # k=0,i=0
        W2[1, c] = W1[1, c]  # k=0,i=1
        W2[2, c] = W1[2, c]  # k=1,i=0
        W2[3, c] = W1[3, c]  # k=1,i=1
    
    # Terms 7-10: Schoolbook for column 2
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U2[0, 7] = 1
    V2[2, 7] = 1   # j=0, k=2 -> row 0*3+2=2
    W2[4, 7] = 1   # k=2, i=0 -> row 2*2+0=4
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U2[1, 8] = 1
    V2[5, 8] = 1   # j=1, k=2 -> row 1*3+2=5
    W2[4, 8] = 1   # k=2, i=0 -> row 2*2+0=4
    
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U2[2, 9] = 1
    V2[2, 9] = 1   # j=0, k=2 -> row 0*3+2=2
    W2[5, 9] = 1   # k=2, i=1 -> row 2*2+1=5
    
    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U2[3, 10] = 1
    V2[5, 10] = 1  # j=1, k=2 -> row 1*3+2=5
    W2[5, 10] = 1  # k=2, i=1 -> row 2*2+1=5
    
    # ===== Kronecker product to get 4x4x6 (rank 77) =====
    rank = 7 * 11
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    # Index mappings:
    # A[i,j]: i=i1*2+i2, j=j1*2+j2, row = i*4+j = i1*8+i2*4+j1*2+j2
    # B[j,k]: j=j1*2+j2, k=k1*3+k2, row = j*6+k = j1*12+j2*6+k1*3+k2
    # C[i,k]: i=i1*2+i2, k=k1*3+k2, row = k*4+i = k1*12+k2*4+i1*2+i2
    # U1 row: i1*2+j1, U2 row: i2*2+j2
    # V1 row: j1*2+k1, V2 row: j2*3+k2
    # W1 row: k1*2+i1, W2 row: k2*2+i2
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            # U
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1 * 2 + j1, c1]
                            u2 = U2[i2 * 2 + j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1 * 8 + i2 * 4 + j1 * 2 + j2
                                U[row, c] = u1 * u2
            # V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1 * 2 + k1, c1]
                            v2 = V2[j2 * 3 + k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1 * 12 + j2 * 6 + k1 * 3 + k2
                                V[row, c] = v1 * v2
            # W
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1 * 2 + i1, c1]
                            w2 = W2[k2 * 2 + i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1 * 12 + k2 * 4 + i1 * 2 + i2
                                W[row, c] = w1 * w2
    
    # ===== Verify =====
    def _verify(U_mat, V_mat, W_mat, n, m, p):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # ===== Rank reduction via linear dependency detection =====
    if time.time() < deadline - 10:
        U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _reduce_rank(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linear dependencies among rank-1 terms.
    
    A set of rank-1 terms {U_r, V_r, W_r} is linearly dependent if there exist
    coefficients c_r (not all zero) such that sum_r c_r * U_r ⊗ V_r ⊗ W_r = 0.
    
    We detect this by checking if removing a term still preserves the decomposition.
    Use a greedy approach with random restarts.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    rank = current_U.shape[1]
    
    def _verify_fast(U_mat, V_mat, W_mat):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Greedy removal: try removing each term, check if decomposition still holds
    # Process in random order for better results
    order = list(range(rank))
    np.random.seed(123)
    np.random.shuffle(order)
    
    removed = set()
    for r in order:
        if time.time() > deadline - 5:
            break
        if r in removed:
            continue
        if current_U.shape[1] <= 1:
            break
        
        mask = np.ones(current_U.shape[1], dtype=bool)
        mask[r] = False
        n_active = mask.sum()
        
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        
        if _verify_fast(U_test, V_test, W_test):
            current_U = U_test
            current_V = V_test
            current_W = W_test
            removed.add(r)
    
    return current_U, current_V, current_W
# EVOLVE-BLOCK-END
