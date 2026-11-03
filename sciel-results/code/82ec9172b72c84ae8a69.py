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
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # ===== Build Strassen 2x2x2 base (rank 7) =====
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
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
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)
    W2 = np.zeros((6, 11), dtype=float)
    
    # Terms 0-6: Strassen for columns 0,1
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
    
    # Terms 7-10: Schoolbook for column 2
    U2[0, 7] = 1; V2[2, 7] = 1; W2[4, 7] = 1
    U2[1, 8] = 1; V2[5, 8] = 1; W2[4, 8] = 1
    U2[2, 9] = 1; V2[2, 9] = 1; W2[5, 9] = 1
    U2[3, 10] = 1; V2[5, 10] = 1; W2[5, 10] = 1
    
    # ===== Kronecker product to get 4x4x6 (rank 77) =====
    rank = 7 * 11
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((24, rank), dtype=float)
    W = np.zeros((24, rank), dtype=float)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1 * 2 + j1, c1]
                            u2 = U2[i2 * 2 + j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1 * 8 + i2 * 4 + j1 * 2 + j2
                                U[row, c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1 * 2 + k1, c1]
                            v2 = V2[j2 * 3 + k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1 * 12 + j2 * 6 + k1 * 3 + k2
                                V[row, c] = v1 * v2
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
    def _verify(U_mat, V_mat, W_mat, n, m, p, seed=42):
        rng = np.random.RandomState(seed)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not _verify(U, V, W, n, m, p):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=float)
        V = np.zeros((m * p, rank_school), dtype=float)
        W = np.zeros((p * n, rank_school), dtype=float)
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
    if time.time() < deadline - 5:
        U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)
    
    # Convert to integer/half-integer if close
    U = _to_half_int(U)
    V = _to_half_int(V)
    W = _to_half_int(W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _to_half_int(mat):
    """Convert values close to half-integers to exact half-integers."""
    result = mat.copy()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            val = result[i, j]
            nearest_half = np.round(val * 2) / 2
            if abs(val - nearest_half) < 1e-9:
                result[i, j] = nearest_half
    return result


def _reduce_rank(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linear dependencies among rank-1 terms.
    
    Key insight: If we have rank-1 tensors T_r = u_r ⊗ v_r ⊗ w_r, they are linearly
    dependent iff the flattened vectors have linear dependencies. We can detect this
    by building the tensor matrix and finding its null space.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    rank = current_U.shape[1]
    
    def _verify_fast(U_mat, V_mat, W_mat, seed=42):
        rng = np.random.RandomState(seed)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Strategy: Try removing terms in groups by checking if the remaining terms
    # still span the same space. Use SVD to detect dependencies.
    
    # First pass: greedy single-term removal
    order = list(range(rank))
    rng = np.random.RandomState(123)
    rng.shuffle(order)
    
    removed = set()
    for r in order:
        if time.time() > deadline - 10:
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
    
    rank = current_U.shape[1]
    
    # Second pass: try removing pairs of terms
    if time.time() < deadline - 15:
        for r1 in range(rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 10:
                    break
                
                mask = np.ones(rank, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if _verify_fast(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    rank = current_U.shape[1]
                    break  # Restart pair search with new rank
    
    # Third pass: try removing triples
    if time.time() < deadline - 20:
        rank = current_U.shape[1]
        for r1 in range(rank):
            if time.time() > deadline - 15:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 15:
                    break
                for r3 in range(r2 + 1, rank):
                    if time.time() > deadline - 15:
                        break
                    
                    mask = np.ones(rank, dtype=bool)
                    mask[r1] = False
                    mask[r2] = False
                    mask[r3] = False
                    
                    U_test = current_U[:, mask]
                    V_test = current_V[:, mask]
                    W_test = current_W[:, mask]
                    
                    if _verify_fast(U_test, V_test, W_test):
                        current_U = U_test
                        current_V = V_test
                        current_W = W_test
                        rank = current_U.shape[1]
                        break
    
    return current_U, current_V, current_W
# EVOLVE-BLOCK-END
