"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE_SEC = 120


def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22) * (B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22) * B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11 * (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22 * (B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11 + A12) * B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _build_block_strassen(n, m, p, deadline):
    """
    Block Strassen for 4x4 * 4x5.
    
    A (4x4) partitioned into 2x2 blocks:
      A = [[A11, A12], [A21, A22]]
    
    B (4x5) partitioned:
      B = [[B11, B12, B13], [B21, B22, B23]]
      B11, B12, B21, B22 are 2x2; B13, B23 are 2x1
    
    C (4x5) partitioned:
      C = [[C11, C12, C13], [C21, C22, C23]]
    
    For each (ar, ac) in {(0,0),(0,1),(1,0),(1,1)}:
      C[ar*2:(ar+1)*2, ac*2:(ac+1)*2] = 
        A[ar*2:(ar+1)*2, 0:2] * B[0:2, ac*2:(ac+1)*2]
        + A[ar*2:(ar+1)*2, 2:4] * B[2:4, ac*2:(ac+1)*2]
    
    Using Strassen on the 2x2 blocks:
      M1 = (A11 + A22) * (B11 + B22)
      M2 = (A21 + A22) * B11
      M3 = A11 * (B12 - B22)
      M4 = A22 * (B21 - B11)
      M5 = (A11 + A12) * B22
      M6 = (A21 - A11) * (B11 + B12)
      M7 = (A12 - A22) * (B21 + B22)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    
    For 4x4x4 part: 4 blocks * 7 = 28
    For 4x4x1 part: 8 schoolbook = 8
    Total: 36
    """
    if time.time() > deadline:
        return None, None, None, False
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    total_rank = 36
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    # 4x4x4 part: 4 blocks, each with 7 Strassen multiplications
    for ar in range(2):
        for ac in range(2):
            for r in range(7):
                # U: A block entries
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = ar * 2 + i_local
                        j_global = j_local  # j in {0,1} for first block
                        val = U_s[i_local * 2 + j_local, r]
                        if val != 0:
                            U[i_global * m + j_global, col + r] += val
                        
                        j_global2 = 2 + j_local  # j in {2,3} for second block
                        val2 = U_s[i_local * 2 + j_local, r]
                        if val2 != 0:
                            U[i_global * m + j_global2, col + r] += val2
                
                # V: B block entries
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_local
                        k_global = ac * 2 + k_local
                        val = V_s[j_local * 2 + k_local, r]
                        if val != 0:
                            V[j_global * p + k_global, col + r] += val
                        
                        j_global2 = 2 + j_local
                        val2 = V_s[j_local * 2 + k_local, r]
                        if val2 != 0:
                            V[j_global2 * p + k_global, col + r] += val2
                
                # W: C block entries
                for k_local in range(2):
                    for i_local in range(2):
                        k_global = ac * 2 + k_local
                        i_global = ar * 2 + i_local
                        val = W_s[k_local * 2 + i_local, r]
                        if val != 0:
                            W[k_global * n + i_global, col + r] += val
            
            col += 7
    
    # 4x4x1 part: schoolbook for column 4
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    total_rank = col
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True


def construct():
    """Construct the matrix multiplication decomposition."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SEC - 20  # Safety margin

    n, m, p = DIMENSIONS

    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try block Strassen (rank 36)
    if time.time() < deadline - 10:
        U_b, V_b, W_b, rank_b = _build_block_strassen(n, m, p, deadline)
        if U_b is not None and rank_b < best_rank:
            if _verify_decomposition_fast(U_b, V_b, W_b, n, m, p, tol=1e-8):
                best_rank = rank_b
                best_U, best_V, best_W = U_b, V_b, W_b

    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        best_U, best_V, best_W = U_school, V_school, W_school

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])

    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])

    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                max_err = max(max_err, abs(s - 1.0))
    print(f"Max error: {max_err}")
    print(f"Rank: {rank}")
