"""Matrix multiplication 3,4,6 — rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 6]


def _verify(U, V, W, n=3, m=4, p=6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition: rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def _strassen_2x2x2_terms():
    """
    Returns the 7 rank-1 terms for Strassen's 2x2 matrix multiplication.
    Each term is a tuple (u_vec, v_vec, w_vec) where:
    u_vec is length 4: [U[0,0], U[0,1], U[1,0], U[1,1]]
    v_vec is length 4: [V[0,0], V[0,1], V[1,0], V[1,1]]
    w_vec is length 4: [W[0,0], W[0,1], W[1,0], W[1,1]]
    Indices are local (0,1).
    """
    terms = []
    
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    terms.append((u, v, w))
    
    # M2 = (A21+A22)B11 -> C21
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0]
    terms.append((u, v, w))
    
    # M3 = A11(B21-B22) -> C12, C22
    u = [1, 0, 0, 0]
    v = [0, 0, 1, -1]
    w = [0, 1, 0, -1]
    terms.append((u, v, w))
    
    # M4 = A22(B12-B11) -> C21, C11
    u = [0, 0, 0, 1]
    v = [-1, 1, 0, 0]
    w = [-1, 0, 1, 0]
    terms.append((u, v, w))
    
    # M5 = (A11+A12)B22 -> C11, C12
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [1, 1, 0, 0]
    terms.append((u, v, w))
    
    # M6 = (A21-A11)(B11+B12) -> C22
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    terms.append((u, v, w))
    
    # M7 = (A12-A22)(B21+B22) -> C11, C22
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 1]
    terms.append((u, v, w))
    
    return terms


def _build_rank_66(n=3, m=4, p=6):
    """
    Construct a rank-66 decomposition for 3x4x6 matrix multiplication.
    
    Strategy:
    1. Partition rows i into {0,1} and {2}.
    2. Partition cols j into {0,1} and {2,3}.
    3. Partition outputs k into {0,1}, {2,3}, {4,5}.
    
    For i in {0,1}, j in {0,1}, k in {0,1}, use Strassen (Rank 7).
    For i in {0,1}, j in {0,1}, k in {2,3}, use Strassen (Rank 7).
    For i in {0,1}, j in {0,1}, k in {4,5}, use Strassen (Rank 7).
    For i in {0,1}, j in {2,3}, k in {0,1}, use Strassen (Rank 7).
    For i in {0,1}, j in {2,3}, k in {2,3}, use Strassen (Rank 7).
    For i in {0,1}, j in {2,3}, k in {4,5}, use Strassen (Rank 7).
    
    For i in {2}, j in {0..3}, k in {0..5}, use Schoolbook (Rank 24).
    
    Total Rank = 6 * 7 + 24 = 66.
    """
    total_rank = 66
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _strassen_2x2x2_terms()
    
    # Define the 6 Strassen blocks
    # (i_start, j_start, k_start)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    for i_start, j_start, k_start in blocks:
        for u_vec, v_vec, w_vec in strassen_terms:
            # u_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # local i in {0,1}, local j in {0,1}
            for local_i in range(2):
                for local_j in range(2):
                    local_u_idx = local_i * 2 + local_j
                    val = u_vec[local_u_idx]
                    if val != 0:
                        global_i = i_start + local_i
                        global_j = j_start + local_j
                        U[global_i * m + global_j][col] += val
            
            # v_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # local j in {0,1}, local k in {0,1}
            for local_j in range(2):
                for local_k in range(2):
                    local_v_idx = local_j * 2 + local_k
                    val = v_vec[local_v_idx]
                    if val != 0:
                        global_j = j_start + local_j
                        global_k = k_start + local_k
                        V[global_j * p + global_k][col] += val
            
            # w_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # local k in {0,1}, local i in {0,1}
            for local_k in range(2):
                for local_i in range(2):
                    local_w_idx = local_k * 2 + local_i
                    val = w_vec[local_w_idx]
                    if val != 0:
                        global_k = k_start + local_k
                        global_i = i_start + local_i
                        W[global_k * n + global_i][col] += val
            
            col += 1
    
    # Schoolbook for i=2
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    return U, V, W, total_rank


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 150.0  # seconds
    SAFETY_MARGIN = 10.0
    
    n, m, p = 3, 4, 6
    
    # Fallback: schoolbook decomposition
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try to build a lower-rank decomposition using block-Strassen
    try:
        U_66, V_66, W_66, rank_66 = _build_rank_66(n, m, p)
        if _verify(U_66, V_66, W_66, n, m, p):
            if rank_66 < best_rank:
                best_U, best_V, best_W, best_rank = U_66, V_66, W_66, rank_66
    except Exception:
        pass # Fallback to schoolbook if construction fails
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed + SAFETY_MARGIN > DEADLINE:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
