"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import copy

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U and len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication. Rank 7."""
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U (for A entries):
    # r=0: A[0,0]=1, A[1,1]=1  (a, d)
    # r=1: A[1,0]=1, A[1,1]=1  (c, d)
    # r=2: A[0,0]=1             (a)
    # r=3: A[1,1]=1             (d)
    # r=4: A[0,0]=1, A[0,1]=1  (a, b)
    # r=5: A[1,0]=1, A[0,0]=-1 (c, -a)
    # r=6: A[1,1]=1, A[0,1]=-1 (d, -b)
    
    # V (for B entries):
    # r=0: B[0,0]=1, B[1,1]=1  (e, h)
    # r=1: B[0,0]=1             (e)
    # r=2: B[0,1]=1, B[1,1]=-1 (f, -h)
    # r=3: B[1,0]=1, B[0,0]=-1 (g, -e)
    # r=4: B[1,1]=1             (h)
    # r=5: B[0,0]=1, B[0,1]=1  (e, f)
    # r=6: B[1,0]=1, B[1,1]=1  (g, h)
    
    # W (for C entries):
    # r=0: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=1
    # r=1: C[1,0]=-1, C[1,1]=-1
    # r=2: C[0,1]=1, C[1,1]=1
    # r=3: C[0,0]=1, C[1,0]=1
    # r=4: C[0,1]=-1, C[1,1]=-1
    # r=5: C[0,0]=1, C[1,0]=1
    # r=6: C[0,1]=-1, C[1,1]=-1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    U[0][2] = 1  # a
    U[3][3] = 1  # d
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    U[2][5] = 1  # c
    U[0][5] = -1 # -a
    U[3][6] = 1  # d
    U[1][6] = -1 # -b
    
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    V[0][1] = 1  # e
    V[1][2] = 1  # f
    V[3][2] = -1 # -h
    V[2][3] = 1  # g
    V[0][3] = -1 # -e
    V[3][4] = 1  # h
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    
    W[0][0] = 1  # C[0,0]
    W[1][0] = 1  # C[0,1]
    W[2][0] = 1  # C[1,0]
    W[3][0] = 1  # C[1,1]
    W[2][1] = -1 # C[1,0]
    W[3][1] = -1 # C[1,1]
    W[1][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    W[1][4] = -1 # C[0,1]
    W[3][4] = -1 # C[1,1]
    W[0][5] = 1  # C[0,0]
    W[2][5] = 1  # C[1,0]
    W[1][6] = -1 # C[0,1]
    W[3][6] = -1 # C[1,1]
    
    return U, V, W


def _combine_split_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two decompositions:
    First: n x m1 x p with U1 (n*m1 rows), V1 (m1*p rows), W1 (p*n rows)
    Second: n x m2 x p with U2 (n*m2 rows), V2 (m2*p rows), W2 (p*n rows)
    Result: n x (m1+m2) x p
    
    The full A has columns [A1 | A2], B has rows [B1; B2]
    C = A1*B1 + A2*B2
    
    U_full has rows: U1 (for A1 columns), then U2 (for A2 columns)
    V_full has rows: V1 (for B1 rows), then V2 (for B2 rows)
    W_full = W1 + W2 (element-wise, same shape p*n)
    """
    n_full = n
    m_full = m1 + m2
    p_full = p
    
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    rank_full = rank1 + rank2
    
    U_full = [[0] * rank_full for _ in range(n_full * m_full)]
    V_full = [[0] * rank_full for _ in range(m_full * p_full)]
    W_full = [[0] * rank_full for _ in range(p_full * n_full)]
    
    # Fill U_full
    for i in range(n):
        for j in range(m1):
            U_full[i * m_full + j] = U1[i * m1 + j] + [0] * rank2
        for j in range(m2):
            U_full[i * m_full + (m1 + j)] = [0] * rank1 + U2[i * m2 + j]
    
    # Fill V_full
    for j in range(m1):
        for k in range(p):
            V_full[j * p_full + k] = V1[j * p + k] + [0] * rank2
    for j in range(m2):
        for k in range(p):
            V_full[(m1 + j) * p_full + k] = [0] * rank1 + V2[j * p + k]
    
    # Fill W_full
    for k in range(p):
        for i in range(n):
            idx = k * n + i
            W_full[idx] = [W1[idx][r] for r in range(rank1)] + [W2[idx][r] for r in range(rank2)]
    
    return U_full, V_full, W_full


def _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """
    Combine two decompositions:
    First: n x m x p1 with U1 (n*m rows), V1 (m*p1 rows), W1 (p1*n rows)
    Second: n x m x p2 with U2 (n*m rows), V2 (m*p2 rows), W2 (p2*n rows)
    Result: n x m x (p1+p2)
    
    C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    
    U_full = U1 + U2 (element-wise, same shape)
    V_full has rows: V1 (for B1), then V2 (for B2)
    W_full has rows: W1 (for C1 cols), then W2 (for C2 cols)
    """
    n_full = n
    m_full = m
    p_full = p1 + p2
    
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    rank_full = rank1 + rank2
    
    U_full = [[0] * rank_full for _ in range(n_full * m_full)]
    V_full = [[0] * rank_full for _ in range(m_full * p_full)]
    W_full = [[0] * rank_full for _ in range(p_full * n_full)]
    
    # Fill U_full
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U_full[idx] = [U1[idx][r] for r in range(rank1)] + [U2[idx][r] for r in range(rank2)]
    
    # Fill V_full
    for j in range(m):
        for k in range(p1):
            V_full[j * p_full + k] = V1[j * p1 + k] + [0] * rank2
    for j in range(m):
        for k in range(p2):
            V_full[j * p_full + (p1 + k)] = [0] * rank1 + V2[j * p2 + k]
    
    # Fill W_full
    for k in range(p1):
        for i in range(n):
            idx = k * n + i
            W_full[idx] = W1[idx] + [0] * rank2
    for k in range(p2):
        for i in range(n):
            idx = (p1 + k) * n + i
            W_full[idx] = [0] * rank1 + W2[idx]
    
    return U_full, V_full, W_full


def _combine_split_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
    """
    Combine two decompositions:
    First: n1 x m x p with U1 (n1*m rows), V1 (m*p rows), W1 (p*n1 rows)
    Second: n2 x m x p with U2 (n2*m rows), V2 (m*p rows), W2 (p*n2 rows)
    Result: (n1+n2) x m x p
    
    C = [C1; C2] where C1 = A1*B, C2 = A2*B
    
    U_full has rows: U1 (for A1 rows), then U2 (for A2 rows)
    V_full = V1 + V2 (element-wise, same shape)
    W_full has rows: W1 (for C1 rows), then W2 (for C2 rows)
    """
    n_full = n1 + n2
    m_full = m
    p_full = p
    
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    rank_full = rank1 + rank2
    
    U_full = [[0] * rank_full for _ in range(n_full * m_full)]
    V_full = [[0] * rank_full for _ in range(m_full * p_full)]
    W_full = [[0] * rank_full for _ in range(p_full * n_full)]
    
    # Fill U_full
    for i in range(n1):
        for j in range(m):
            U_full[i * m + j] = U1[i * m + j] + [0] * rank2
    for i in range(n2):
        for j in range(m):
            U_full[(n1 + i) * m + j] = [0] * rank1 + U2[i * m + j]
    
    # Fill V_full
    for j in range(m):
        for k in range(p):
            idx = j * p + k
            V_full[idx] = [V1[idx][r] for r in range(rank1)] + [V2[idx][r] for r in range(rank2)]
    
    # Fill W_full
    for k in range(p):
        for i in range(n1):
            idx = k * n_full + i
            W_full[idx] = W1[k * n1 + i] + [0] * rank2
    for k in range(p):
        for i in range(n2):
            idx = k * n_full + (n1 + i)
            W_full[idx] = [0] * rank1 + W2[k * n2 + i]
    
    return U_full, V_full, W_full


def _decompose(n, m, p):
    """
    Recursive decomposition for n x m x p matrix multiplication.
    Returns U (n*m rows), V (m*p rows), W (p*n rows).
    """
    if time.time() > _DEADLINE - 10.0:
        return _schoolbook(n, m, p)
    
    # Base cases
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    if n == 1 and m == 1:
        # 1 x 1 x p: C[0,0] = sum_j A[0,j]*B[j,0]... wait, m=1 so C[0,0]=A[0,0]*B[0,0]
        # Actually for 1x1xp: A is 1x1, B is 1xp, C is 1xp
        # C[0,k] = A[0,0]*B[0,k]
        rank = p
        U = [[1] * rank]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p)]
        for k in range(p):
            V[k][k] = 1
            W[k][k] = 1
        return U, V, W
    
    if n == 1 and p == 1:
        # 1 x m x 1: C[0,0] = sum_j A[0,j]*B[j,0]
        rank = m
        U = [[0] * rank]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank]
        for j in range(m):
            U[0][j] = 1
            V[j][j] = 1
            W[0][j] = 1
        return U, V, W
    
    if m == 1 and p == 1:
        # n x 1 x 1: C[i,0] = A[i,0]*B[0,0]
        rank = n
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank]
        W = [[0] * rank for _ in range(n)]
        for i in range(n):
            U[i][i] = 1
            V[0][i] = 1
            W[i][i] = 1
        return U, V, W
    
    if n == 1:
        # 1 x m x p
        # C[0,k] = sum_j A[0,j]*B[j,k]
        # This is just dot product, rank = m*p
        # But we can do better by splitting
        if m == 1:
            return _decompose(1, 1, p)
        # Split m
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1 = _decompose(1, m1, p)
        U2, V2, W2 = _decompose(1, m2, p)
        return _combine_split_m(U1, V1, W1, U2, V2, W2, 1, m1, m2, p)
    
    if p == 1:
        # n x m x 1
        # C[i,0] = sum_j A[i,j]*B[j,0]
        # Rank = n*m
        if m == 1:
            return _decompose(n, 1, 1)
        # Split m
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1 = _decompose(n, m1, 1)
        U2, V2, W2 = _decompose(n, m2, 1)
        return _combine_split_m(U1, V1, W1, U2, V2, W2, n, m1, m2, 1)
    
    if m == 1:
        # n x 1 x p
        # C[i,k] = A[i,0]*B[0,k]
        # Rank = n*p
        if n == 1:
            return _decompose(1, 1, p)
        # Split n
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1 = _decompose(n1, 1, p)
        U2, V2, W2 = _decompose(n2, 1, p)
        return _combine_split_n(U1, V1, W1, U2, V2, W2, n1, n2, 1, p)
    
    # Check for Strassen case
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Recursive cases - choose split dimension
    # Prefer splitting the larger dimension
    
    # Try splitting m
    if m >= 2:
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1 = _decompose(n, m1, p)
        U2, V2, W2 = _decompose(n, m2, p)
        return _combine_split_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
    
    # Try splitting p
    if p >= 2:
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1 = _decompose(n, m, p1)
        U2, V2, W2 = _decompose(n, m, p2)
        return _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
    
    # Try splitting n
    if n >= 2:
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1 = _decompose(n1, m, p)
        U2, V2, W2 = _decompose(n2, m, p)
        return _combine_split_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
    
    # Fallback
    return _schoolbook(n, m, p)


def _decompose_optimized(n, m, p):
    """
    Optimized recursive decomposition that specifically handles
    3x4x6 by using the factorization structure.
    """
    if time.time() > _DEADLINE - 10.0:
        return _schoolbook(n, m, p)
    
    # Handle 3x4x6 specifically with optimal splits
    if n == 3 and m == 4 and p == 6:
        # Split m=4 into 2+2
        # Each gives 3x2x6
        U1, V1, W1 = _decompose_optimized(3, 2, 6)
        U2, V2, W2 = _decompose_optimized(3, 2, 6)
        return _combine_split_m(U1, V1, W1, U2, V2, W2, 3, 2, 2, 6)
    
    if n == 3 and m == 2 and p == 6:
        # Split p=6 into 3+3
        # Each gives 3x2x3
        U1, V1, W1 = _decompose_optimized(3, 2, 3)
        U2, V2, W2 = _decompose_optimized(3, 2, 3)
        return _combine_split_p(U1, V1, W1, U2, V2, W2, 3, 2, 3, 3)
    
    if n == 3 and m == 2 and p == 3:
        # Split n=3 into 1+2
        # 1x2x3 and 2x2x3
        U1, V1, W1 = _decompose_optimized(1, 2, 3)
        U2, V2, W2 = _decompose_optimized(2, 2, 3)
        return _combine_split_n(U1, V1, W1, U2, V2, W2, 1, 2, 2, 3)
    
    if n == 1 and m == 2 and p == 3:
        # 1x2x3: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # Rank 6 (schoolbook) or can we do better?
        # Actually for 1xm x p, the tensor is the "dot product" tensor
        # Rank of 1xm x p tensor is m*p (it's the identity in a sense)
        # No, wait. Let me think again.
        # C[0,k] = sum_j A[0,j]*B[j,k]
        # This is a bilinear form, rank = m*p
        # But we can split m or p
        # Split m=2 into 1+1
        U1, V1, W1 = _decompose_optimized(1, 1, 3)
        U2, V2, W2 = _decompose_optimized(1, 1, 3)
        return _combine_split_m(U1, V1, W1, U2, V2, W2, 1, 1, 1, 3)
    
    if n == 2 and m == 2 and p == 3:
        # 2x2x3: try to find low-rank decomposition
        return _find_2x2x3_decomposition()
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    if n == 2 and m == 2 and p == 1:
        # 2x2x1: C[i,0] = sum_j A[i,j]*B[j,0]
        # Rank 4
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        U[0][0] = 1  # A[0,0]
        U[1][1] = 1  # A[0,1]
        U[2][2] = 1  # A[1,0]
        U[3][3] = 1  # A[1,1]
        V[0][0] = 1  # B[0,0]
        V[1][1] = 1  # B[1,0]
        V[0][2] = 1  # B[0,0]
        V[1][3] = 1  # B[1,0]
        W[0][0] = 1  # C[0,0]
        W[0][1] = 1  # C[0,0]
        W[1][2] = 1  # C[1,0]
        W[1][3] = 1  # C[1,0]
        return U, V, W
    
    # General fallback to _decompose
    return _decompose(n, m, p)


def _try_rank_am(n, m, p, rank, max_iter=300, tol=1e-8, seed=None):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    obj += (s - 1.0) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None


def _find_2x2x3_decomposition():
    """Try to find a low-rank decomposition for 2x2x3."""
    n, m, p = 2, 2, 3
    best_result = None
    best_rank = 11  # Strassen + schoolbook fallback
    
    for rank in range(8, 11):
        if time.time() > _DEADLINE - 20.0:
            break
        for seed in range(10):
            if time.time() > _DEADLINE - 10.0:
                break
            result = _try_rank_am(n, m, p, rank, max_iter=500, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    best_result = (U, V, W)
                    best_rank = rank
                    break
        if best_result is not None and best_rank <= 9:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Strassen(2x2x2) + schoolbook(2x2x1)
    U1, V1, W1 = _strassen_2x2x2()
    rank = 11
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(4)]
    
    # Copy Strassen for p=0,1
    for i in range(4):
        for r in range(7):
            U[i][r] = U1[i][r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j * 3 + k][r] = V1[j * 2 + k][r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k * 2 + i][r] = W1[k * 2 + i][r]
    
    # Schoolbook for p=2
    for i in range(2):
        for j in range(2):
            idx = 7 + i * 2 + j
            U[i * 2 + j][idx] = 1
            V[j * 3 + 2][idx] = 1
            W[2 * 2 + i][idx] = 1
    
    return U, V, W


def _try_rank_gradient(n, m, p, rank, max_iter=500, tol=1e-10, lr=0.01, seed=None):
    """Try to find a decomposition using gradient descent on the smooth objective."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Compute objective and gradients
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = s - 1.0
                    obj += err * err
                    for r in range(rank):
                        grad_U[i * m + j, r] += 2 * err * V[j * p + k, r] * W[k * n + i, r]
                        grad_V[j * p + k, r] += 2 * err * U[i * m + j, r] * W[k * n + i, r]
                        grad_W[k * n + i, r] += 2 * err * U[i * m + j, r] * V[j * p + k, r]
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
        
        # Gradient step with adaptive learning rate
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm < 1e-15:
            break
        
        step = lr / (1.0 + it * 0.01)
        U -= step * grad_U
        V -= step * grad_V
        W -= step * grad_W
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # First, try the optimized recursive decomposition
    try:
        U, V, W = _decompose_optimized(n, m, p)
        if _verify(U, V, W, n, m, p, tol=1e-6):
            best_result = (U, V, W)
            best_rank = len(U[0])
    except:
        pass
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
