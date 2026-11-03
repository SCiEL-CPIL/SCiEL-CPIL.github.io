import numpy as np
import time

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m x p matrix multiplication."""
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


def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication using Strassen's algorithm.
    Each term is (U_vec, V_vec, W_vec) where each vec has 4 entries.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[C00, C01], [C10, C11]]
    """
    terms = []
    # m1 = (a+d)(e+h)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # m2 = (c+d)e
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # m3 = a(f-h)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # m4 = d(g-e)
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 1, 0]))
    # m5 = (a+b)g
    terms.append(([1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]))
    # m6 = (c-a)(e+f)
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 1]))
    # m7 = (b-d)(h+f)
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [-1, 1, 1, 0]))
    return terms


def _strassen_4x4_terms():
    """
    Build 4x4x4 matrix multiplication using Kronecker product of 2x2x2 Strassen blocks.
    This gives rank 49 (7^2) instead of schoolbook rank 64.
    Returns list of (U_vec, V_vec, W_vec) where each vec has 16 entries.
    """
    base_terms = _strassen_2x2_terms()
    result_terms = []
    
    for t1 in base_terms:
        for t2 in base_terms:
            u1, v1, w1 = t1
            u2, v2, w2 = t2
            
            # Kronecker product: for each pair of 2x2 blocks
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            for a in range(4):
                for b in range(4):
                    # U index: (i*m + j) where i,j in 0..3
                    # Block structure: i = 2*(a//2) + (a%2), j = 2*(b//2) + (b%2)
                    i = a // 2
                    j = a % 2
                    k = b // 2
                    l = b % 2
                    U_vec[i * 4 + j] += u1[i * 2 + k] * u2[j * 2 + l]
                    
                    # V index: (j*p + k) where j,k in 0..3
                    # Block structure: j = 2*(a//2) + (a%2), k = 2*(b//2) + (b%2)
                    V_vec[j * 4 + k] += v1[i * 2 + k] * v2[j * 2 + l]
                    
                    # W index: (k*n + i) where k,i in 0..3
                    # Block structure: k = 2*(a//2) + (a%2), i = 2*(b//2) + (b%2)
                    W_vec[k * 4 + i] += w1[i * 2 + k] * w2[j * 2 + l]
            
            result_terms.append((U_vec, V_vec, W_vec))
    
    return result_terms


def _embed_terms(terms, n, m, p, i_offset=0, j_offset=0, k_offset=0, 
                 i_size=None, j_size=None, k_size=None):
    """
    Embed rank-1 terms into full n x m x p space.
    terms: list of (U_vec, V_vec, W_vec) for a sub-block
    Returns list of full-size vectors.
    """
    if i_size is None:
        i_size = int(np.sqrt(len(terms[0][0]))) if terms else 1
    if j_size is None:
        j_size = int(np.sqrt(len(terms[0][1]))) if terms else 1
    if k_size is None:
        k_size = int(np.sqrt(len(terms[0][2]))) if terms else 1
    
    embedded = []
    for u_sub, v_sub, w_sub in terms:
        U_full = [0] * (n * m)
        V_full = [0] * (m * p)
        W_full = [0] * (p * n)
        
        # Embed U: sub-index (i,j) -> full-index (i_offset+i, j_offset+j)
        for idx, val in enumerate(u_sub):
            if val != 0:
                i_sub = idx // j_size
                j_sub = idx % j_size
                i_full = i_offset + i_sub
                j_full = j_offset + j_sub
                if 0 <= i_full < n and 0 <= j_full < m:
                    U_full[i_full * m + j_full] = val
        
        # Embed V: sub-index (j,k) -> full-index (j_offset+j, k_offset+k)
        for idx, val in enumerate(v_sub):
            if val != 0:
                j_sub = idx // k_size
                k_sub = idx % k_size
                j_full = j_offset + j_sub
                k_full = k_offset + k_sub
                if 0 <= j_full < m and 0 <= k_full < p:
                    V_full[j_full * p + k_full] = val
        
        # Embed W: sub-index (k,i) -> full-index (k_offset+k, i_offset+i)
        for idx, val in enumerate(w_sub):
            if val != 0:
                k_sub = idx // i_size
                i_sub = idx % i_size
                k_full = k_offset + k_sub
                i_full = i_offset + i_sub
                if 0 <= k_full < p and 0 <= i_full < n:
                    W_full[k_full * n + i_full] = val
        
        embedded.append((U_full, V_full, W_full))
    
    return embedded


def _verify(U, V, W, rank, n, m, p, num_trials=3):
    """Verify the decomposition on random integer matrices."""
    import random
    for _ in range(num_trials):
        if remaining() <= 0:
            return False
        A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u = U[i * m + j][c]
                        if u == 0:
                            continue
                        acc += u * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True


def _build_strassen_4x4x4(n, m, p):
    """
    Build decomposition using Strassen for the 4x4x4 sub-block and schoolbook for the rest.
    Partition:
    - 4x4x4 block: i in [0,4), j in [0,4), k in [0,4) -> Strassen rank 49
    - 4x4x2 block: i in [0,4), j in [0,4), k in [4,6) -> schoolbook rank 32
    - 4x1x6 block: i in [0,4), j in [4,5), k in [0,6) -> schoolbook rank 24
    Total rank: 49 + 32 + 24 = 105
    """
    all_terms = []
    
    # 4x4x4 Strassen block
    strassen_terms = _strassen_4x4_terms()
    embedded_strassen = _embed_terms(strassen_terms, n, m, p, 
                                       i_offset=0, j_offset=0, k_offset=0,
                                       i_size=4, j_size=4, k_size=4)
    all_terms.extend(embedded_strassen)
    
    # 4x4x2 schoolbook block (k=4,5)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                U_vec[i * m + j] = 1
                V_vec[j * p + k] = 1
                W_vec[k * n + i] = 1
                all_terms.append((U_vec, V_vec, W_vec))
    
    # 4x1x6 schoolbook block (j=4)
    for i in range(4):
        for k in range(6):
            U_vec = [0] * (n * m)
            V_vec = [0] * (m * p)
            W_vec = [0] * (p * n)
            U_vec[i * m + 4] = 1
            V_vec[4 * p + k] = 1
            W_vec[k * n + i] = 1
            all_terms.append((U_vec, V_vec, W_vec))
    
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    best = None
    
    # Try Strassen-based decomposition (expected rank 105)
    try:
        if remaining() > 10:
            U, V, W, rank = _build_strassen_4x4x4(n, m, p)
            if remaining() > 5 and _verify(U, V, W, rank, n, m, p):
                best = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 120) always valid
    if best is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best = {"U": U, "V": V, "W": W}
    
    return best
