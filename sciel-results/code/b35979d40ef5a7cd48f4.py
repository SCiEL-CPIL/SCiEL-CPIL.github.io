"""Matrix multiplication 2x2 by 2x16 - rank-minimal tensor decomposition."""
import time
import numpy as np

N, M, P = 2, 2, 16


def _target_tensor():
    """Build the matrix-multiplication tensor T[a,b,c] = 1 iff i=i', j=j', k=k'."""
    T = np.zeros((N * M, M * P, P * N), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1
    return T


def _verify(U, V, W):
    """Exact check that sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c] for all a,b,c."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    T = _target_tensor().astype(np.float64)
    return bool(np.allclose(recon, T, atol=1e-9, rtol=0.0))


def _strassen_decomposition():
    """
    Rank-56 decomposition: 2x2 * 2x16 = direct sum over 8 disjoint 2x2 * 2x2
    sub-problems (blocks of 2 output columns), each using Strassen's 7 factors.
    The B and C slot supports of different blocks are disjoint, so the rank is
    additive: 8 * 7 = 56.
    """
    n, m, p = N, M, P
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, -1]),  # M7
    ]
    num_blocks = p // 2  # 8 blocks of 2 columns
    rank = num_blocks * len(S)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for q in range(num_blocks):
        col_start = 2 * q
        for a_form, b_form, c_contrib in S:
            U[:, col] = a_form
            for jp in range(2):
                for kp in range(2):
                    V[jp * p + (col_start + kp), col] = b_form[jp * 2 + kp]
            for ip in range(2):
                for kp in range(2):
                    W[(col_start + kp) * n + ip, col] = c_contrib[ip * 2 + kp]
            col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook():
    """Standard rank 64 decomposition (fallback)."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _coppersmith_2x2x4():
    """
    Attempt a Coppersmith-Winograd style decomposition for 2x2 * 2x4.
    This is a structural experiment: reshape the 2x4 multiplication as a 
    convolution-like structure and try to find rank < 28.
    
    For 2x2 * 2x4:
    A is 2x2 (4 entries), B is 2x4 (8 entries), C is 4x2 (8 entries)
    Standard rank: 4*2*4 = 32... wait no, it's 2*2*4 = 16 schoolbook.
    Strassen block: 2 blocks of 2x2*2x2, rank 14.
    
    Let's try to find rank < 14 for 2x2 * 2x4.
    """
    n, m, p = 2, 2, 4
    # Schoolbook rank for 2x2 * 2x4 is 2*2*4 = 16
    # Strassen gives 2*7 = 14
    
    # Try to find a rank-13 or lower decomposition
    # This is a small enough problem to attempt a search
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # We'll try a structured approach: 
    # The tensor has a specific structure. Let's see if we can find 
    # a decomposition with rank < 14.
    
    # For now, return the Strassen-based rank-14 as a baseline
    # and note that finding rank < 14 would require new methods.
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, -1]),
    ]
    num_blocks = p // 2
    rank = num_blocks * len(S)
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for q in range(num_blocks):
        col_start = 2 * q
        for a_form, b_form, c_contrib in S:
            U[:, col] = a_form
            for jp in range(2):
                for kp in range(2):
                    V[jp*p + (col_start+kp), col] = b_form[jp*2+kp]
            for ip in range(2):
                for kp in range(2):
                    W[(col_start+kp)*n + ip, col] = c_contrib[ip*2+kp]
            col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}


def _combined_decomposition():
    """
    Structural approach: treat 2x2 * 2x16 as two interleaved 2x2 * 2x8 problems,
    or as a single problem with a different block structure.
    
    Key insight: The 2x16 matrix B can be viewed as 4 blocks of 2x4, or 
    8 blocks of 2x2. The current approach uses 8 blocks of 2x2 with Strassen.
    
    Alternative: Use a 2x2 * 2x4 sub-problem solved with rank 13 (if possible),
    giving 4 * 13 = 52 total rank.
    
    Since we haven't found rank < 14 for 2x2*2x4, let's try a different 
    structural decomposition.
    
    Another approach: View the multiplication as C = A * B where A is 2x2 
    and B is 2x16. This is equivalent to computing two matrix-vector products:
    C[:, 0:16] = A * B[:, 0:16]
    
    We can write A = [[a, b], [c, d]] and B has rows B1 (length 16) and B2 (length 16).
    C = [[a*B1 + b*B2], [c*B1 + d*B2]]
    
    This is a linear map from the 4-dimensional space of A to the 32-dimensional 
    space of C, with B as parameters. The tensor structure is:
    T[i,j,k] where i indexes A (4 values), j indexes B (32 values), k indexes C (32 values).
    
    Actually, let's think about this differently. The standard decomposition 
    has rank 64 (schoolbook). Strassen gives 56. Can we do better?
    
    The lower bound for matrix multiplication rank is related to the 
    bilinear complexity. For 2x2 * 2x16, the schoolbook is 64, Strassen 
    blocks give 56.
    
    Let's try a mixed approach: use Strassen for some blocks and a 
    different method for others, or find a global structure.
    """
    # For now, return the standard Strassen decomposition as the best known
    return _strassen_decomposition()


def construct():
    """Return the rank-minimal tensor decomposition for 2x2 * 2x16."""
    start = time.time()
    deadline = 130.0  # safety margin under the 150s limit

    # Primary: Rank-56 Strassen (provable optimum for block decomposition)
    try:
        cand = _strassen_decomposition()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass

    if time.time() - start > deadline:
        return _schoolbook()

    # Fallback: schoolbook rank 64
    cand = _schoolbook()
    if _verify(cand["U"], cand["V"], cand["W"]):
        return cand
    return cand
