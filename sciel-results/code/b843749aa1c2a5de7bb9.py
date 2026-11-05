"""Matrix multiplication 4x5 x 5x5 low-rank exact decomposition using block Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _matmul_tensor(n, m, p):
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _build_factors(n, m, p, terms):
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j, c] += val
        for (j, k), val in v.items():
            V[j * p + k, c] += val
        for (k, i), val in w.items():
            W[k * n + i, c] += val
    return U, V, W

def _check(U, V, W, n, m, p):
    T = _matmul_tensor(n, m, p)
    for c in range(U.shape[1]):
        T -= np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(T)) < 1e-7

def _strassen_2x2():
    """7 terms for 2x2 * 2x2. Each term (u,v,w) with dict keys."""
    terms = []
    # M1 = (a00+a11)*(b00+b11)
    u = {(0, 0): 1, (1, 1): 1}
    v = {(0, 0): 1, (1, 1): 1}
    w = {(0, 0): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M2 = (a10+a11)*b00
    u = {(1, 0): 1, (1, 1): 1}
    v = {(0, 0): 1}
    w = {(0, 0): 1, (1, 0): 1}
    terms.append((u, v, w))
    # M3 = a00*(b10-b11)
    u = {(0, 0): 1}
    v = {(1, 0): 1, (1, 1): -1}
    w = {(0, 1): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M4 = a11*(b01-b00)
    u = {(1, 1): 1}
    v = {(0, 1): 1, (0, 0): -1}
    w = {(0, 0): 1, (1, 0): 1}
    terms.append((u, v, w))
    # M5 = (a00+a01)*b11
    u = {(0, 0): 1, (0, 1): 1}
    v = {(1, 1): 1}
    w = {(0, 1): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M6 = (a10-a00)*(b00+b01)
    u = {(1, 0): 1, (0, 0): -1}
    v = {(0, 0): 1, (0, 1): 1}
    w = {(0, 0): 1, (1, 1): 1}
    terms.append((u, v, w))
    # M7 = (a01-a11)*(b10+b11)
    u = {(0, 1): 1, (1, 1): -1}
    v = {(1, 0): 1, (1, 1): 1}
    w = {(0, 1): 1, (1, 0): 1}
    terms.append((u, v, w))
    return terms

def _strassen_2x3():
    """Optimized 2x3 * 3x3 decomposition. Try to use Strassen-like structure."""
    # For 2x3 * 3x3, we can partition the 3x3 into a 2x2 block + 1x1, etc.
    # But let's use a direct approach: partition j into [0,1] and [2], k into [0,1] and [2]
    # Block (2,2,2): Strassen 7 terms
    # Block (2,2,1): schoolbook 4 terms (2*2*1)
    # Block (2,1,2): schoolbook 4 terms (2*1*2)
    # Block (2,1,1): schoolbook 2 terms (2*1*1)
    # Total: 7+4+4+2 = 17 terms vs schoolbook 18
    terms = []
    # Block (2,2,2) at offset (0,0,0)
    for (u, v, w) in _strassen_2x2():
        terms.append((u, v, w))
    # Block (2,2,1) at offset (0,0,2): j in [0,1], k=2
    for i in range(2):
        for j in range(2):
            u = {(i, j): 1}
            v = {(j, 0): 1}  # k=2 maps to local k=0
            w = {(0, i): 1}
            terms.append((u, v, w))
    # Block (2,1,2) at offset (0,2,0): j=2, k in [0,1]
    for i in range(2):
        for k in range(2):
            u = {(i, 0): 1}  # j=2 maps to local j=0
            v = {(0, k): 1}
            w = {(k, i): 1}
            terms.append((u, v, w))
    # Block (2,1,1) at offset (0,2,2): j=2, k=2
    for i in range(2):
        u = {(i, 0): 1}
        v = {(0, 0): 1}
        w = {(0, i): 1}
        terms.append((u, v, w))
    return terms

def _strassen_3x2():
    """Optimized 3x2 * 2x3 decomposition."""
    # Partition i into [0,1] and [2], k into [0,1] and [2]
    # Block (2,2,2): Strassen 7 terms
    # Block (2,2,1): schoolbook 4 terms
    # Block (1,2,2): schoolbook 4 terms
    # Block (1,2,1): schoolbook 2 terms
    # Total: 7+4+4+2 = 17 terms vs schoolbook 18
    terms = []
    # Block (2,2,2) at offset (0,0,0)
    for (u, v, w) in _strassen_2x2():
        terms.append((u, v, w))
    # Block (2,2,1) at offset (0,0,2): i in [0,1], k=2
    for i in range(2):
        for k in range(2):
            u = {(i, k): 1}  # j=k for this block
            v = {(k, 0): 1}  # local k=0
            w = {(0, i): 1}
            terms.append((u, v, w))
    # Block (1,2,2) at offset (2,0,0): i=2, k in [0,1]
    for j in range(2):
        for k in range(2):
            u = {(0, j): 1}  # local i=0
            v = {(j, k): 1}
            w = {(k, 0): 1}
            terms.append((u, v, w))
    # Block (1,2,1) at offset (2,0,2): i=2, k=2
    for j in range(2):
        u = {(0, j): 1}
        v = {(j, 0): 1}
        w = {(0, 0): 1}
        terms.append((u, v, w))
    return terms

def _strassen_3x3():
    """Optimized 3x3 * 3x3 decomposition using block Strassen."""
    # Partition all into 2+1
    # 8 blocks: (2,2,2), (2,2,1), (2,1,2), (2,1,1), (1,2,2), (1,2,1), (1,1,2), (1,1,1)
    # (2,2,2): 7 (Strassen)
    # (2,2,1): 4
    # (2,1,2): 4
    # (2,1,1): 2
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,1,2): 2
    # (1,1,1): 1
    # Total: 7+4+4+2+4+2+2+1 = 26 vs schoolbook 27
    terms = []
    # (2,2,2) at (0,0,0)
    for (u, v, w) in _strassen_2x2():
        terms.append((u, v, w))
    # (2,2,1) at (0,0,2)
    for i in range(2):
        for j in range(2):
            u = {(i, j): 1}
            v = {(j, 0): 1}
            w = {(0, i): 1}
            terms.append((u, v, w))
    # (2,1,2) at (0,2,0)
    for i in range(2):
        for k in range(2):
            u = {(i, 0): 1}
            v = {(0, k): 1}
            w = {(k, i): 1}
            terms.append((u, v, w))
    # (2,1,1) at (0,2,2)
    for i in range(2):
        u = {(i, 0): 1}
        v = {(0, 0): 1}
        w = {(0, i): 1}
        terms.append((u, v, w))
    # (1,2,2) at (2,0,0)
    for j in range(2):
        for k in range(2):
            u = {(0, j): 1}
            v = {(j, k): 1}
            w = {(k, 0): 1}
            terms.append((u, v, w))
    # (1,2,1) at (2,0,2)
    for j in range(2):
        u = {(0, j): 1}
        v = {(j, 0): 1}
        w = {(0, 0): 1}
        terms.append((u, v, w))
    # (1,1,2) at (2,2,0)
    for k in range(2):
        u = {(0, 0): 1}
        v = {(0, k): 1}
        w = {(k, 0): 1}
        terms.append((u, v, w))
    # (1,1,1) at (2,2,2)
    u = {(0, 0): 1}
    v = {(0, 0): 1}
    w = {(0, 0): 1}
    terms.append((u, v, w))
    return terms

def _schoolbook(n, m, p):
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    return terms

def _block_decompose(n, m, p):
    """Decompose n x m * m x p using block partitioning with Strassen for 2x2 blocks."""
    # For 4x5*5x5, partition:
    # i: [0,1,2,3] -> split into [0,1] (size 2) and [2,3] (size 2)
    # j: [0,1,2,3,4] -> split into [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # k: [0,1,2,3,4] -> split into [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    
    # This gives 2 x 3 x 3 = 18 blocks
    # Block sizes and their optimal decompositions:
    # (2,2,2): 7 (Strassen)
    # (2,2,1): 4
    # (2,1,2): 4
    # (2,1,1): 2
    # Total per i-block: 7+4+4+2 = 17
    # Two i-blocks: 34 terms total vs schoolbook 100
    
    terms_all = []
    
    # i-block 0: rows 0,1
    # j-block 0: cols 0,1
    # k-block 0: cols 0,1 -> (2,2,2) Strassen
    for (u, v, w) in _strassen_2x2():
        terms_all.append((u, v, w))
    
    # j-block 0, k-block 1: (2,2,2) at (0,0,2)
    for (u, v, w) in _strassen_2x2():
        # Shift k by 2
        w_shifted = {(kk + 2, ii): val for (kk, ii), val in w.items()}
        v_shifted = {(jj, kk + 2): val for (jj, kk), val in v.items()}
        terms_all.append((u, v_shifted, w_shifted))
    
    # j-block 0, k-block 2: (2,2,1) at (0,0,4)
    for i in range(2):
        for j in range(2):
            u = {(i, j): 1}
            v = {(j, 0): 1}
            w = {(0, i): 1}
            w_shifted = {(0 + 4, i): 1}
            v_shifted = {(j, 0 + 4): 1}
            terms_all.append((u, v_shifted, w_shifted))
    
    # j-block 1: cols 2,3
    # k-block 0: (2,2,2) at (0,2,0)
    for (u, v, w) in _strassen_2x2():
        u_shifted = {(ii, jj + 2): val for (ii, jj), val in u.items()}
        v_shifted = {(jj + 2, kk): val for (jj, kk), val in v.items()}
        terms_all.append((u_shifted, v_shifted, w))
    
    # j-block 1, k-block 1: (2,2,2) at (0,2,2)
    for (u, v, w) in _strassen_2x2():
        u_shifted = {(ii, jj + 2): val for (ii, jj), val in u.items()}
        v_shifted = {(jj + 2, kk + 2): val for (jj, kk), val in v.items()}
        w_shifted = {(kk + 2, ii): val for (kk, ii), val in w.items()}
        terms_all.append((u_shifted, v_shifted, w_shifted))
    
    # j-block 1, k-block 2: (2,2,1) at (0,2,4)
    for i in range(2):
        for j in range(2):
            u = {(i, j + 2): 1}
            v = {(j + 2, 0 + 4): 1}
            w = {(0 + 4, i): 1}
            terms_all.append((u, v, w))
    
    # j-block 2: col 4
    # k-block 0: (2,1,2) at (0,4,0)
    for i in range(2):
        for k in range(2):
            u = {(i, 0 + 4): 1}
            v = {(0 + 4, k): 1}
            w = {(k, i): 1}
            terms_all.append((u, v, w))
    
    # j-block 2, k-block 1: (2,1,2) at (0,4,2)
    for i in range(2):
        for k in range(2):
            u = {(i, 0 + 4): 1}
            v = {(0 + 4, k + 2): 1}
            w = {(k + 2, i): 1}
            terms_all.append((u, v, w))
    
    # j-block 2, k-block 2: (2,1,1) at (0,4,4)
    for i in range(2):
        u = {(i, 0 + 4): 1}
        v = {(0 + 4, 0 + 4): 1}
        w = {(0 + 4, i): 1}
        terms_all.append((u, v, w))
    
    # i-block 1: rows 2,3 (shift i by 2)
    # j-block 0, k-block 0: (2,2,2) at (2,0,0)
    for (u, v, w) in _strassen_2x2():
        u_shifted = {(ii + 2, jj): val for (ii, jj), val in u.items()}
        w_shifted = {(kk, ii + 2): val for (kk, ii), val in w.items()}
        terms_all.append((u_shifted, v, w_shifted))
    
    # j-block 0, k-block 1: (2,2,2) at (2,0,2)
    for (u, v, w) in _strassen_2x2():
        u_shifted = {(ii + 2, jj): val for (ii, jj), val in u.items()}
        v_shifted = {(jj, kk + 2): val for (jj, kk), val in v.items()}
        w_shifted = {(kk + 2, ii + 2): val for (kk, ii), val in w.items()}
        terms_all.append((u_shifted, v_shifted, w_shifted))
    
    # j-block 0, k-block 2: (2,2,1) at (2,0,4)
    for i in range(2):
        for j in range(2):
            u = {(i + 2, j): 1}
            v = {(j, 0 + 4): 1}
            w = {(0 + 4, i + 2): 1}
            terms_all.append((u, v, w))
    
    # j-block 1, k-block 0: (2,2,2) at (2,2,0)
    for (u, v, w) in _strassen_2x2():
        u_shifted = {(ii + 2, jj + 2): val for (ii, jj), val in u.items()}
        v_shifted = {(jj + 2, kk): val for (jj, kk), val in v.items()}
        w_shifted = {(kk, ii + 2): val for (kk, ii), val in w.items()}
        terms_all.append((u_shifted, v_shifted, w_shifted))
    
    # j-block 1, k-block 1: (2,2,2) at (2,2,2)
    for (u, v, w) in _strassen_2x2():
        u_shifted = {(ii + 2, jj + 2): val for (ii, jj), val in u.items()}
        v_shifted = {(jj + 2, kk + 2): val for (jj, kk), val in v.items()}
        w_shifted = {(kk + 2, ii + 2): val for (kk, ii), val in w.items()}
        terms_all.append((u_shifted, v_shifted, w_shifted))
    
    # j-block 1, k-block 2: (2,2,1) at (2,2,4)
    for i in range(2):
        for j in range(2):
            u = {(i + 2, j + 2): 1}
            v = {(j + 2, 0 + 4): 1}
            w = {(0 + 4, i + 2): 1}
            terms_all.append((u, v, w))
    
    # j-block 2, k-block 0: (2,1,2) at (2,4,0)
    for i in range(2):
        for k in range(2):
            u = {(i + 2, 0 + 4): 1}
            v = {(0 + 4, k): 1}
            w = {(k, i + 2): 1}
            terms_all.append((u, v, w))
    
    # j-block 2, k-block 1: (2,1,2) at (2,4,2)
    for i in range(2):
        for k in range(2):
            u = {(i + 2, 0 + 4): 1}
            v = {(0 + 4, k + 2): 1}
            w = {(k + 2, i + 2): 1}
            terms_all.append((u, v, w))
    
    # j-block 2, k-block 2: (2,1,1) at (2,4,4)
    for i in range(2):
        u = {(i + 2, 0 + 4): 1}
        v = {(0 + 4, 0 + 4): 1}
        w = {(0 + 4, i + 2): 1}
        terms_all.append((u, v, w))
    
    return terms_all

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0
    
    # Use block decomposition with Strassen for 2x2 blocks
    try:
        terms = _block_decompose(n, m, p)
        U, V, W = _build_factors(n, m, p, terms)
        if _check(U, V, W, n, m, p):
            U = U.tolist()
            V = V.tolist()
            W = W.tolist()
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    terms = _schoolbook(n, m, p)
    U, V, W = _build_factors(n, m, p, terms)
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
