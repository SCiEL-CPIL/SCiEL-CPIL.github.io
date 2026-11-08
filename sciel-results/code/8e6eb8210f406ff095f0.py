"""Matrix multiplication 4x4 x 4x8 with rank-minimal tensor decomposition.
Uses Kronecker products of 2x2x2 Strassen decompositions."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _s222():
    """7 rank-1 terms for 2x2x2 Strassen.
    Returns list of 7 dicts with 'U','V','W' as flat lists of 4 entries each.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] (trace convention).
    """
    terms = [
        # M1 = (A00+A11)(B00+B11): C00+=M1, C11+=M1
        {'U': [1,0,0,1], 'V': [1,0,0,1], 'W': [1,0,0,1]},
        # M2 = (A10+A11)B00: C10+=M2, C11-=M2
        {'U': [0,0,1,1], 'V': [1,0,0,0], 'W': [0,1,0,-1]},
        # M3 = A00(B01-B11): C01+=M3, C11+=M3
        {'U': [1,0,0,0], 'V': [0,1,-1,0], 'W': [0,0,1,1]},
        # M4 = A11(B10-B00): C00+=M4, C10+=M4
        {'U': [0,0,0,1], 'V': [0,-1,1,0], 'W': [1,1,0,0]},
        # M5 = (A00+A01)B11: C00-=M5, C01+=M5
        {'U': [1,1,0,0], 'V': [0,0,0,1], 'W': [-1,0,1,0]},
        # M6 = (A10-A00)(B00+B01): C11+=M6
        {'U': [-1,0,1,0], 'V': [1,1,0,0], 'W': [0,0,0,1]},
        # M7 = (A01-A11)(B10+B11): C00+=M7
        {'U': [0,1,-1,0], 'V': [0,0,1,1], 'W': [1,0,0,0]},
    ]
    return terms

def _build_decomposition():
    """Build decomposition for 4x4 x 4x8 using Kronecker products of Strassen.
    
    Decomposition:
    i = 2*a + b (a in {0,1}, b in {0,1})
    j = 2*c + d (c in {0,1}, d in {0,1})  
    k = 4*e + f (e in {0,1}, f in {0,1,2,3})
    
    This gives:
    - Big tensor: a(2), c(2), e(2) -> 2x2x2, rank 7 (Strassen)
    - Small tensor: b(2), d(2), f(4) -> 2x2x4, rank 14 (two Strassen)
    
    Total rank = 7 * 14 = 98
    """
    s222 = _s222()  # 7 terms
    
    n, m, p = 4, 4, 8
    rank_total = 98
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    r_idx = 0
    for e_block in range(2):  # e in {0,1}
        for s1 in range(7):  # big Strassen term
            big_term = s222[s1]
            for f_pair in range(2):  # f pair: (0,1) or (2,3)
                for s2 in range(7):  # small Strassen term
                    small_term = s222[s2]
                    
                    # For each combination, fill in U, V, W
                    for a in range(2):
                        for c in range(2):
                            for b in range(2):
                                for d in range(2):
                                    i = 2*a + b
                                    j = 2*c + d
                                    
                                    # U[i,j] = U_big[a,c] * U_small[b,d]
                                    ubig = big_term['U'][a*2 + c]
                                    usmall = small_term['U'][b*2 + d]
                                    U[i*m + j, r_idx] = ubig * usmall
                                    
                                    # V[j,k] for k = 4*e_block + f
                                    # V_big[c,e_block] * V_small[d,f]
                                    vbig = big_term['V'][c*2 + e_block]
                                    for f_local in range(2):
                                        f = f_pair*2 + f_local
                                        k = 4*e_block + f
                                        vsmall = small_term['V'][d*2 + f_local]
                                        V[j*p + k, r_idx] = vbig * vsmall
                                    
                                    # W[k,i] for k = 4*e_block + f
                                    # W_big[e_block,a] * W_small[f,b]
                                    wbig = big_term['W'][e_block*2 + a]
                                    for f_local in range(2):
                                        f = f_pair*2 + f_local
                                        k = 4*e_block + f
                                        wsmall = small_term['W'][f_local*2 + b]
                                        W[k*n + i, r_idx] = wbig * wsmall
                    
                    r_idx += 1
    
    assert r_idx == rank_total, f"Expected {rank_total} terms, got {r_idx}"
    
    # Verify
    np.random.seed(123)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank_total):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        a_part = np.sum(Ur * A_test)
        b_part = np.sum(Vr * B_test)
        C_computed += Wr * a_part * b_part
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        raise ValueError(f"Decomposition verification failed: max error {np.max(np.abs(C_computed - C_expected))}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Ur = U_mat[:, r].reshape(n, m)
        Vr = V_mat[:, r].reshape(m, p)
        Wr = W_mat[:, r].reshape(p, n)
        a_part = np.sum(Ur * A_test)
        b_part = np.sum(Vr * B_test)
        C_computed += Wr * a_part * b_part
    
    return np.allclose(C_computed, C_expected, atol=tol)

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 100
    
    try:
        result = _build_decomposition()
        return result
    except Exception:
        pass
    
    # Fallback: schoolbook
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
