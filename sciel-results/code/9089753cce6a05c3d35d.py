import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """
    Returns the 7 terms of the Strassen decomposition for 2x2x2.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u1 = np.array([1, 0, 0, 1])
    v1 = np.array([1, 0, 0, 1])
    w1 = np.array([1, 0, 0, 1])
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)B00
    u2 = np.array([0, 0, 1, 1])
    v2 = np.array([1, 0, 0, 0])
    w2 = np.array([0, 1, 0, -1])
    terms.append((u2, v2, w2))
    
    # M3 = A00(B01-B11)
    u3 = np.array([1, 0, 0, 0])
    v3 = np.array([0, 1, -1, 0])
    w3 = np.array([0, 0, 1, 1])
    terms.append((u3, v3, w3))
    
    # M4 = A11(B10-B00)
    u4 = np.array([0, 0, 0, 1])
    v4 = np.array([0, -1, 1, 0])
    w4 = np.array([1, 1, 0, 0])
    terms.append((u4, v4, w4))
    
    # M5 = (A00+A01)B11
    u5 = np.array([1, 1, 0, 0])
    v5 = np.array([0, 0, 0, 1])
    w5 = np.array([-1, 0, 1, 0])
    terms.append((u5, v5, w5))
    
    # M6 = (A10-A00)(B00+B01)
    u6 = np.array([-1, 0, 1, 0])
    v6 = np.array([1, 1, 0, 0])
    w6 = np.array([0, 0, 0, 1])
    terms.append((u6, v6, w6))
    
    # M7 = (A01-A11)(B10+B11)
    u7 = np.array([0, 1, -1, 0])
    v7 = np.array([0, 0, 1, 1])
    w7 = np.array([1, 0, 0, 0])
    terms.append((u7, v7, w7))
    
    return terms

def _build_4x4x4_strassen():
    """
    Build a 4x4x4 decomposition using Kronecker product of two Strassen decompositions.
    Rank = 7 * 7 = 49.
    """
    s222 = _strassen_2x2x2()
    terms_444 = []
    for u1, v1, w1 in s222:
        for u2, v2, w2 in s222:
            u = np.zeros(16)
            v = np.zeros(16)
            w = np.zeros(16)
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            u[i*4+j] = u1[i1*2+j1] * u2[i2*2+j2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            v[j*4+k] = v1[j1*2+k1] * v2[j2*2+k2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = 2*k1 + k2
                            i = 2*i1 + i2
                            w[k*4+i] = w1[k1*2+i1] * w2[k2*2+i2]
            terms_444.append((u, v, w))
    return terms_444

def _build_4x4x8():
    """
    Construct 4x4x8 decomposition.
    Rank = 2 * 49 = 98.
    """
    terms_444 = _build_4x4x4_strassen()
    rank_444 = len(terms_444)
    rank_total = 2 * rank_444
    
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    for idx, (u, v, w) in enumerate(terms_444):
        r_idx = idx
        U[:, r_idx] = u
        V[0:16, r_idx] = v
        W[0:16, r_idx] = w
        
        r_idx = rank_444 + idx
        U[:, r_idx] = u
        V[16:32, r_idx] = v
        W[16:32, r_idx] = w
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_4x4x8()
        
        # Verification
        np.random.seed(123)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        rank = U.shape[1]
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            a_part = Ur * A_test
            b_part = Vr * B_test
            C_computed += np.outer(np.sum(a_part, axis=1), np.sum(b_part, axis=0)) * Wr.reshape(p, n)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            raise ValueError("Verification failed")
            
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
        
    except Exception as e:
        # Fallback to schoolbook rank 128
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
        
        return {
            "U": [[float(x) for x in row] for row in U_sb],
            "V": [[float(x) for x in row] for row in V_sb],
            "W": [[float(x) for x in row] for row in W_sb]
        }
