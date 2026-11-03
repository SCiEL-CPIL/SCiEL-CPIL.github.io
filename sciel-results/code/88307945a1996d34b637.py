import time
import numpy as np

_DIMS = (3, 5, 5)
_T0 = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _strassen_triplets():
    """Return the 7 Strassen (U,V,W) index patterns for 2x2x2."""
    P = []
    # M1 = (A11+A22)(B11+B22)
    P.append(((0,0,1),(1,1,1)),
              ((0,0,1),(1,1,1)),
              ((0,0,1),(1,1,1)))
    # M2 = (A21+A22)B11
    P.append(((1,0,1),(1,1,1)),
              ((0,0,1),),
              ((0,1,1),(1,1,-1)))
    # M3 = A11(B21-B22)
    P.append(((0,0,1),),
              ((0,1,1),(1,1,-1)),
              ((1,0,1),(1,1,1)))
    # M4 = A22(B12-B11)
    P.append(((1,1,1),),
              ((1,0,1),(0,0,-1)),
              ((0,0,1),(0,1,1)))
    # M5 = (A11+A12)B22
    P.append(((0,0,1),(0,1,1)),
              ((1,1,1),),
              ((0,0,-1),(1,0,1)))
    # M6 = (A21-A11)(B11+B12)
    P.append(((1,0,1),(0,0,-1)),
              ((0,0,1),(0,1,1)),
              ((1,1,1),))
    # M7 = (A12-A22)(B21+B22)
    P.append(((0,1,1),(1,1,-1)),
              ((1,0,1),(1,1,1)),
              ((0,0,1),))
    return P

def construct():
    n, m, p = _DIMS
    
    # Block partition: n=[2,1], m=[2,2,1], p=[2,2,1]
    # This yields 4 Strassen 2x2x2 blocks and 14 schoolbook blocks
    n_parts = (2, 1)
    m_parts = (2, 2, 1)
    p_parts = (2, 2, 1)
    
    # Precompute Strassen patterns
    s_pat = _strassen_triplets()
    
    # Accumulate into dense arrays; first determine total rank
    total_rank = 0
    for nb in n_parts:
        for mb in m_parts:
            for pb in p_parts:
                if nb == 2 and mb == 2 and pb == 2:
                    total_rank += 7
                else:
                    total_rank += nb * mb * pb
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    ni = 0
    for nb in n_parts:
        mi = 0
        for mb in m_parts:
            pi = 0
            for pb in p_parts:
                if nb == 2 and mb == 2 and pb == 2:
                    for (u_idx, v_idx, w_idx) in s_pat:
                        for (li, lj, cv) in u_idx:
                            U[ni + li, mi + lj, col] = cv
                        for (lj, lk, cv) in v_idx:
                            V[mi + lj, pi + lk, col] = cv
                        for (lk, li, cv) in w_idx:
                            W[pi + lk, ni + li, col] = cv
                        col += 1
                else:
                    for a in range(nb):
                        for b in range(mb):
                            for c in range(pb):
                                U[ni + a, mi + b, col] = 1.0
                                V[mi + b, pi + c, col] = 1.0
                                W[pi + c, ni + a, col] = 1.0
                                col += 1
                pi += pb
            mi += mb
        ni += nb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
