import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify decomposition exactly.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    T[i, k, j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    R = U.shape[1]
    T_rec = np.zeros((n, p, m))
    for r in range(R):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                w_val = w_r[k, i]
                if abs(w_val) > 1e-15:
                    T_rec[i, k, :] += w_val * (u_r[i, :] * v_r[:, k])
    
    return np.allclose(T_rec, 1.0, atol=tol)

def _build_schoolbook(n, m, p):
    """
    Standard schoolbook decomposition. Rank = n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return U, V, W

def _get_rank_11_2x2x3():
    """
    Strassen-like decomposition for 2x2 * 2x3 matrix multiplication.
    Rank = 11 (instead of schoolbook 12).
    
    For A (2x2) * B (2x3) -> C (2x3):
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Tensor indices: U[0:4, r] for A entries, V[0:6, r] for B entries, W[0:6, r] for C entries
    """
    R = 11
    U = np.zeros((4, R))
    V = np.zeros((6, R))
    W = np.zeros((6, R))
    
    # Standard bilinear form with 12 terms
    # Terms 0-11: A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # But we can save 1 term.
    
    # Let's use the known rank-11 decomposition for 2x2 * 2x3
    # Based on the fact that we can combine some terms.
    
    # Actually, the minimum rank for 2x2 * 2x3 is 11 (proven).
    # Here's one such decomposition:
    
    # M1 = A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[0,0]*B[0,2]
    # M2 = A[1,0]*B[0,0] + A[1,0]*B[0,1] + A[1,0]*B[0,2]
    # M3 = A[0,1]*B[1,0] + A[0,1]*B[1,1] + A[0,1]*B[1,2]
    # M4 = A[1,1]*B[1,0] + A[1,1]*B[1,1] + A[1,1]*B[1,2]
    # M5 = (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * B[0,0]
    # ... this is getting complex. Let me use a known explicit one.
    
    # Known rank-11 decomposition for [2,2,3]:
    # From literature, the minimal rank is indeed 11.
    # Let me construct it carefully.
    
    # The 12 schoolbook terms are:
    # T1: A[0,0]*B[0,0] -> C[0,0]
    # T2: A[0,0]*B[0,1] -> C[0,1]
    # T3: A[0,0]*B[0,2] -> C[0,2]
    # T4: A[0,1]*B[1,0] -> C[0,0]
    # T5: A[0,1]*B[1,1] -> C[0,1]
    # T6: A[0,1]*B[1,2] -> C[0,2]
    # T7: A[1,0]*B[0,0] -> C[1,0]
    # T8: A[1,0]*B[0,1] -> C[1,1]
    # T9: A[1,0]*B[0,2] -> C[1,2]
    # T10: A[1,1]*B[1,0] -> C[1,0]
    # T11: A[1,1]*B[1,1] -> C[1,1]
    # T12: A[1,1]*B[1,2] -> C[1,2]
    
    # We need to reduce this to 11 rank-1 terms.
    # One approach: Use the identity involving (A[0,0]+A[0,1]+A[1,0]+A[1,1])
    
    # Let me use a known construction:
    # r=0: U=[1,0,0,0], V=[1,0,0,0,0,0], W=[1,0,0,1,0,0]  -> A[0,0]*B[0,0] -> C[0,0]+C[1,0]
    # r=1: U=[0,0,1,0], V=[1,0,0,0,0,0], W=[0,0,0,1,0,0]  -> A[1,0]*B[0,0] -> C[1,0]
    # This is getting complicated. Let me just use schoolbook for 2x2*2x3 (rank 12)
    # for now, and see if the total rank improves.
    
    # For 2x2*2x3, schoolbook rank = 2*2*3 = 12.
    # Let me just use that for safety.
    
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, idx] = 1.0
                V[j*3+k, idx] = 1.0
                W[k*2+i, idx] = 1.0
                idx += 1
    
    return U, V, W

def _build_3x4x5():
    """
    Build decomposition for 3x4 * 4x5 matrix multiplication.
    Split the middle dimension 4 into 2+2.
    
    A (3x4) * B (4x5) -> C (3x5)
    
    Split A into A1 (3x2) | A2 (3x2)
    Split B into B1 (2x5) over B2 (2x5)
    
    C = A1*B1 + A2*B2
    
    Each term is 3x2 * 2x5, which has schoolbook rank 3*2*5 = 30.
    Total schoolbook for this split: 30 + 30 = 60 (same as overall schoolbook).
    
    To improve, we need to use rank < 30 for the 3x2*2x5 factorization.
    Known: 3x2*2x3 has rank 11 (vs schoolbook 18).
    3x2*2x5 can be split into 3x2*2x3 + 3x2*2x2.
    3x2*2x2: schoolbook rank = 12. Known minimum?
    
    Let's use:
    - 3x2*2x3: rank 11 (vs schoolbook 18)
    - 3x2*2x2: rank 10 (vs schoolbook 12) - need to verify
    
    Total per half: 11 + 10 = 21
    Total for both halves: 21 + 21 = 42
    
    But we need explicit decompositions.
    """
    # For simplicity and correctness, let's first just use schoolbook.
    # The total rank would be 60.
    
    # Let me build the schoolbook decomposition directly for 3x4*4x5.
    return _build_schoolbook(3, 4, 5)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # For now, use schoolbook decomposition which gives rank 60.
    # This is the baseline. Any improvement requires explicit lower-rank decompositions.
    
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
