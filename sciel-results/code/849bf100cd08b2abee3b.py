import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if k == j else 0
    Actually T is the 3-tensor where C_{k,i} = sum_j A_{i,j} B_{j,k}
    The tensor T[i, j, k] = 1 if the product A_{i,j} * B_{j,k} contributes to C_{k,i}
    So T[i, j, k] = 1 for all valid i, j, k (since each term A_{i,j}*B_{j,k} goes to C_{k,i})
    Wait, let me reconsider.
    
    The tensor T has shape (n*m, m*p, p*n) where:
    T[a, b, c] = 1 if the product of A entry a and B entry b contributes to C entry c
    
    A index a = i*m + j (i in 0..n-1, j in 0..m-1)
    B index b = j*p + k (j in 0..m-1, k in 0..p-1)
    C index c = k*n + i (k in 0..p-1, i in 0..n-1)
    
    For a given (a, b): a = i*m+j, b = j'*p+k
    This contributes to c = k*n + i if j == j' (the inner dimension matches)
    So T[a, b, c] = 1 if j == j' and c = k*n + i
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    reconstructed = np.zeros_like(T)
    for r in range(rank):
        reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(reconstructed, T, atol=tol)

def _try_rank_search(T, target_rank, time_limit=60.0):
    """Try to find a decomposition of rank target_rank using structured search."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    T = np.array(T, dtype=float)
    
    # Use a structured approach: start with schoolbook and try to eliminate terms
    # by finding linear dependencies
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, idx] = 1
                V_sb[b, idx] = 1
                W_sb[c, idx] = 1
                idx += 1
    
    # Now try to find a lower rank decomposition using a greedy/structured approach
    # The idea: if we can express some of the rank-1 terms as linear combinations
    # of others, we can reduce the rank.
    
    # For now, return the schoolbook as fallback
    return None

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication."""
    # Standard Strassen for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (c+d)e
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (c-a)(e+f)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x1x2():
    """Schoolbook for 2x1 * 1x2 = 2x2. Rank 4."""
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    
    # A = [[a0], [a1]], B = [[b0, b1]], C = [[a0*b0, a0*b1], [a1*b0, a1*b1]]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
    U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    
    return U, V, W

def _schoolbook_2x3x1():
    """Schoolbook for 2x3 * 3x1 = 2x1. Rank 6."""
    U = np.zeros((6, 6))
    V = np.zeros((3, 6))
    W = np.zeros((2, 6))
    
    # A is 2x3: indices (i,j) -> i*3+j
    # B is 3x1: indices (j,0) -> j
    # C is 2x1: indices (0,i) -> i (but with trace convention C[k*n+i] where k=0, so C[i])
    # Actually C has shape (p*n) = (1*2) = 2, indexed by k*n+i = 0*2+i = i
    
    for i in range(2):
        for j in range(3):
            r = i * 3 + j
            U[i * 3 + j, r] = 1
            V[j, r] = 1
            W[i, r] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try to find a rank-16 decomposition first
    # If that fails, use rank-17
    
    # Rank 17 decomposition (known valid)
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Map Strassen indices to global
    # A local: 0->0, 1->1, 2->3, 3->4
    # B local: 0->0, 1->1, 2->3, 3->4
    # C local: 0->0, 1->1, 2->2, 3->3
    
    for r in range(7):
        for a_local in range(4):
            if S_U[a_local, r] != 0:
                a_global = [0, 1, 3, 4][a_local]
                U17[a_global, r] += S_U[a_local, r]
        for b_local in range(4):
            if S_V[b_local, r] != 0:
                b_global = [0, 1, 3, 4][b_local]
                V17[b_global, r] += S_V[b_local, r]
        for c_local in range(4):
            if S_W[c_local, r] != 0:
                c_global = [0, 1, 2, 3][c_local]
                W17[c_global, r] += S_W[c_local, r]
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0, 1, 2, 3
    
    # r=7: A02 B20 -> C00
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    # r=8: A02 B21 -> C01
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    # r=9: A12 B20 -> C10
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    # r=10: A12 B21 -> C11
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    for r_offset, (a_idx, b_idx, c_idx) in enumerate([
        (0, 2, 4), (1, 5, 4), (2, 8, 4),
        (3, 2, 5), (4, 5, 5), (5, 8, 5)
    ]):
        r = 11 + r_offset
        U17[a_idx, r] = 1
        V17[b_idx, r] = 1
        W17[c_idx, r] = 1
    
    # Verify rank 17
    if _verify_decomposition(U17, V17, W17, T):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Fallback: schoolbook rank 18
    U18 = np.zeros((n * m, 18))
    V18 = np.zeros((m * p, 18))
    W18 = np.zeros((p * n, 18))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U18[a, idx] = 1
                V18[b, idx] = 1
                W18[c, idx] = 1
                idx += 1
    
    return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
