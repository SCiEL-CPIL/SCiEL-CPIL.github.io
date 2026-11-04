"""Rewritten program: deterministic rank-17 tensor decomposition for 2x3x3 matrix multiplication."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
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
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=tol)

def _rank17_decomposition():
    """
    Deterministic rank-17 decomposition using Strassen for the 2x2x2 core block
    and direct multiplications for the remaining parts.
    
    A is 2x3, B is 3x3, C = A @ B is 2x3.
    
    Indexing:
      a = i*3 + j  (i in 0..1, j in 0..2) -> 6 entries
      b = j*3 + k  (j in 0..2, k in 0..2) -> 9 entries
      c = k*2 + i  (k in 0..2, i in 0..1) -> 6 entries
    
    Block structure:
      - j in {0,1}, k in {0,1}: 2x2x2 core -> Strassen (7 terms)
      - j in {0,1}, k=2: 2x2x1 -> direct (4 terms)
      - j=2, k in {0,1,2}: 2x1x3 -> direct (6 terms)
    Total: 7 + 4 + 6 = 17
    """
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = n * m, m * p, p * n  # 6, 9, 6
    rank = 17
    
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_U(i, j, r, val):
        U[i * m + j, r] = val
        
    def set_V(j, k, r, val):
        V[j * p + k, r] = val
        
    def set_W(k, i, r, val):
        W[k * n + i, r] = val

    # === Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1}) ===
    # A1 = [[a0, a1], [a3, a4]] where a_{i,j} = A[i,j]
    # B1 = [[b0, b1], [b3, b4]] where b_{j,k} = B[j,k]
    # C1 = [[c0, c2], [c1, c3]] where c_{i,k} = C[i,k] (c index = k*2+i)
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    # M3: A00(B01-B11) -> C01 += 1, C11 += 1
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
    
    # M4: A11(B10-B00) -> C00 += 1, C10 += 1
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
    
    # M6: (A10-A00)(B00+B01) -> C11 += 1
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    # M7: (A01-A11)(B10+B11) -> C00 += 1
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)

    # === Block 2: Direct 2x2x1 (j in {0,1}, k=2) ===
    # C[i,2] += A[i,0]*B[0,2] + A[i,1]*B[1,2]
    
    # Term 7: A00 * B02 -> C02
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    # Term 8: A01 * B12 -> C02
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    # Term 9: A10 * B02 -> C12
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    # Term 10: A11 * B12 -> C12
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)

    # === Block 3: Direct 2x1x3 (j=2, k in {0,1,2}) ===
    # C[i,k] += A[i,2]*B[2,k]
    
    # Term 11: A02 * B20 -> C00
    set_U(0, 2, 11, 1)
    set_V(2, 0, 11, 1)
    set_W(0, 0, 11, 1)
    
    # Term 12: A02 * B21 -> C01
    set_U(0, 2, 12, 1)
    set_V(2, 1, 12, 1)
    set_W(1, 0, 12, 1)
    
    # Term 13: A02 * B22 -> C02
    set_U(0, 2, 13, 1)
    set_V(2, 2, 13, 1)
    set_W(2, 0, 13, 1)
    
    # Term 14: A12 * B20 -> C10
    set_U(1, 2, 14, 1)
    set_V(2, 0, 14, 1)
    set_W(0, 1, 14, 1)
    
    # Term 15: A12 * B21 -> C11
    set_U(1, 2, 15, 1)
    set_V(2, 1, 15, 1)
    set_W(1, 1, 15, 1)
    
    # Term 16: A12 * B22 -> C12
    set_U(1, 2, 16, 1)
    set_V(2, 2, 16, 1)
    set_W(2, 1, 16, 1)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n * m, m * p, p * n  # 6, 9, 6
    
    # Build the rank-17 decomposition
    U, V, W = _rank17_decomposition()
    
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook rank-18
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
