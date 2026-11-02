"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                r = round(v * 2) / 2
                if abs(r - round(r)) < 1e-9:
                    new_row.append(int(round(r)))
                else:
                    new_row.append(r)
            result.append(new_row)
        return result
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        U_t = np.array(U, dtype=float).reshape(n, m, -1)
        V_t = np.array(V, dtype=float).reshape(m, p, -1)
        W_t = np.array(W, dtype=float).reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True
    
    def snap_to_half_integers(U, V, W):
        U_s = np.round(np.array(U, dtype=float) * 2) / 2
        V_s = np.round(np.array(V, dtype=float) * 2) / 2
        W_s = np.round(np.array(W, dtype=float) * 2) / 2
        return U_s, V_s, W_s
    
    def compute_error(U, V, W):
        n, m, p = DIMENSIONS
        U_t = np.array(U, dtype=float).reshape(n, m, -1)
        V_t = np.array(V, dtype=float).reshape(m, p, -1)
        W_t = np.array(W, dtype=float).reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.sum((T - 1.0)**2)
    
    def strassen_83():
        """Construct rank-83 decomposition using Strassen for the 2x2x2 sub-block."""
        R = 83
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        # Strassen terms for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[0*m + 0, 0] = 1
        U[1*m + 1, 0] = 1
        V[0*p + 0, 0] = 1
        V[1*p + 1, 0] = 1
        W[0*n + 0, 0] = 1
        W[1*n + 1, 0] = 1
        
        # Term 2: M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
        U[1*m + 0, 1] = 1
        U[1*m + 1, 1] = 1
        V[0*p + 0, 1] = 1
        W[0*n + 1, 1] = 1
        W[1*n + 1, 1] = 1
        
        # Term 3: M3 = A00(B10-B11) -> C01 += M3, C11 -= M3
        U[0*m + 0, 2] = 1
        V[1*p + 0, 2] = 1
        V[1*p + 1, 2] = -1
        W[1*n + 0, 2] = 1
        W[1*n + 1, 2] = -1
        
        # Term 4: M4 = A11(B01-B00) -> C00 -= M4, C01 += M4
        U[1*m + 1, 3] = 1
        V[0*p + 1, 3] = 1
        V[0*p + 0, 3] = -1
        W[0*n + 0, 3] = -1
        W[1*n + 0, 3] = 1
        
        # Term 5: M5 = (A00+A10)B11 -> C00 += M5, C10 += M5
        U[0*m + 0, 4] = 1
        U[1*m + 0, 4] = 1
        V[1*p + 1, 4] = 1
        W[0*n + 0, 4] = 1
        W[0*n + 1, 4] = 1
        
        # Term 6: M6 = (A01-A00)(B00+B10) -> C11 -= M6
        U[0*m + 1, 5] = 1
        U[0*m + 0, 5] = -1
        V[0*p + 0, 5] = 1
        V[1*p + 0, 5] = 1
        W[1*n + 1, 5] = -1
        
        # Term 7: M7 = (A10-A11)(B01+B11) -> C00 += M7
        U[1*m + 0, 6] = 1
        U[1*m + 1, 6] = -1
        V[0*p + 1, 6] = 1
        V[1*p + 1, 6] = 1
        W[0*n + 0, 6] = 1
        
        # Fill remaining with schoolbook terms for the complement
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        if col != R:
            return None
        return U, V, W
    
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U[i * m + j, idx] = 1
            V[j * p + k, idx] = 1
            W[k * n + i, idx] = 1
        return U, V, W
    
    # Try Strassen-83 first
    result = strassen_83()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
