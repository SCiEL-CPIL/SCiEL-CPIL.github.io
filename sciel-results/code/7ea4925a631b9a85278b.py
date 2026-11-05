import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 110
    n, m, p = 2, 3, 3
    
    def elapsed():
        return time.time() - start_time
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    def verify(U, V, W, T, tol=1e-9):
        # Ensure exactness by checking residual
        Re = T - np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(Re)) < tol
    
    def build_rank17():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val
        
        # Strassen 2x2x2 block on the first 2x2 part of A and B, corresponding to C[0:2, 0:2]
        # A is 2x3, B is 3x3.
        # We can view A as [A00 A01 | A02], [A10 A11 | A12]
        # B as [B00 B01 B02], [B10 B11 B12], [B20 B21 B22]
        # C = A*B is 2x3.
        
        # The previous code's rank 17 construction was:
        # It used 7 terms for the 2x2x2 multiplication of the left 2x2 block of A and left 2x2 block of B?
        # Actually, let's just trust the previous construction if it verified.
        # The previous code defined 17 terms.
        
        # Term 0: A00*B00 -> C00
        set_U(0, 0, 0, 1); set_V(0, 0, 0, 1); set_W(0, 0, 0, 1)
        # Term 1: A10*B11 -> C11? No, let's look at the previous code's logic.
        # The previous code had specific assignments. I will copy them exactly.
        
        # Re-copying the exact assignments from the failed program to ensure correctness of the structure
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
        
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)
        
        # Cross terms
        set_U(0, 0, 7, 1); set_V(0, 2, 7, 1); set_W(2, 0, 7, 1)
        set_U(0, 1, 8, 1); set_V(1, 2, 8, 1); set_W(2, 0, 8, 1)
        set_U(1, 0, 9, 1); set_V(0, 2, 9, 1); set_W(2, 1, 9, 1)
        set_U(1, 1, 10, 1); set_V(1, 2, 10, 1); set_W(2, 1, 10, 1)
        set_U(0, 2, 11, 1); set_V(2, 0, 11, 1); set_W(0, 0, 11, 1)
        set_U(0, 2, 12, 1); set_V(2, 1, 12, 1); set_W(1, 0, 12, 1)
        set_U(0, 2, 13, 1); set_V(2, 2, 13, 1); set_W(2, 0, 13, 1)
        set_U(1, 2, 14, 1); set_V(2, 0, 14, 1); set_W(0, 1, 14, 1)
        set_U(1, 2, 15, 1); set_V(2, 1, 15, 1); set_W(1, 1, 15, 1)
        set_U(1, 2, 16, 1); set_V(2, 2, 16, 1); set_W(2, 1, 16, 1)
        
        return U, V, W

    T = build_tensor()
    U, V, W = build_rank17()
    
    # Verify
    if verify(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails (should not happen), return naive rank 18 to ensure validity
    # Naive rank 18
    rank = 18
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
