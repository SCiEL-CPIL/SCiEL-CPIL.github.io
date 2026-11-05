import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    # Deadline guard: 110 seconds
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def _u(i, j, r, val):
        U[i * m + j, r] = val
        
    def _v(j, k, r, val):
        V[j * p + k, r] = val
        
    def _w(k, i, r, val):
        W[k * n + i, r] = val

    # --- Strassen 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}) ---
    
    # S1: (A00 + A11)(B00 + B11) -> C00, C11
    _u(0, 0, 0, 1); _u(1, 1, 0, 1)
    _v(0, 0, 0, 1); _v(1, 1, 0, 1)
    _w(0, 0, 0, 1); _w(1, 1, 0, 1)
    
    # S2: (A10 + A11)B00 -> C10, -C11
    _u(1, 0, 1, 1); _u(1, 1, 1, 1)
    _v(0, 0, 1, 1)
    _w(0, 1, 1, 1); _w(1, 1, 1, -1)
    
    # S3: A00(B01 - B11) -> C01, C11
    _u(0, 0, 2, 1)
    _v(0, 1, 2, 1); _v(1, 1, 2, -1)
    _w(1, 0, 2, 1); _w(1, 1, 2, 1)
    
    # S4: A11(B10 - B00) -> C00, C10
    _u(1, 1, 3, 1)
    _v(1, 0, 3, 1); _v(0, 0, 3, -1)
    _w(0, 0, 3, 1); _w(0, 1, 3, 1)
    
    # S5: (A00 + A01)B11 -> -C00, C01
    _u(0, 0, 4, 1); _u(0, 1, 4, 1)
    _v(1, 1, 4, 1)
    _w(0, 0, 4, -1); _w(1, 0, 4, 1)
    
    # S6: (A10 - A00)(B00 + B01) -> C11
    _u(1, 0, 5, 1); _u(0, 0, 5, -1)
    _v(0, 0, 5, 1); _v(0, 1, 5, 1)
    _w(1, 1, 5, 1)
    
    # S7: (A01 - A11)(B10 + B11) -> C00
    _u(0, 1, 6, 1); _u(1, 1, 6, -1)
    _v(1, 0, 6, 1); _v(1, 1, 6, 1)
    _w(0, 0, 6, 1)

    # --- Column 2 of B (k=2), j in {0,1} ---
    # Direct terms for i in {0,1}, j in {0,1}, k=2
    
    # P1: A00 B02 -> C02 (k=2, i=0 -> W idx 4)
    _u(0, 0, 7, 1); _v(0, 2, 7, 1); _w(2, 0, 7, 1)
    
    # P2: A01 B12 -> C02 (k=2, i=0 -> W idx 4)
    _u(0, 1, 8, 1); _v(1, 2, 8, 1); _w(2, 0, 8, 1)
    
    # P3: A10 B02 -> C12 (k=2, i=1 -> W idx 5)
    _u(1, 0, 9, 1); _v(0, 2, 9, 1); _w(2, 1, 9, 1)
    
    # P4: A11 B12 -> C12 (k=2, i=1 -> W idx 5)
    _u(1, 1, 10, 1); _v(1, 2, 10, 1); _w(2, 1, 10, 1)

    # --- Row 2 of A (j=2), k in {0,1,2} ---
    # Direct terms for i in {0,1}, j=2, k in {0,1,2}
    
    # P5: A02 B20 -> C00 (k=0, i=0 -> W idx 0)
    _u(0, 2, 11, 1); _v(2, 0, 11, 1); _w(0, 0, 11, 1)
    
    # P6: A02 B21 -> C01 (k=1, i=0 -> W idx 2)
    _u(0, 2, 12, 1); _v(2, 1, 12, 1); _w(1, 0, 12, 1)
    
    # P7: A02 B22 -> C02 (k=2, i=0 -> W idx 4)
    _u(0, 2, 13, 1); _v(2, 2, 13, 1); _w(2, 0, 13, 1)
    
    # P8: A12 B20 -> C10 (k=0, i=1 -> W idx 1)
    _u(1, 2, 14, 1); _v(2, 0, 14, 1); _w(0, 1, 14, 1)
    
    # P9: A12 B21 -> C11 (k=1, i=1 -> W idx 3)
    _u(1, 2, 15, 1); _v(2, 1, 15, 1); _w(1, 1, 15, 1)
    
    # P10: A12 B22 -> C12 (k=2, i=1 -> W idx 5)
    _u(1, 2, 16, 1); _v(2, 2, 16, 1); _w(2, 1, 16, 1)

    # Check deadline (should be well within)
    if time.time() > deadline:
        # If we somehow timed out (unlikely), return what we have
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
