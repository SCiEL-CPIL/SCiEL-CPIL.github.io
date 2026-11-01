import time
import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(10)]
    W = [[0.0] * rank for _ in range(10)]
    
    c = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal c
        for idx, val in u_coeffs.items():
            U[idx][c] += val
        for idx, val in v_coeffs.items():
            V[idx][c] += val
        for idx, val in w_coeffs.items():
            W[idx][c] += val
        c += 1
    
    # Strassen block 1: columns 0-1
    # M1 = (A00+A11)(B00+B11)
    add_term({0:1, 3:1}, {0:1, 6:1}, {0:1, 3:1})
    # M2 = (A10+A11)B00
    add_term({2:1, 3:1}, {0:1}, {1:1, 3:1})
    # M3 = A00(B01-B11)
    add_term({0:1}, {1:1, 6:-1}, {2:1, 3:-1})
    # M4 = A11(B10-B00)
    add_term({3:1}, {5:1, 0:-1}, {0:-1, 1:1})
    # M5 = (A00+A01)B11
    add_term({0:1, 1:1}, {6:1}, {2:1, 3:-1})
    # M6 = (A10-A00)(B00+B01)
    add_term({2:1, 0:-1}, {0:1, 1:1}, {0:-1, 1:-1, 2:1, 3:1})
    # M7 = (A01-A11)(B10+B11)
    add_term({1:1, 3:-1}, {5:1, 6:1}, {0:1, 1:1, 2:-1, 3:-1})
    
    # Strassen block 2: columns 2-3
    add_term({0:1, 3:1}, {2:1, 8:1}, {4:1, 7:1})
    add_term({2:1, 3:1}, {2:1}, {5:1, 7:1})
    add_term({0:1}, {3:1, 8:-1}, {6:1, 7:-1})
    add_term({3:1}, {7:1, 2:-1}, {4:-1, 5:1})
    add_term({0:1, 1:1}, {8:1}, {6:1, 7:-1})
    add_term({2:1, 0:-1}, {2:1, 3:1}, {4:-1, 5:-1, 6:1, 7:1})
    add_term({1:1, 3:-1}, {7:1, 8:1}, {4:1, 5:1, 6:-1, 7:-1})
    
    # Column 4: 4 separate products
    add_term({0:1}, {4:1}, {8:1})
    add_term({1:1}, {9:1}, {8:1})
    add_term({2:1}, {4:1}, {9:1})
    add_term({3:1}, {9:1}, {9:1})
    
    assert c == rank, f"Expected {rank}, got {c}"
    
    return {"U": U, "V": V, "W": W}
