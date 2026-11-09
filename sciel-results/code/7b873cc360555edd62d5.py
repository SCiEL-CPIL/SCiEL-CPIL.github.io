import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Block decomposition: 4x4 + 1
    # Rank = 16 (A11 B11) + 16 (A12 B21) + 16 (A11 B12) + 4 (A12 B22)
    #       + 16 (A21 B11) + 4 (A22 B21) + 4 (A21 B12) + 1 (A22 B22)
    # Total = 77
    
    rank = 77
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
        col += 1
        
    # Block 1: i in 0-3, j in 0-3, k in 0-3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term(i, j, k)
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Block 2: i in 0-3, j=4, k in 0-3
    for i in range(4):
        for k in range(4):
            add_term(i, 4, k)
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Block 3: i in 0-3, j in 0-3, k=4
    for i in range(4):
        for j in range(4):
            add_term(i, j, 4)
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Block 4: i in 0-3, j=4, k=4
    for i in range(4):
        add_term(i, 4, 4)
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Block 5: i=4, j in 0-3, k in 0-3
    for j in range(4):
        for k in range(4):
            add_term(4, j, k)
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Block 6: i=4, j=4, k in 0-3
    for k in range(4):
        add_term(4, 4, k)
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Block 7: i=4, j in 0-3, k=4
    for j in range(4):
        add_term(4, j, 4)
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Block 8: i=4, j=4, k=4
    add_term(4, 4, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
