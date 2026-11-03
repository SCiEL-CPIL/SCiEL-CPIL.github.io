import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 9 decomposition for 4x5x5
    # Split A (4x5) into A1 (4x4) and A2 (4x1)
    # Split B (5x5) into B1 (5x4) and B2 (5x1)
    # C = A1 B1 + A2 B2
    # Split B1 into B1a (4x4) and B1b (4x1)
    # C = A1 B1a + A1 B1b + A2 B2
    
    # 1. A1 B1a: 4x4x4, Rank 7 (Strassen)
    # 2. A1 B1b: 4x4x1, Rank 1
    # 3. A2 B2: 4x1x5, Rank 1
    
    rank = 9
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # U: i*m + j (i in 0..3, j in 0..4)
    # V: j*p + k (j in 0..4, k in 0..4)
    # W: k*n + i (k in 0..4, i in 0..3)
    
    # --- Part 1: Strassen for A1 (4x4) * B1a (4x4) ---
    # A1 indices: i in 0..3, j in 0..3
    # B1a indices: j in 0..3, k in 0..3
    # C1 indices: i in 0..3, k in 0..3
    
    # Strassen factors for 4x4x4
    # M0 = (A11+A22)(B11+B22)
    # M1 = (A21+A22)B11
    # M2 = A11(B12-B22)
    # M3 = A22(B21-B11)
    # M4 = (A11+A12)B22
    # M5 = (A21-A11)(B11+B12)
    # M6 = (A12-A22)(B21+B22)
    
    # We define the 7 terms
    
    # Term 0: M0
    # U: A11+A22 -> i,j in {0,1}x{0,1} U {2,3}x{2,3}
    # V: B11+B22 -> j,k in {0,1}x{0,1} U {2,3}x{2,3}
    # W: C11+C22 -> k,i in {0,1}x{0,1} U {2,3}x{2,3}
    
    for i in range(4):
        for j in range(4):
            if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                U[i * m + j, 0] = 1.0
    for j in range(4):
        for k in range(4):
            if (j < 2 and k < 2) or (j >= 2 and k >= 2):
                V[j * p + k, 0] = 1.0
    for k in range(4):
        for i in range(4):
            if (k < 2 and i < 2) or (k >= 2 and i >= 2):
                W[k * n + i, 0] = 1.0
                
    # Term 1: M1 = (A21+A22)B11
    # U: A21+A22 -> i in {2,3}, j in {0,1} U {2,3} => i in {2,3}, all j
    # V: B11 -> j in {0,1}, k in {0,1}
    # W: C21 -> i in {2,3}, k in {0,1}
    
    for i in range(4):
        for j in range(4):
            if i >= 2:
                U[i * m + j, 1] = 1.0
    for j in range(4):
        for k in range(4):
            if j < 2 and k < 2:
                V[j * p + k, 1] = 1.0
    for k in range(4):
        for i in range(4):
            if k < 2 and i >= 2:
                W[k * n + i, 1] = 1.0

    # Term 2: M2 = A11(B12-B22)
    # U: A11 -> i in {0,1}, j in {0,1}
    # V: B12-B22 -> j in {0,1}, k in {2,3} (1) ; j in {2,3}, k in {2,3} (-1)
    # W: C12 -> i in {0,1}, k in {2,3}
    
    for i in range(4):
        for j in range(4):
            if i < 2 and j < 2:
                U[i * m + j, 2] = 1.0
    for j in range(4):
        for k in range(4):
            if j < 2 and k >= 2:
                V[j * p + k, 2] = 1.0
            elif j >= 2 and k >= 2:
                V[j * p + k, 2] = -1.0
    for k in range(4):
        for i in range(4):
            if k >= 2 and i < 2:
                W[k * n + i, 2] = 1.0

    # Term 3: M3 = A22(B21-B11)
    # U: A22 -> i in {2,3}, j in {2,3}
    # V: B21-B11 -> j in {2,3}, k in {0,1} (1) ; j in {0,1}, k in {0,1} (-1)
    # W: C21 -> i in {2,3}, k in {0,1}
    
    for i in range(4):
        for j in range(4):
            if i >= 2 and j >= 2:
                U[i * m + j, 3] = 1.0
    for j in range(4):
        for k in range(4):
            if j >= 2 and k < 2:
                V[j * p + k, 3] = 1.0
            elif j < 2 and k < 2:
                V[j * p + k, 3] = -1.0
    for k in range(4):
        for i in range(4):
            if k < 2 and i >= 2:
                W[k * n + i, 3] = 1.0

    # Term 4: M4 = (A11+A12)B22
    # U: A11+A12 -> i in {0,1}, j in {0,1} U {2,3} => i in {0,1}, all j
    # V: B22 -> j in {2,3}, k in {2,3}
    # W: C12 -> i in {0,1}, k in {2,3}
    
    for i in range(4):
        for j in range(4):
            if i < 2:
                U[i * m + j, 4] = 1.0
    for j in range(4):
        for k in range(4):
            if j >= 2 and k >= 2:
                V[j * p + k, 4] = 1.0
    for k in range(4):
        for i in range(4):
            if k >= 2 and i < 2:
                W[k * n + i, 4] = 1.0

    # Term 5: M5 = (A21-A11)(B11+B12)
    # U: A21-A11 -> i in {2,3}, j in {0,1} (1) ; i in {0,1}, j in {0,1} (-1)
    # V: B11+B12 -> j in {0,1}, k in {0,1} U {2,3} => j in {0,1}, all k
    # W: C22 -> i in {2,3}, k in {2,3}
    
    for i in range(4):
        for j in range(4):
            if i >= 2 and j < 2:
                U[i * m + j, 5] = 1.0
            elif i < 2 and j < 2:
                U[i * m + j, 5] = -1.0
    for j in range(4):
        for k in range(4):
            if j < 2:
                V[j * p + k, 5] = 1.0
    for k in range(4):
        for i in range(4):
            if k >= 2 and i >= 2:
                W[k * n + i, 5] = 1.0

    # Term 6: M6 = (A12-A22)(B21+B22)
    # U: A12-A22 -> i in {0,1}, j in {2,3} (1) ; i in {2,3}, j in {2,3} (-1)
    # V: B21+B22 -> j in {2,3}, k in {0,1} U {2,3} => j in {2,3}, all k
    # W: C11 -> i in {0,1}, k in {0,1}
    
    for i in range(4):
        for j in range(4):
            if i < 2 and j >= 2:
                U[i * m + j, 6] = 1.0
            elif i >= 2 and j >= 2:
                U[i * m + j, 6] = -1.0
    for j in range(4):
        for k in range(4):
            if j >= 2:
                V[j * p + k, 6] = 1.0
    for k in range(4):
        for i in range(4):
            if k < 2 and i < 2:
                W[k * n + i, 6] = 1.0

    # --- Part 2: A1 B1b (4x4x1) ---
    # A1: i in 0..3, j in 0..3
    # B1b: j in 0..3, k=4
    # C: i in 0..3, k=4
    # Rank 1
    
    for i in range(4):
        for j in range(4):
            U[i * m + j, 7] = 1.0
    for j in range(4):
        V[j * p + 4, 7] = 1.0
    for i in range(4):
        W[4 * n + i, 7] = 1.0

    # --- Part 3: A2 B2 (4x1x5) ---
    # A2: i in 0..3, j=4
    # B2: j=4, k in 0..4
    # C: i in 0..3, k in 0..4
    # Rank 1
    
    for i in range(4):
        U[i * m + 4, 8] = 1.0
    for k in range(5):
        V[4 * p + k, 8] = 1.0
    for k in range(5):
        for i in range(4):
            W[k * n + i, 8] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
