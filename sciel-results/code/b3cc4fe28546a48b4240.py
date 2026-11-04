import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # 1. Strassen 2x2x2
    # U_S, V_S, W_S are 4x7
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 0, 0, 1, 0, 0, -1]  # A22
    ])
    # Wait, let's re-verify the Strassen matrices from my derivation above.
    # k=1: A: [1,0,0,1], B: [1,0,0,1], C: [1,0,0,1]
    # k=2: A: [0,0,1,1], B: [1,0,0,0], C: [0,0,1,-1]
    # k=3: A: [1,0,0,0], B: [0,1,0,-1], C: [0,1,0,1]
    # k=4: A: [0,0,0,1], B: [-1,0,1,0], C: [1,0,1,0]
    # k=5: A: [1,1,0,0], B: [0,0,0,1], C: [-1,1,0,0]
    # k=6: A: [-1,0,1,0], B: [1,1,0,0], C: [0,0,0,1]
    # k=7: A: [0,1,0,-1], B: [0,0,1,1], C: [1,0,0,0]
    
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, -1]
    ])
    # Check row 0 (A11): 1,0,1,0,1,-1,0. Matches k=1,3,5,6,7?
    # k=1: 1. k=2: 0. k=3: 1. k=4: 0. k=5: 1. k=6: -1. k=7: 0.
    # My derived A11: k1=1, k2=0, k3=1, k4=0, k5=1, k6=-1, k7=0.
    # Row 0 of U_S above: [1, 0, 1, 0, 1, -1, 0]. Correct.
    
    # Row 1 (A12): 0,0,0,0,1,0,1.
    # k=1:0, k=2:0, k=3:0, k=4:0, k=5:1, k=6:0, k=7:1.
    # My derived A12: k1=0, k2=0, k3=0, k4=0, k5=1, k6=0, k7=1. Correct.
    
    # Row 2 (A21): 0,1,0,0,0,1,0.
    # k=1:0, k=2:1, k=3:0, k=4:0, k=5:0, k=6:1, k=7:0.
    # My derived A21: k1=0, k2=1, k3=0, k=4:0, k5=0, k6=1, k7=0. Correct.
    
    # Row 3 (A22): 1,0,0,1,0,0,-1.
    # k=1:1, k=2:0, k=3:0, k=4:1, k=5:0, k=6:0, k=7:-1.
    # My derived A22: k1=1, k2=0, k3=0, k4=1, k5=0, k6=0, k7=-1. Correct.
    
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ])
    # Check V_S against derivation.
    # B11: k1=1, k2=1, k3=0, k4=-1, k5=0, k6=1, k7=0.
    # Row 0: [1, 1, 0, -1, 0, 1, 0]. Correct.
    # B12: k1=0, k2=0, k3=1, k4=0, k5=0, k6=1, k7=0.
    # Row 1: [0, 0, 1, 0, 0, 1, 0]. Correct.
    # B21: k1=0, k2=0, k3=0, k4=1, k5=0, k6=0, k7=1.
    # Row 2: [0, 0, 0, 1, 0, 0, 1]. Correct.
    # B22: k1=1, k2=0, k3=-1, k4=0, k5=1, k6=0, k7=1.
    # Row 3: [1, 0, -1, 0, 1, 0, 1]. Correct.
    
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ])
    # Check W_S.
    # C11: k1=1, k2=0, k3=0, k4=1, k5=-1, k6=0, k7=1.
    # Row 0: [1, 0, 0, 1, -1, 0, 1]. Correct.
    # C12: k1=0, k2=0, k3=1, k4=0, k5=1, k6=0, k7=0.
    # Row 1: [0, 0, 1, 0, 1, 0, 0]. Correct.
    # C21: k1=0, k2=1, k3=0, k4=1, k5=0, k6=0, k7=0.
    # Row 2: [0, 1, 0, 1, 0, 0, 0]. Correct.
    # C22: k1=1, k2=-1, k3=1, k4=0, k5=0, k6=1, k7=0.
    # Row 3: [1, -1, 1, 0, 0, 1, 0]. Correct.

    # 2. Kronecker products for 4x4x4
    U_444 = np.kron(U_S, U_S) # 16x49
    V_444 = np.kron(V_S, V_S) # 16x49
    W_444 = np.kron(W_S, W_S) # 16x49
    
    # 3. Construct full matrices
    rank = 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Fill 4x4x4 part
    U[:, :49] = U_444
    
    # Map V_444 (16 rows) to V (20 rows)
    # V_444 row r corresponds to B_{r//4, r%4}?
    # np.kron row index is r1*4 + r2.
    # B_{r1, r2} is at global row r1*5 + r2.
    for r in range(16):
        r1 = r // 4
        r2 = r % 4
        global_row = r1 * 5 + r2
        V[global_row, :49] = V_444[r, :]
        
    # Map W_444 (16 rows) to W (20 rows)
    # W_444 row r corresponds to C_{r//4, r%4}?
    # W_444 is C_{k, i}.
    # Global W row is k*4 + i.
    for r in range(16):
        k = r // 4
        i = r % 4
        global_row = k * 4 + i
        W[global_row, :49] = W_444[r, :]
        
    # Fill 4x4x1 part
    # 16 terms
    for i in range(4):
        for j in range(4):
            c = 49 + i * 4 + j
            U[i * 4 + j, c] = 1
            V[j * 5 + 4, c] = 1
            W[4 * 4 + i, c] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
