import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2
    # U: A factors
    # V: B factors
    # W: C factors
    # Indices for A: 00, 01, 10, 11
    # Indices for B: 00, 01, 10, 11
    # Indices for C: 00, 01, 10, 11
    
    # Strassen intermediates:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U rows (A indices: 0=A00, 1=A01, 2=A10, 3=A11)
    # V rows (B indices: 0=B00, 1=B01, 2=B10, 3=B11)
    # W rows (C indices: 0=C00, 1=C10, 2=C01, 3=C11) 
    # Note: The prompt's W index is k*n+i.
    # For a 2x2 block, k is local col, i is local row.
    # Local C indices:
    # C00 (i=0, k=0) -> local w=0
    # C10 (i=1, k=0) -> local w=1
    # C01 (i=0, k=1) -> local w=2
    # C11 (i=1, k=1) -> local w=3
    
    strassen_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11
        [0, 1, 0, 1],  # M3 -> C01, C11
        [1, 0, 1, 0],  # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00, C01
        [0, 0, 0, 1],  # M6 -> C11
        [1, 0, 0, 0]   # M7 -> C00
    ]
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for l in range(8):
        for s in range(7):
            col = 7 * l + s
            
            # U: same for all blocks
            U[:, col] = strassen_U[s]
            
            # V: local B indices mapped to global
            # B local indices: 0, 1, 2, 3
            # Global B indices for block l (cols 2l, 2l+1):
            # B00 (j=0, k=2l) -> 0*16 + 2l = 2l
            # B01 (j=0, k=2l+1) -> 0*16 + 2l+1 = 2l+1
            # B10 (j=1, k=2l) -> 1*16 + 2l = 16+2l
            # B11 (j=1, k=2l+1) -> 1*16 + 2l+1 = 16+2l+1
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            V[b_rows, col] = strassen_V[s]
            
            # W: local C indices mapped to global
            # C local indices: 0, 1, 2, 3
            # Global C indices for block l:
            # C00 (i=0, k=2l) -> 2l*2 + 0 = 4l
            # C10 (i=1, k=2l) -> 2l*2 + 1 = 4l+1
            # C01 (i=0, k=2l+1) -> (2l+1)*2 + 0 = 4l+2
            # C11 (i=1, k=2l+1) -> (2l+1)*2 + 1 = 4l+3
            c_rows = [4*l, 4*l+1, 4*l+2, 4*l+3]
            W[c_rows, col] = strassen_W[s]
                
    # Verify using vectorized operations
    # Check valid terms
    is_valid = True
    # We can verify a sample or all terms. All terms is fast enough.
    # T_{u,v,w} should be 1 if u=2i+j, v=16j+k, w=2k+i
    # Else 0.
    
    # Efficient verification:
    # Construct the tensor implied by U, V, W
    # T = U @ (V * W) ? No.
    # T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
    
    # We can check specific entries.
    for i in range(n):
        for j in range(m):
            u = i * m + j
            for k in range(p):
                v = j * p + k
                w = k * n + i
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                if abs(val - 1.0) > 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if is_valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
