import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin

def _check_deadline():
    return (time.time() - _START_TIME) >= _DEADLINE

def construct():
    n, m, p = 2, 2, 8
    
    # Strassen factors for 2x2 matrix multiplication
    # U: shape (4, 7) - rows correspond to A indices (0:00, 1:01, 2:10, 3:11)
    # V: shape (4, 7) - rows correspond to B indices (0:00, 1:01, 2:10, 3:11)
    # W: shape (4, 7) - rows correspond to C indices (0:00, 1:01, 2:10, 3:11)
    
    # Standard Strassen decomposition
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    U_s = np.array([
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ], dtype=int)
    
    V_s = np.array([
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ], dtype=int)
    
    W_s = np.array([
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ], dtype=int)
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col_offset = 0
    for t in range(num_blocks):
        # U is the same for all blocks, so we just repeat the columns
        U[:, col_offset:col_offset+7] = U_s.T
        
        # V mapping for block t:
        # Local B indices 0,1,2,3 map to:
        # 0: row 0, col 2t -> global idx 2t
        # 1: row 0, col 2t+1 -> global idx 2t+1
        # 2: row 1, col 2t -> global idx 8 + 2t
        # 3: row 1, col 2t+1 -> global idx 8 + 2t + 1
        
        v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
        # V_s is (7, 4) where rows are P terms and cols are local B indices.
        # We need to fill V[global_idx, col] with V_s[term, local_idx]
        # So for each term r, V[global_idx[r], col_offset+r] = V_s[r, local_idx]
        for r in range(7):
            for l in range(4):
                V[v_indices[l], col_offset + r] = V_s[r, l]
                
        # W mapping for block t:
        # Convention W[k*n + i, r] where k is output col, i is output row.
        # Local C indices 0,1,2,3 (corresponding to 00, 01, 10, 11 in 2x2 block)
        # 0: c[0][2t] -> i=0, k=2t -> global idx 2t*2 + 0 = 4t
        # 1: c[0][2t+1] -> i=0, k=2t+1 -> global idx (2t+1)*2 + 0 = 4t+2
        # 2: c[1][2t] -> i=1, k=2t -> global idx 2t*2 + 1 = 4t+1
        # 3: c[1][2t+1] -> i=1, k=2t+1 -> global idx (2t+1)*2 + 1 = 4t+3
        
        w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
        for r in range(7):
            for l in range(4):
                W[w_indices[l], col_offset + r] = W_s[r, l]
                
        col_offset += 7
        
    # Convert to lists of lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
