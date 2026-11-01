import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 2, 6
    rank = 21
    
    # Initialize U, V, W as numpy arrays
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 7 products for 2x2
    # U indices: a00=0, a01=1, a10=2, a11=3
    # V indices for block b: b00=2b, b01=2b+1, b10=6+2b, b11=7+2b
    # W indices for block b: c00=4b, c01=4b+2, c10=4b+1, c11=4b+3
    # Note: W index is k*n + i.
    # c00: k=2b, i=0 -> 2b*2 + 0 = 4b
    # c01: k=2b+1, i=0 -> (2b+1)*2 + 0 = 4b+2
    # c10: k=2b, i=1 -> 2b*2 + 1 = 4b+1
    # c11: k=2b+1, i=1 -> (2b+1)*2 + 1 = 4b+3
    
    strassen_steps = [
        # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        # P1: a00 * (b00 + b11) -> c00 += P1, c11 += P1
        ([0], [1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # P2: (a10 + a11) * b00 -> c10 += P2, c11 -= P2
        ([2, 3], [1, 1], [0], [1], [1, 3], [1, -1]),
        # P3: a00 * (b10 - b11) -> c01 += P3, c11 += P3
        ([0], [1], [2, 3], [1, -1], [2, 3], [1, 1]),
        # P4: a11 * (b00 - b10) -> c00 += P4, c10 += P4
        ([3], [1], [0, 2], [1, -1], [0, 1], [1, 1]),
        # P5: (a00 + a01) * b11 -> c00 -= P5, c01 += P5
        ([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]),
        # P6: (a10 - a00) * (b00 + b01) -> c11 += P6
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # P7: (a01 - a11) * (b10 + b11) -> c00 += P7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_steps[s]
            
            # Map local indices to global indices
            # U is global, no offset
            for i, val in zip(u_idx, u_val):
                U[i, col] += val
            
            # V offset: local 0,1,2,3 -> 2b, 2b+1, 6+2b, 7+2b
            v_map = {0: 2*b, 1: 2*b+1, 2: 6+2*b, 3: 7+2*b}
            for i, val in zip(v_idx, v_val):
                V[v_map[i], col] += val
            
            # W offset: local 0,1,2,3 -> 4b, 4b+1, 4b+2, 4b+3
            w_map = {0: 4*b, 1: 4*b+1, 2: 4*b+2, 3: 4*b+3}
            for i, val in zip(w_idx, w_val):
                W[w_map[i], col] += val
                
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
