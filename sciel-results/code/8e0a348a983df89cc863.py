"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_t0 = time.time()
_DEADLINE = _t0 + 130.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank 53 decomposition:
    # 7 Strassen blocks (rank 7 each) for 14 columns
    # 1 trivial block (rank 4) for the last column
    # Total rank = 7*7 + 4 = 53
    
    rank = 53
    
    # U: (n*m, rank) = (4, 53)
    # V: (m*p, rank) = (30, 53)
    # W: (p*n, rank) = (30, 53)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 factors
    # Indices for U: 0=A00, 1=A01, 2=A10, 3=A11
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    # Define the 7 Strassen multiplications
    # P1 = (A00+A11)(B0k0+B1k1)
    # P2 = (A10+A11)B0k0
    # P3 = A00(B0k1-B1k1)
    # P4 = A11(B1k0-B0k0)
    # P5 = (A00+A01)B1k1
    # P6 = (A10-A00)(B0k0+B0k1)
    # P7 = (A01-A11)(B1k0+B1k1)
    
    # Reconstruction:
    # C0k0 = P1 - P3 - P4 - P6 + P7  (Wait, I verified this is wrong in the previous thought process)
    # Let's use the verified coefficients from the working code provided in the prompt.
    
    # The prompt's code uses:
    # u_entries, v_entries, w_entries
    # I will copy that logic but implement it cleanly with numpy arrays.
    
    # u_entries[t] is a list of (row, coeff)
    # v_entries[t] is a list of (code, coeff)
    # w_entries[t] is a list of (code, coeff)
    
    u_entries = [
        [(0, 1), (3, 1)],   # P1
        [(2, 1), (3, 1)],   # P2
        [(0, 1)],           # P3
        [(3, 1)],           # P4
        [(0, 1), (1, 1)],   # P5
        [(2, 1), (0, -1)],  # P6
        [(1, 1), (3, -1)],  # P7
    ]
    v_entries = [
        [(('B0k0', 1)), (('B1k1', 1))],  # P1
        [(('B0k0', 1))],                  # P2
        [(('B0k1', 1)), (('B1k1', -1))],  # P3
        [(('B1k0', 1)), (('B0k0', -1))],  # P4
        [(('B1k1', 1))],                  # P5
        [(('B0k0', 1)), (('B0k1', 1))],   # P6
        [(('B1k0', 1)), (('B1k1', 1))],   # P7
    ]
    w_entries = [
        [(('C0k0', 1)), (('C1k1', 1))],  # P1
        [(('C1k0', 1)), (('C1k1', -1))], # P2
        [(('C0k1', 1)), (('C1k1', 1))],  # P3
        [(('C0k0', 1)), (('C1k0', 1))],  # P4
        [(('C0k0', -1)), (('C0k1', 1))], # P5
        [(('C1k1', 1))],                 # P6
        [(('C0k0', 1))],                 # P7
    ]
    
    def get_v_row(code, k0, k1):
        # code is like 'B0k0'
        j = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return j * p + k
        
    def get_w_row(code, k0, k1):
        # code is like 'C0k0'
        i = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return k * n + i

    # Place 7 Strassen blocks for columns (0,1), (2,3), ..., (12,13)
    for blk in range(7):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        base = 7 * blk
        for t in range(7):
            r = base + t
            # U
            for (u_row_i, c) in u_entries[t]:
                U[u_row_i, r] += c
            # V
            for (code, c) in v_entries[t]:
                v_row_idx = get_v_row(code, k0, k1)
                V[v_row_idx, r] += c
            # W
            for (code, c) in w_entries[t]:
                w_row_idx = get_w_row(code, k0, k1)
                W[w_row_idx, r] += c
                
    # Place the residual 2x2x1 block for column 14 (rank 4, trivial)
    # C[0,14] = A[0,0]B[0,14] + A[0,1]B[1,14]
    # C[1,14] = A[1,0]B[0,14] + A[1,1]B[1,14]
    b14 = 49
    # Term 1: A00 * B0,14 -> C0,14
    U[0, b14 + 0] = 1
    V[0 * p + 14, b14 + 0] = 1
    W[14 * n + 0, b14 + 0] = 1
    
    # Term 2: A01 * B1,14 -> C0,14
    U[1, b14 + 1] = 1
    V[1 * p + 14, b14 + 1] = 1
    W[14 * n + 0, b14 + 1] = 1
    
    # Term 3: A10 * B0,14 -> C1,14
    U[2, b14 + 2] = 1
    V[0 * p + 14, b14 + 2] = 1
    W[14 * n + 1, b14 + 2] = 1
    
    # Term 4: A11 * B1,14 -> C1,14
    U[3, b14 + 3] = 1
    V[1 * p + 14, b14 + 3] = 1
    W[14 * n + 1, b14 + 3] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
