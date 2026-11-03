import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We'll build a decomposition using Strassen for the 3x3x3 part
    # and schoolbook for the 3x3x4 part, but with some sharing.
    
    # Collect all (U, V, W) triplets for each multiplication
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a term: u_val * U[u_idx] * v_val * V[v_idx] contributes w_val to W[w_idx]"""
        U_cols.append((u_idx, u_val))
        V_cols.append((v_idx, v_val))
        W_cols.append((w_idx, w_val))
    
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 3x3, we partition as:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # B = [[B00, B01, B02, B03, B04, B05, B06],
    #      [B10, B11, B12, B13, B14, B15, B16],
    #      [B20, B21, B22, B23, B24, B25, B26]]
    
    # Use Strassen for the 2x2 block (rows 0,1 of A; cols 0,1 of B) for k in {0,1,2}
    # This handles C[0:2, 0:2] contributions from j in {0,1}
    
    # M1 = (A00+A11) * (B00+B11) for each k in {0,1,2}
    for k in range(3):
        add_term(0*m+0, 1, 0*p+k, 1, k*n+0, 1)  # A00 * B0k
        add_term(1*m+1, 1, 0*p+k, 1, k*n+0, 1)  # A11 * B0k
        add_term(0*m+0, 1, 1*p+k, 1, k*n+0, 1)  # A00 * B1k
        add_term(1*m+1, 1, 1*p+k, 1, k*n+0, 1)  # A11 * B1k
        # But Strassen combines these! Let me redo this properly.
    
    # Let me restart with a cleaner approach.
    # I'll explicitly list each multiplication term.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_mult(a_i, a_j, b_j, b_k, c_i, c_k, sign=1):
        """Add term: sign * A[a_i, a_j] * B[b_j, b_k] contributes to C[c_i, c_k]"""
        U_cols.append((a_i * m + a_j, sign))
        V_cols.append((b_j * p + b_k, 1))
        W_cols.append((c_k * n + c_i, 1))
    
    # Strassen for 2x2 (i,k in {0,1}, j in {0,1}), for each k_block in {0,1,2}
    # We'll handle k=0,1,2 separately for now
    
    # Actually, let me just do a direct but optimized decomposition.
    # For 3x3x3, use 26 multiplications (Strassen + direct for remaining)
    # For 3x3x4, use 36 multiplications (schoolbook)
    # Total: 62
    
    # 3x3x3 part: k in {0,1,2}
    # Strassen for the 2x2 subproblem (i in {0,1}, k in {0,1}, j in {0,1})
    
    # M1 = (A00+A11)(B00+B11)
    for k in range(3):
        if k < 2:  # Strassen handles k=0,1
            pass
    
    # This is getting complicated. Let me just use a straightforward approach
    # that's guaranteed to be correct, even if not optimal.
    
    # Schoolbook for everything: rank 63
    # But let me try to implement Strassen properly.
    
    # Clean restart: implement Strassen for 3x3x3 explicitly
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen multiplications for the 2x2 block (i,k in {0,1}), for k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    # For k=0: contributes to C00, C11
    # For k=1: contributes to C01, C10 (via different combinations)
    
    # Let me just do schoolbook but verify it's correct, then try to improve.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_cols.append((i * m + j, 1))
                V_cols.append((j * p + k, 1))
                W_cols.append((k * n + i, 1))
    
    rank = len(U_cols)
    
    # Build the matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        u_idx, u_val = U_cols[r]
        v_idx, v_val = V_cols[r]
        w_idx, w_val = W_cols[r]
        U[u_idx, r] = u_val
        V[v_idx, r] = v_val
        W[w_idx, r] = w_val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
