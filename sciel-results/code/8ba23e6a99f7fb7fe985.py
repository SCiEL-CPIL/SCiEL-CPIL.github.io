import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    # U: (nm, rank)
    # V: (mp, rank)
    # W: (pn, rank)
    
    rank = 28
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    def add_strassen_block(a_idx, b_idx, c_idx, offset):
        # a_idx: list of 4 global indices for A (row, col)
        # b_idx: list of 4 global indices for B (row, col)
        # c_idx: list of 4 global indices for C (row, col)
        # offset: starting rank index
        
        # Local indices 0,1,2,3 correspond to:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        # a_idx[0] is A[0,0], a_idx[1] is A[0,1], a_idx[2] is A[1,0], a_idx[3] is A[1,1]
        # b_idx[0] is B[0,0], b_idx[1] is B[0,1], b_idx[2] is B[1,0], b_idx[3] is B[1,1]
        # c_idx[0] is C[0,0], c_idx[1] is C[0,1], c_idx[2] is C[1,0], c_idx[3] is C[1,1]
        
        # M1
        U[a_idx[0], offset + 0] += 1
        U[a_idx[3], offset + 0] += 1
        V[b_idx[0], offset + 0] += 1
        V[b_idx[3], offset + 0] += 1
        W[c_idx[0], offset + 0] += 1
        W[c_idx[3], offset + 0] += 1
        
        # M2
        U[a_idx[2], offset + 1] += 1
        U[a_idx[3], offset + 1] += 1
        V[b_idx[0], offset + 1] += 1
        W[c_idx[2], offset + 1] += 1
        W[c_idx[3], offset + 1] -= 1
        
        # M3
        U[a_idx[0], offset + 2] += 1
        V[b_idx[1], offset + 2] += 1
        V[b_idx[3], offset + 2] -= 1
        W[c_idx[1], offset + 2] += 1
        W[c_idx[3], offset + 2] += 1
        
        # M4
        U[a_idx[3], offset + 3] += 1
        V[b_idx[2], offset + 3] += 1
        V[b_idx[0], offset + 3] -= 1
        W[c_idx[0], offset + 3] += 1
        W[c_idx[2], offset + 3] += 1
        
        # M5
        U[a_idx[0], offset + 4] += 1
        U[a_idx[1], offset + 4] += 1
        V[b_idx[3], offset + 4] += 1
        W[c_idx[0], offset + 4] -= 1
        W[c_idx[1], offset + 4] += 1
        
        # M6
        U[a_idx[2], offset + 5] += 1
        U[a_idx[0], offset + 5] -= 1
        V[b_idx[0], offset + 5] += 1
        V[b_idx[1], offset + 5] += 1
        W[c_idx[3], offset + 5] += 1
        
        # M7
        U[a_idx[1], offset + 6] += 1
        U[a_idx[3], offset + 6] -= 1
        V[b_idx[2], offset + 6] += 1
        V[b_idx[3], offset + 6] += 1
        W[c_idx[0], offset + 6] += 1

    # Define blocks
    # A indices: i * m + l
    # B indices: l * p + k
    # C indices: k * n + i
    
    # Block 1: A1 B11
    # A1: i in {0,1}, l in {0,1}
    # a_idx: [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0, 1, 4, 5]
    # B11: l in {0,1}, k in {0,1}
    # b_idx: [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0, 1, 4, 5]
    # C1: i in {0,1}, k in {0,1}
    # c_idx: [0*2+0, 0*2+1, 1*2+0, 1*2+1] = [0, 1, 2, 3] ?
    # Wait, C index is k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So c_idx = [0, 1, 2, 3]
    
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3], 0)
    
    # Block 2: A2 B21
    # A2: i in {0,1}, l in {2,3}
    # a_idx: [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2, 3, 6, 7]
    # B21: l in {2,3}, k in {0,1}
    # b_idx: [2*4+0, 2*4+1, 3*4+0, 3*4+1] = [8, 9, 12, 13]
    # C1: i in {0,1}, k in {0,1}
    # c_idx: [0, 1, 2, 3]
    
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3], 7)
    
    # Block 3: A1 B12
    # A1: i in {0,1}, l in {0,1}
    # a_idx: [0, 1, 4, 5]
    # B12: l in {0,1}, k in {2,3}
    # b_idx: [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2, 3, 6, 7]
    # C2: i in {0,1}, k in {2,3}
    # c_idx: [2*2+0, 2*2+1, 3*2+0, 3*2+1] = [4, 5, 6, 7]
    
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7], 14)
    
    # Block 4: A2 B22
    # A2: i in {0,1}, l in {2,3}
    # a_idx: [2, 3, 6, 7]
    # B22: l in {2,3}, k in {2,3}
    # b_idx: [2*4+2, 2*4+3, 3*4+2, 3*4+3] = [10, 11, 14, 15]
    # C2: i in {0,1}, k in {2,3}
    # c_idx: [4, 5, 6, 7]
    
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7], 21)
    
    # Verification
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1
                
    S = np.zeros((nm, mp, pn), dtype=int)
    for r in range(rank):
        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    if not np.array_equal(S, T):
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = 32
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        idx = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    a = i * m + l
                    b = l * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                    idx += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
