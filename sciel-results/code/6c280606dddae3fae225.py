import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_strassen_block(k_start, r_start):
        # U: i*m+j, V: j*p+k, W: k*n+i
        # Term 1
        U[0, r_start+0] = 1
        U[3, r_start+0] = 1
        V[k_start, r_start+0] = 1
        V[9 + k_start + 1, r_start+0] = 1
        W[k_start, r_start+0] = 1
        W[2 + k_start, r_start+0] = 1
        W[k_start + 2, r_start+0] = 1
        W[4 + k_start, r_start+0] = 1
        
        # Term 2
        U[2, r_start+1] = 1
        U[3, r_start+1] = 1
        V[k_start, r_start+1] = 1
        W[k_start + 2, r_start+1] = 1
        W[4 + k_start, r_start+1] = 1
        
        # Term 3
        U[0, r_start+2] = 1
        V[9 + k_start + 1, r_start+2] = 1
        V[18 + k_start + 1, r_start+2] = -1
        W[2 + k_start, r_start+2] = 1
        W[4 + k_start, r_start+2] = 1
        
        # Term 4
        U[3, r_start+3] = 1
        V[18 + k_start + 1, r_start+3] = 1
        V[k_start, r_start+3] = -1
        W[k_start, r_start+3] = 1
        W[k_start + 2, r_start+3] = 1
        
        # Term 5
        U[0, r_start+4] = 1
        U[2, r_start+4] = 1
        V[18 + k_start + 1, r_start+4] = 1
        W[k_start, r_start+4] = 1
        W[2 + k_start, r_start+4] = 1
        
        # Term 6
        U[2, r_start+5] = 1
        U[0, r_start+5] = -1
        V[k_start, r_start+5] = 1
        V[9 + k_start + 1, r_start+5] = 1
        W[4 + k_start, r_start+5] = 1
        
        # Term 7
        U[1, r_start+6] = 1
        U[3, r_start+6] = -1
        V[18 + k_start + 1, r_start+6] = 1
        V[9 + k_start, r_start+6] = 1
        W[k_start, r_start+6] = 1
        W[k_start + 2, r_start+6] = 1

    def add_schoolbook_2x1_block(k_start, r_start):
        # Term 1: A00 B00 -> C0
        U[0, r_start+0] = 1
        V[k_start, r_start+0] = 1
        W[k_start, r_start+0] = 1
        
        # Term 2: A01 B10 -> C0
        U[1, r_start+1] = 1
        V[9 + k_start, r_start+1] = 1
        W[k_start, r_start+1] = 1
        
        # Term 3: A10 B00 -> C1
        U[2, r_start+2] = 1
        V[k_start, r_start+2] = 1
        W[2 + k_start, r_start+2] = 1
        
        # Term 4: A11 B10 -> C1
        U[3, r_start+3] = 1
        V[9 + k_start, r_start+3] = 1
        W[2 + k_start, r_start+3] = 1

    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    add_strassen_block(6, 21)
    add_schoolbook_2x1_block(8, 28)

    return {"U": U, "V": V, "W": W}
