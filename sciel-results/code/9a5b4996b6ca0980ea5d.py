import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (a21+a22)b11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = a11(b12-b22)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = a22(b21-b11)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (a11+a12)b22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _rank1_block(n, m, p, i_off, j_off, k_off, U, V, W, col_start):
    """
    Adds a rank-1 block for C[i_off:i_off+n, k_off:k_off+p] = A[i_off:i_off+n, j_off:j_off+m] * B[j_off:j_off+m, k_off:k_off+p]
    where A block is n x m and B block is m x p, but effectively rank 1.
    This assumes the block multiplication is rank 1.
    For 2x1x1, 1x2x1, 2x2x1 (if structured), etc.
    Here we just add 1 column to U, V, W.
    U rows: (i_off + ii) * m_total + (j_off + jj) -- Wait, U index is i*m+j.
    We need to know m_total to index U correctly.
    """
    pass # Not used directly, we will construct manually.

def construct():
    n, m, p = 2, 5, 6
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (12, rank)
    
    # We will build the matrices column by column.
    # Total rank estimated 45.
    # Let's allocate 45 columns.
    R = 45
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        """
        Add Strassen 2x2x2 block at A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        i_off: 0 or 1 (but n=2, so only 0)
        j_off: 0, 2, 4 (for j=1,2; 3,4; 5,6? No, m=5. j=0,1; 2,3; 4? No.)
        
        m=5 indices 0..4.
        j_off can be 0 (j=0,1), 2 (j=2,3), 4 (j=4? No, needs 2).
        
        Our partition:
        A1: j=0,1.
        A2a: j=2,3.
        A2b: j=4.
        
        B1: j=0,1. k=0..5.
        B2a: j=2,3. k=0..5.
        B2b: j=4. k=0..5.
        
        Strassen blocks:
        1. A1 * B1a (k=0,1): i=0, j=0, k=0.
        2. A1 * B1b1 (k=2,3): i=0, j=0, k=2.
        3. A1 * B1b2 (k=4,5): i=0, j=0, k=4.
        4. A2a * B2a1 (k=0,1): i=0, j=2, k=0.
        5. A2a * B2a2 (k=2,3): i=0, j=2, k=2.
        6. A2a * B2a3 (k=4,5): i=0, j=2, k=4.
        """
        nonlocal col
        U_s, V_s, W_s = _strassen_222()
        for c in range(7):
            # U
            for ii in range(2):
                for jj in range(2):
                    u_idx = (i_off + ii) * m + (j_off + jj)
                    U[u_idx, col + c] += U_s[ii*2+jj, c]
            # V
            for jj in range(2):
                for kk in range(2):
                    v_idx = (j_off + jj) * p + (k_off + kk)
                    V[v_idx, col + c] += V_s[jj*2+kk, c]
            # W
            for kk in range(2):
                for ii in range(2):
                    w_idx = (k_off + kk) * n + (i_off + ii)
                    W[w_idx, col + c] += W_s[kk*2+ii, c]
        col += 7

    def add_rank1(i_off, j_off, k_off, n_b, m_b, p_b):
        """
        Add rank 1 block for A[i_off:i_off+n_b, j_off:j_off+m_b] * B[j_off:j_off+m_b, k_off:k_off+p_b]
        This assumes the product is rank 1.
        We set U, V, W such that U*V*W = 1 for the valid indices.
        For rank 1, we can set:
        U[row, col] = 1 for all rows in block? No.
        
        For 2x1x1 (n=2, m=1, p=1):
        C_i = A_i B.
        U[i*1+0, col] = 1
        V[0*1+0, col] = 1
        W[0*2+i, col] = 1
        
        For 2x2x1 (n=2, m=2, p=1) with C_i = (A_i0 + A_i1) B:
        U[i*2+0, col] = 1
        U[i*2+1, col] = 1
        V[0*1+0, col] = 1
        V[1*1+0, col] = 1
        W[0*2+i, col] = 1
        
        We need to know the structure.
        """
        nonlocal col
        # This function is too generic. We will handle specific cases manually.
        pass

    # Block 1: A1 * B1a (j=0,1; k=0,1)
    add_strassen(0, 0, 0)
    # Block 2: A1 * B1b1 (j=0,1; k=2,3)
    add_strassen(0, 0, 2)
    # Block 3: A1 * B1b2 (j=0,1; k=4,5)
    add_strassen(0, 0, 4)
    
    # Block 4: A2a * B2a1 (j=2,3; k=0,1)
    add_strassen(0, 2, 0)
    # Block 5: A2a * B2a2 (j=2,3; k=2,3)
    add_strassen(0, 2, 2)
    # Block 6: A2a * B2a3 (j=2,3; k=4,5)
    add_strassen(0, 2, 4)
    
    # Block 7: A2a * B2b (j=2,3; k=0..5)
    # C_i = (A_i2 + A_i3) B_4k
    # Rank 1.
    # U[i*5+2, col] = 1, U[i*5+3, col] = 1
    # V[4*6+k, col] = 1
    # W[k*2+i, col] = 1
    for i in range(2):
        U[i*m + 2, col] = 1
        U[i*m + 3, col] = 1
    for k in range(6):
        V[4*p + k, col] = 1
        W[k*n + i, col] = 1
    col += 1
    
    # Block 8: A2b * B2a (j=4; k=0..5)
    # C_i = A_i4 (B_2k + B_3k)
    # Rank 1.
    # U[i*5+4, col] = 1
    # V[2*6+k, col] = 1, V[3*6+k, col] = 1
    # W[k*2+i, col] = 1
    for i in range(2):
        U[i*m + 4, col] = 1
    for k in range(6):
        V[2*p + k, col] = 1
        V[3*p + k, col] = 1
        W[k*n + i, col] = 1
    col += 1
    
    # Block 9: A2b * B2b (j=4; k=0..5)
    # C_i = A_i4 B_4k
    # Rank 1.
    # U[i*5+4, col] = 1
    # V[4*6+k, col] = 1
    # W[k*2+i, col] = 1
    for i in range(2):
        U[i*m + 4, col] = 1
    for k in range(6):
        V[4*p + k, col] = 1
        W[k*n + i, col] = 1
    col += 1
    
    # Total columns used: 6*7 + 3 = 45.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
