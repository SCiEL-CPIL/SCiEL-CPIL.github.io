import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition of m=5 into [2, 2, 1]
    # We will handle each block independently and concatenate the rank terms.
    # Block 1: A (2x2) * B (2x6) -> C (2x6)
    # Block 2: A (2x2) * B (2x6) -> C (2x6)
    # Block 3: A (2x1) * B (1x6) -> C (2x6)
    
    # Total Rank:
    # Block 1: 2x2x6. Split p=6 into three 2s. 3 * Rank(2x2x2) = 3 * 7 = 21.
    # Block 2: 2x2x6. Same as Block 1. 21.
    # Block 3: 2x1x6. Rank is 2*1*6 = 12.
    # Total Rank = 21 + 21 + 12 = 54.
    
    # Indices:
    # A is n x m. Row index in U: i*m + j.
    # B is m x p. Row index in V: j*p + k.
    # C is n x p. Row index in W: k*n + i.
    
    # Let's build the U, V, W matrices.
    # We'll use a list of columns first, then convert to matrices.
    # Each column corresponds to a rank-1 term.
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: dict {row_index: value}
        v_vec: dict {row_index: value}
        w_vec: dict {row_index: value}
        """
        cols_U.append(u_vec)
        cols_V.append(v_vec)
        cols_W.append(w_vec)
        
    # Strassen decomposition for 2x2x2 multiplication
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    #
    # The 7 rank-1 terms for Strassen (in terms of linear indices):
    # We need to map the Strassen intermediates to the specific rows of U, V, W.
    #
    # Let's define the indices for a generic 2x2x2 block starting at A_row offset a_off, B_col offset b_off, C_col offset c_off.
    # Wait, the indices are global.
    #
    # Let's define a helper to add a 2x2x2 Strassen block.
    # a_off: offset in A's row index (i*m + j). Since we are splitting m, the j offset is the block start in m.
    # b_off: offset in B's row index (j*p + k). The j offset is the block start in m. The k offset is the block start in p.
    # c_off: offset in C's row index (k*n + i). The k offset is the block start in p.
    
    def add_strassen_2x2x2(a_j_start, b_j_start, k_start):
        """
        Adds 7 terms for multiplying A[:, j_start:j_start+2] (2x2) by B[j_start:j_start+2, k_start:k_start+2] (2x2)
        contributing to C[:, k_start:k_start+2].
        
        a_j_start: the j index in the original m dimension for the start of this 2x2 block.
        b_j_start: same as a_j_start, since B's rows correspond to A's cols.
        k_start: the k index in the original p dimension for the start of this 2x2 block.
        
        A indices: i in {0,1}, j in {j_start, j_start+1}.
        U row: i*m + j.
        
        B indices: j in {j_start, j_start+1}, k in {k_start, k_start+1}.
        V row: j*p + k.
        
        C indices: i in {0,1}, k in {k_start, k_start+1}.
        W row: k*n + i.
        """
        
        # Strassen Formulas:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to construct the vectors for U, V, W for each M.
        
        # M1: (A11 + A22) * (B11 + B22)
        # U: A11=1, A22=1. Others 0.
        # V: B11=1, B22=1. Others 0.
        # W: C11+=1, C22+=1. Others 0.
        u = {0*m + j_start: 1.0, 1*m + (j_start+1): 1.0}
        v = {j_start*p + k_start: 1.0, (j_start+1)*p + (k_start+1): 1.0}
        w = {k_start*n + 0: 1.0, (k_start+1)*n + 1: 1.0}
        add_term(u, v, w)
        
        # M2: (A21 + A22) * B11
        # U: A21=1, A22=1.
        # V: B11=1.
        # W: C21+=1.
        u = {1*m + j_start: 1.0, 1*m + (j_start+1): 1.0}
        v = {j_start*p + k_start: 1.0}
        w = {k_start*n + 1: 1.0}
        add_term(u, v, w)
        
        # M3: A11 * (B12 - B22)
        # U: A11=1.
        # V: B12=1, B22=-1.
        # W: C12+=1.
        u = {0*m + j_start: 1.0}
        v = {j_start*p + (k_start+1): 1.0, (j_start+1)*p + (k_start+1): -1.0}
        w = {(k_start+1)*n + 0: 1.0}
        add_term(u, v, w)
        
        # M4: A22 * (B21 - B11)
        # U: A22=1.
        # V: B21=1, B11=-1.
        # W: C11+=1, C21+=1.
        u = {1*m + (j_start+1): 1.0}
        v = {(j_start+1)*p + k_start: 1.0, j_start*p + k_start: -1.0}
        w = {k_start*n + 0: 1.0, k_start*n + 1: 1.0}
        add_term(u, v, w)
        
        # M5: (A11 + A12) * B22
        # U: A11=1, A12=1.
        # V: B22=1.
        # W: C11-=1, C12+=1.
        u = {0*m + j_start: 1.0, 0*m + (j_start+1): 1.0}
        v = {(j_start+1)*p + (k_start+1): 1.0}
        w = {k_start*n + 0: -1.0, (k_start+1)*n + 0: 1.0}
        add_term(u, v, w)
        
        # M6: (A21 - A11) * (B11 + B12)
        # U: A21=1, A11=-1.
        # V: B11=1, B12=1.
        # W: C22+=1.
        u = {1*m + j_start: 1.0, 0*m + j_start: -1.0}
        v = {j_start*p + k_start: 1.0, j_start*p + (k_start+1): 1.0}
        w = {(k_start+1)*n + 1: 1.0}
        add_term(u, v, w)
        
        # M7: (A12 - A22) * (B21 + B22)
        # U: A12=1, A22=-1.
        # V: B21=1, B22=1.
        # W: C11+=1.
        u = {0*m + (j_start+1): 1.0, 1*m + (j_start+1): -1.0}
        v = {(j_start+1)*p + k_start: 1.0, (j_start+1)*p + (k_start+1): 1.0}
        w = {k_start*n + 0: 1.0}
        add_term(u, v, w)

    # Block 1: m indices 0,1. p indices 0,1; 2,3; 4,5.
    # A rows 0,1. B rows 0,1.
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    add_strassen_2x2x2(0, 0, 4)
    
    # Block 2: m indices 2,3. p indices 0,1; 2,3; 4,5.
    # A rows 2,3. B rows 2,3.
    add_strassen_2x2x2(2, 2, 0)
    add_strassen_2x2x2(2, 2, 2)
    add_strassen_2x2x2(2, 2, 4)
    
    # Block 3: m index 4. p indices 0..5.
    # This is 2x1x6. Rank 12.
    # For each i in {0,1} and k in {0..5}:
    # C[i,k] = A[i,4] * B[4,k]
    # Term: U[i*m+4]=1, V[4*p+k]=1, W[k*n+i]=1.
    for i in range(2):
        for k in range(6):
            u = {i*m + 4: 1.0}
            v = {4*p + k: 1.0}
            w = {k*n + i: 1.0}
            add_term(u, v, w)
            
    # Now construct the matrices
    rank = len(cols_U)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        for row, val in cols_U[r].items():
            U[row, r] = val
        for row, val in cols_V[r].items():
            V[row, r] = val
        for row, val in cols_W[r].items():
            W[row, r] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Using block Strassen decomposition for the 2x2 parts reduces the rank from 60 to 54 by exploiting the low rank of 2x2 matrix multiplication.", q: 0.9, delta: [0.0, 0.1]}
