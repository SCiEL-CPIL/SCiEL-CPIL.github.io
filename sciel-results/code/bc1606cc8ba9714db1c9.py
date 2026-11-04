import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # n: 2, 2
    # m: 2, 2, 1
    # p: 2, 2, 1
    
    n_splits = [2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    # Calculate total rank
    # 8 blocks of 2x2x2 (rank 7)
    # 4 blocks of 2x2x1 (rank 4)
    # 4 blocks of 2x1x2 (rank 4)
    # 2 blocks of 2x1x1 (rank 2)
    total_rank = 8*7 + 4*4 + 4*4 + 2*2
    total_rank = 56 + 16 + 16 + 4
    total_rank = 92
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    rank_offset = 0
    
    # Helper to add schoolbook block
    def add_schoolbook(i_start, j_start, k_start, n_i, m_j, p_k):
        nonlocal rank_offset
        for i in range(n_i):
            for j in range(m_j):
                for k in range(p_k):
                    u_idx = (i_start + i) * m + (j_start + j)
                    v_idx = (j_start + j) * p + (k_start + k)
                    w_idx = (k_start + k) * n + (i_start + i)
                    
                    U[u_idx, rank_offset] = 1
                    V[v_idx, rank_offset] = 1
                    W[w_idx, rank_offset] = 1
                    rank_offset += 1

    # Helper to add Strassen block (2x2x2)
    def add_strassen(i_start, j_start, k_start):
        nonlocal rank_offset
        # i, j, k are 0..1 relative to block
        # Global indices:
        # i0 = i_start, i1 = i_start + 1
        # j0 = j_start, j1 = j_start + 1
        # k0 = k_start, k1 = k_start + 1
        
        # P1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[i_start*m + j_start, rank_offset] = 1
        U[(i_start+1)*m + (j_start+1), rank_offset] = 1
        V[j_start*p + k_start, rank_offset] = 1
        V[(j_start+1)*p + (k_start+1), rank_offset] = 1
        W[k_start*n + i_start, rank_offset] = 1
        W[(k_start+1)*n + (i_start+1), rank_offset] = 1
        rank_offset += 1
        
        # P2 = (A10 + A11)B01
        U[i_start*m + (j_start+1), rank_offset] = 1 # A10? No. A_{i,j}. i=1, j=0 is A10.
        # Wait, A is n x m. A[i,j].
        # P2 = (A_{21} + A_{22}) B_{12}
        # Indices: A_{1,0} + A_{1,1} times B_{0,1}
        # In 0-indexed: A[1,0], A[1,1], B[0,1]
        # C_{12} = C[1,1] gets P2. C_{21} = C[1,0]? No.
        # C = [[C00, C01], [C10, C11]]
        # C01 = P2 + P3
        # C10 = P6 + P5
        # C00 = P1 + P4 - P5 + P7
        # C11 = P1 - P2 + P4 + P6
        
        # Let's stick to 1-based logic for Strassen variables then map.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # P1 = (A11+A22)(B11+B22) -> C11+=P1, C22+=P1
        # P2 = (A21+A22)B12 -> C12+=P2
        # P3 = A11(B21+B22) -> C12+=P3
        # P4 = A22(B11-B21) -> C11+=P4, C22+=P4
        # P5 = (A11+A12)B22 -> C11-=P5, C21+=P5
        # P6 = (A21-A11)(B11+B12) -> C11+=P6, C21+=P6, C22+=P6
        # P7 = (A12-A22)(B21+B22) -> C11+=P7, C22-=P7
        
        # Mapping to 0-indexed global:
        # A11 -> U[i0, j0]
        # A12 -> U[i0, j1]
        # A21 -> U[i1, j0]
        # A22 -> U[i1, j1]
        
        # B11 -> V[j0, k0]
        # B12 -> V[j0, k1]
        # B21 -> V[j1, k0]
        # B22 -> V[j1, k1]
        
        # C11 -> W[k0, i0]
        # C12 -> W[k1, i0]
        # C21 -> W[k0, i1]
        # C22 -> W[k1, i1]
        
        # P2: (A21+A22)B12
        U[i_start*m + (j_start+1), rank_offset] = 1 # A21 (i1, j0) -> Wait. A21 is row 2, col 1. i=1, j=0.
        # My previous line was U[i_start*m + (j_start+1)] which is i0, j1. That is A12.
        # Correct: A21 is i1, j0. A22 is i1, j1.
        U[i_start*m + j_start, rank_offset] = 0 # Clear just in case
        U[(i_start+1)*m + j_start, rank_offset] = 1 # A21
        U[(i_start+1)*m + (j_start+1), rank_offset] = 1 # A22
        V[j_start*p + (k_start+1), rank_offset] = 1 # B12 (j0, k1)
        W[(k_start+1)*n + i_start, rank_offset] = 1 # C12 (k1, i0)
        rank_offset += 1
        
        # P3: A11(B21+B22)
        U[i_start*m + j_start, rank_offset] = 1 # A11
        V[(j_start+1)*p + k_start, rank_offset] = 1 # B21 (j1, k0)
        V[(j_start+1)*p + (k_start+1), rank_offset] = 1 # B22 (j1, k1)
        W[(k_start+1)*n + i_start, rank_offset] = 1 # C12
        rank_offset += 1
        
        # P4: A22(B11-B21)
        U[(i_start+1)*m + (j_start+1), rank_offset] = 1 # A22
        V[j_start*p + k_start, rank_offset] = 1 # B11
        V[(j_start+1)*p + k_start, rank_offset] = -1 # B21
        W[k_start*n + i_start, rank_offset] = 1 # C11
        W[(k_start+1)*n + (i_start+1), rank_offset] = 1 # C22
        rank_offset += 1
        
        # P5: (A11+A12)B22
        U[i_start*m + j_start, rank_offset] = 1 # A11
        U[i_start*m + (j_start+1), rank_offset] = 1 # A12
        V[(j_start+1)*p + (k_start+1), rank_offset] = 1 # B22
        W[k_start*n + i_start, rank_offset] = -1 # C11
        W[k_start*n + (i_start+1), rank_offset] = 1 # C21
        rank_offset += 1
        
        # P6: (A21-A11)(B11+B12)
        U[(i_start+1)*m + j_start, rank_offset] = 1 # A21
        U[i_start*m + j_start, rank_offset] = -1 # A11
        V[j_start*p + k_start, rank_offset] = 1 # B11
        V[j_start*p + (k_start+1), rank_offset] = 1 # B12
        W[k_start*n + i_start, rank_offset] = 1 # C11
        W[k_start*n + (i_start+1), rank_offset] = 1 # C21
        W[(k_start+1)*n + (i_start+1), rank_offset] = 1 # C22
        rank_offset += 1
        
        # P7: (A12-A22)(B21+B22)
        U[i_start*m + (j_start+1), rank_offset] = 1 # A12
        U[(i_start+1)*m + (j_start+1), rank_offset] = -1 # A22
        V[(j_start+1)*p + k_start, rank_offset] = 1 # B21
        V[(j_start+1)*p + (k_start+1), rank_offset] = 1 # B22
        W[k_start*n + i_start, rank_offset] = 1 # C11
        W[(k_start+1)*n + (i_start+1), rank_offset] = -1 # C22
        rank_offset += 1

    # Iterate blocks
    i_start = 0
    for n_i in n_splits:
        j_start = 0
        for m_j in m_splits:
            k_start = 0
            for p_k in p_splits:
                if n_i == 2 and m_j == 2 and p_k == 2:
                    add_strassen(i_start, j_start, k_start)
                else:
                    add_schoolbook(i_start, j_start, k_start, n_i, m_j, p_k)
                k_start += p_k
            j_start += m_j
        i_start += n_i
        
    return {"U": U, "V": V, "W": W}
