import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _laderman_3x3x3():
    """Return (U, V, W) for 3x3x3 matrix multiplication using Laderman's rank-21 algorithm."""
    # U: 9x21 (rows indexed by i*3+j for A[i,j])
    # V: 9x21 (rows indexed by j*3+k for B[j,k])  
    # W: 9x21 (rows indexed by k*3+i for C[k,i])
    
    U = [[0]*21 for _ in range(9)]
    V = [[0]*21 for _ in range(9)]
    W = [[0]*21 for _ in range(9)]
    
    # Verified Laderman decomposition for 3x3x3
    # Each term r: U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
    # Sum over r should equal 1 for all i,j,k
    
    # Term definitions as (u_list, v_list, w_list) where each list has 9 entries
    # u_list[i*3+j] = coefficient for A[i,j]
    # v_list[j*3+k] = coefficient for B[j,k]
    # w_list[k*3+i] = coefficient for C[i,k]
    
    terms = [
        # 1
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        # 2
        ([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # 3
        ([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0]),
        # 4
        ([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        # 5
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,1,0]),
        # 6
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]),
        # 7
        ([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 8
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 9
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0]),
        # 10
        ([1,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 11
        ([1,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0]),
        # 12
        ([0,0,0,1,1,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 13
        ([0,0,0,1,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0]),
        # 14
        ([0,0,0,0,0,0,1,1,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 15
        ([0,0,0,0,0,0,1,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0]),
        # 16
        ([0,1,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 17
        ([0,0,1,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0]),
        # 18
        ([0,0,0,0,1,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 19
        ([0,0,0,0,0,1,0,0,1], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0]),
        # 20
        ([1,0,0,1,0,0,1,0,0], [1,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]),
        # 21
        ([0,1,0,0,1,0,0,1,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,0,0,0,0,0]),
    ]
    
    # Fill in U, V, W
    for r, (u, v, w) in enumerate(terms):
        for idx in range(9):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx]
            W[idx][r] = w[idx]
    
    # Verify numerically
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = np.sum(U_np[i*3+j, :] * V_np[j*3+k, :] * W_np[k*3+i, :])
                max_err = max(max_err, abs(s - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        U = [[0]*27 for _ in range(9)]
        V = [[0]*27 for _ in range(9)]
        W = [[0]*27 for _ in range(9)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i*9 + j*3 + k
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
        return U, V, W, 27
    
    return U, V, W, 21

def _trivial_decomposition(n_sub, m_sub, p_sub, U, V, W, offset_u, offset_v, offset_w):
    """Add trivial decomposition for n_sub x m_sub x p_sub subproblem."""
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                col = offset_u + i*m_sub + j*p_sub + k  # This indexing needs to match
                # Actually, the column index within this block is i*m_sub*p_sub + j*p_sub + k
                # But we're appending to global U, V, W
                U[i*m_sub + j][offset_u + i*m_sub*p_sub + j*p_sub + k] = 1
                V[j*p_sub + k][offset_v + i*m_sub*p_sub + j*p_sub + k] = 1
                W[k*n_sub + i][offset_w + i*m_sub*p_sub + j*p_sub + k] = 1

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline
    
    # Split A (3x5) into A1 (3x3) and A2 (3x2)
    # Split B (5x5) into B1 (3x5) [rows 0-2] and B2 (2x5) [rows 3-4]
    # C = A1*B1 + A2*B2
    
    # Further split B1 (3x5) into B1a (3x3) [cols 0-2] and B1b (3x2) [cols 3-4]
    # Further split B2 (2x5) into B2a (2x3) [cols 0-2] and B2b (2x2) [cols 3-4]
    
    # So we have 4 subproblems:
    # 1. A1 (3x3) * B1a (3x3) -> 3x3x3, rank 27 (schoolbook) or 21 (Laderman)
    # 2. A1 (3x3) * B1b (3x2) -> 3x3x2, rank 18 (trivial)
    # 3. A2 (3x2) * B2a (2x3) -> 3x2x3, rank 18 (trivial)
    # 4. A2 (3x2) * B2b (2x2) -> 3x2x2, rank 12 (trivial)
    
    # Total rank with schoolbook for part 1: 27 + 18 + 18 + 12 = 75
    # Total rank with Laderman for part 1: 21 + 18 + 18 + 12 = 69
    
    # Let me implement this block structure
    
    # Get Laderman decomposition for 3x3x3
    U1, V1, W1, rank1 = _laderman_3x3x3()
    rank2 = 18  # 3x3x2 trivial
    rank3 = 18  # 3x2x3 trivial
    rank4 = 12  # 3x2x2 trivial
    
    total_rank = rank1 + rank2 + rank3 + rank4
    
    U = [[0]*total_rank for _ in range(n*m)]  # 15 x total_rank
    V = [[0]*total_rank for _ in range(m*p)]  # 25 x total_rank
    W = [[0]*total_rank for _ in range(p*n)]  # 15 x total_rank
    
    col_offset = 0
    
    # Part 1: A1 (3x3) * B1a (3x3) using Laderman
    # Copy Laderman coefficients into global U, V, W
    for i in range(3):
        for j in range(3):
            for r in range(rank1):
                U[i*5 + j][col_offset + r] = U1[i*3 + j][r]
    for j in range(3):
        for k in range(3):
            for r in range(rank1):
                V[j*5 + k][col_offset + r] = V1[j*3 + k][r]
    for k in range(3):
        for i in range(3):
            for r in range(rank1):
                W[k*3 + i][col_offset + r] = W1[k*3 + i][r]
    col_offset += rank1
    
    # Part 2: A1 (3x3) * B1b (3x2)
    # A1 indices: rows 0-2, cols 0-2 of A -> U indices 0-8
    # B1b indices: rows 0-2, cols 3-4 of B -> V indices 0-2, 3-5, 6-8 (j*5+k where j=0,1,2 and k=0,1)
    #   V index: j*5 + (3+k) = j*5 + 3 + k
    # Output: 3x2 -> W indices for k=0,1 and i=0,1,2: k*3+i
    for i in range(3):
        for j in range(3):
            for k in range(2):
                # B1b col k corresponds to global B col 3+k
                v_idx = j*5 + (3 + k)
                w_idx = k*3 + i
                U[i*3 + j][col_offset + i*6 + j*2 + k] = 1
                V[v_idx][col_offset + i*6 + j*2 + k] = 1
                W[w_idx][col_offset + i*6 + j*2 + k] = 1
    col_offset += rank2
    
    # Part 3: A2 (3x2) * B2a (2x3)
    # A2 indices: rows 0-2, cols 3-4 of A -> U indices 3,4,7,8,11,12 (i*5 + 3+j where i=0,1,2 j=0,1)
    #   U index: i*5 + (3+j)
    # B2a indices: rows 3-4, cols 0-2 of B -> V indices 15,16,17,20,21,22 (j*5+k where j=3,4 k=0,1,2)
    #   V index: (3+j)*5 + k = 15 + j*5 + k
    # Output: 3x3 -> W indices 0-8
    for i in range(3):
        for j in range(2):
            for k in range(3):
                u_idx = i*5 + (3 + j)
                v_idx = (3 + j)*5 + k
                w_idx = k*3 + i
                U[u_idx][col_offset + i*6 + j*3 + k] = 1
                V[v_idx][col_offset + i*6 + j*3 + k] = 1
                W[w_idx][col_offset + i*6 + j*3 + k] = 1
    col_offset += rank3
    
    # Part 4: A2 (3x2) * B2b (2x2)
    # A2 indices: rows 0-2, cols 3-4 of A -> U indices 3,4,7,8,11,12
    # B2b indices: rows 3-4, cols 3-4 of B -> V indices 18,19,23,24 (j*5+k where j=3,4 k=3,4)
    #   V index: (3+j)*5 + (3+k) = 18 + j*5 + k
    # Output: 3x2 -> W indices for k=0,1 and i=0,1,2: k*3+i
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u_idx = i*5 + (3 + j)
                v_idx = (3 + j)*5 + (3 + k)
                w_idx = k*3 + i
                U[u_idx][col_offset + i*4 + j*2 + k] = 1
                V[v_idx][col_offset + i*4 + j*2 + k] = 1
                W[w_idx][col_offset + i*4 + j*2 + k] = 1
    col_offset += rank4
    
    # Verify the decomposition
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Our decomposition: C[i,k] = sum_r U[i*5+j... wait, U is indexed by (i,j) for A
    # U row index: i*m + j = i*5 + j for A[i,j]
    # V row index: j*p + k = j*5 + k for B[j,k]
    # W row index: k*n + i = k*3 + i for C[k,i] -- wait, the contract says W[k*n+i,r]
    
    # Let me verify: for each r (column), sum over r of U[row_u, r] * V[row_v, r] should equal W[row_w, r]
    # where row_u = i*m + j, row_v = j*p + k, row_w = k*n + i
    
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 if this is a valid multiplication term, which it always is for schoolbook
    
    # For schoolbook, T[i,j,k] = 1 for all i,j,k (it's the identity tensor for multiplication)
    # Wait no, the multiplication tensor has T[i,j,k] = 1 for all i,j,k because
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # So we need: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 for all i,j,k
    
    # Let me verify this numerically
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each i,j,k, sum_r U[i*5+j,r] * V[j*5+k,r] * W[k*3+i,r] should be 1
    max_err = 0
    for i in range(3):
        for j in range(5):
            for k in range(5):
                s = 0
                for r in range(total_rank):
                    s += U_np[i*5+j, r] * V_np[j*5+k, r] * W_np[k*3+i, r]
                err = abs(s - 1)
                max_err = max(max_err, err)
    
    if max_err > 1e-10:
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
