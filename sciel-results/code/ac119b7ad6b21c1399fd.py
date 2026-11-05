"""Matrix multiplication 5x5x5 with reduced rank decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) where:
    U has shape (4, 7) - for A matrix (2x2), U[i*2+j, r]
    V has shape (4, 7) - for B matrix (2x2), V[j*2+k, r]
    W has shape (4, 7) - for C matrix (2x2), W[k*2+i, r]
    
    C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # p3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g-e)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1
    
    # p5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # p6 = (c-a)(e+f)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p.
    
    U has shape (n*m, n*m*p)
    V has shape (m*p, n*m*p)
    W has shape (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for test in range(num_tests):
        rng = np.random.RandomState(42 + test)
        A = rng.randint(-10, 10, (n, m))
        B = rng.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=int)
        for r in range(U.shape[1]):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
                    C_computed[i, k] += s
        
        if not np.array_equal(C_computed, C_expected):
            return False
    
    return True

def _block_decompose_5x5():
    """Construct 5x5x5 decomposition by partitioning into 3x3 and 2x2 blocks.
    
    Partition:
    A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    B = [[B11, B12], [B21, B22]] similarly
    C = [[C11, C12], [C21, C22]]
    
    C11 = A11*B11 + A12*B21 (3x3)
    C12 = A11*B12 + A12*B22 (3x2)
    C21 = A21*B11 + A22*B21 (2x3)
    C22 = A21*B12 + A22*B22 (2x2)
    
    Uses Strassen for 2x2x2 block, schoolbook for others.
    """
    n, m, p = 5, 5, 5
    
    # Get Strassen 2x2 decomposition
    U22, V22, W22 = _strassen_2x2()
    
    # Block dimensions
    n1, n2 = 3, 2  # row partition
    m1, m2 = 3, 2  # middle dimension partition
    p1, p2 = 3, 2  # output column partition
    
    # Compute ranks for each block multiplication
    # A11*B11: 3x3 * 3x3 -> schoolbook rank 27
    rank_A11B11 = 3 * 3 * 3  # 27
    # A12*B21: 3x2 * 2x3 -> schoolbook rank 6
    rank_A12B21 = 3 * 2 * 3  # 18... wait, 3*2*3 = 18? No, n*m*p = 3*2*3 = 18
    # Actually for A12 (3x2) * B21 (2x3) -> C11 contribution (3x3): n=3, m=2, p=3, rank = 3*2*3 = 18
    # Hmm, but that's the schoolbook rank. Let me recalculate.
    # For multiplying an (n x m) matrix by an (m x p) matrix, schoolbook rank is n*m*p.
    # A12 is 3x2, B21 is 2x3, result is 3x3. Rank = 3*2*3 = 18.
    # Wait, that doesn't seem right either. Let me think again.
    # The tensor for matrix multiplication of (n x m) * (m x p) has rank at most n*m*p.
    # A12 (3x2) * B21 (2x3): n=3, m=2, p=3, so rank <= 3*2*3 = 18.
    # But actually, this is just a specific sub-problem. The schoolbook for this would be:
    # C[i,k] = sum_j A12[i,j] * B21[j,k] where i in 0-2, j in 0-1, k in 0-2
    # So there are 3*2*3 = 18 terms. Rank = 18.
    
    # Let me recompute all:
    # A11*B11: (3x3)*(3x3) -> rank 27
    # A12*B21: (3x2)*(2x3) -> rank 3*2*3 = 18
    # A11*B12: (3x3)*(3x2) -> rank 3*3*2 = 18
    # A12*B22: (3x2)*(2x2) -> rank 3*2*2 = 12
    # A21*B11: (2x3)*(3x3) -> rank 2*3*3 = 18
    # A22*B21: (2x2)*(2x3) -> rank 2*2*3 = 12
    # A21*B12: (2x3)*(3x2) -> rank 2*3*2 = 12
    # A22*B22: (2x2)*(2x2) -> Strassen rank 7
    
    rank_A11B11 = 27
    rank_A12B21 = 18
    rank_A11B12 = 18
    rank_A12B22 = 12
    rank_A21B11 = 18
    rank_A22B21 = 12
    rank_A21B12 = 12
    rank_A22B22 = 7
    
    total_rank = (rank_A11B11 + rank_A12B21 + rank_A11B12 + rank_A12B22 +
                  rank_A21B11 + rank_A22B21 + rank_A21B12 + rank_A22B22)
    # = 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 124
    
    print(f"Total rank: {total_rank}")
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col_offset = 0
    
    def add_block(U_block, V_block, W_block, u_rows, v_rows, w_rows, rank_block):
        nonlocal col_offset
        U[u_rows, col_offset:col_offset + rank_block] = U_block
        V[v_rows, col_offset:col_offset + rank_block] = V_block
        W[w_rows, col_offset:col_offset + rank_block] = W_block
        col_offset += rank_block
    
    # A11*B11: 3x3 * 3x3 -> 3x3
    # A11: A[i*5+j] for i=0-2, j=0-2 -> indices [0,1,2, 5,6,7, 10,11,12]
    # B11: B[j*5+k] for j=0-2, k=0-2 -> indices [0,1,2, 5,6,7, 10,11,12]
    # C11: C[k*5+i] for k=0-2, i=0-2 -> indices [0,1,2, 5,6,7, 10,11,12]
    U_A11B11, V_A11B11, W_A11B11 = _schoolbook_block(3, 3, 3)
    u_rows = [i * 5 + j for i in range(3) for j in range(3)]
    v_rows = [j * 5 + k for j in range(3) for k in range(3)]
    w_rows = [k * 5 + i for k in range(3) for i in range(3)]
    add_block(U_A11B11, V_A11B11, W_A11B11, u_rows, v_rows, w_rows, rank_A11B11)
    
    # A12*B21: 3x2 * 2x3 -> 3x3
    # A12: A[i*5+j] for i=0-2, j=3-4 -> indices [3,4, 8,9, 13,14]
    # B21: B[j*5+k] for j=3-4, k=0-2 -> indices [15,16,17, 20,21,22]
    # C11: C[k*5+i] for k=0-2, i=0-2 -> indices [0,1,2, 5,6,7, 10,11,12]
    U_A12B21, V_A12B21, W_A12B21 = _schoolbook_block(3, 2, 3)
    u_rows = [i * 5 + (3 + j) for i in range(3) for j in range(2)]
    v_rows = [(3 + j) * 5 + k for j in range(2) for k in range(3)]
    w_rows = [k * 5 + i for k in range(3) for i in range(3)]
    add_block(U_A12B21, V_A12B21, W_A12B21, u_rows, v_rows, w_rows, rank_A12B21)
    
    # A11*B12: 3x3 * 3x2 -> 3x2
    # A11: A[i*5+j] for i=0-2, j=0-2 -> indices [0,1,2, 5,6,7, 10,11,12]
    # B12: B[j*5+k] for j=0-2, k=3-4 -> indices [3,4, 8,9, 13,14]
    # C12: C[k*5+i] for k=3-4, i=0-2 -> indices [15,16,17, 20,21,22]
    U_A11B12, V_A11B12, W_A11B12 = _schoolbook_block(3, 3, 2)
    u_rows = [i * 5 + j for i in range(3) for j in range(3)]
    v_rows = [j * 5 + (3 + k) for j in range(3) for k in range(2)]
    w_rows = [(3 + k) * 5 + i for k in range(2) for i in range(3)]
    add_block(U_A11B12, V_A11B12, W_A11B12, u_rows, v_rows, w_rows, rank_A11B12)
    
    # A12*B22: 3x2 * 2x2 -> 3x2
    # A12: A[i*5+j] for i=0-2, j=3-4 -> indices [3,4, 8,9, 13,14]
    # B22: B[j*5+k] for j=3-4, k=3-4 -> indices [18,19, 23,24]
    # C12: C[k*5+i] for k=3-4, i=0-2 -> indices [15,16,17, 20,21,22]
    U_A12B22, V_A12B22, W_A12B22 = _schoolbook_block(3, 2, 2)
    u_rows = [i * 5 + (3 + j) for i in range(3) for j in range(2)]
    v_rows = [(3 + j) * 5 + (3 + k) for j in range(2) for k in range(2)]
    w_rows = [(3 + k) * 5 + i for k in range(2) for i in range(3)]
    add_block(U_A12B22, V_A12B22, W_A12B22, u_rows, v_rows, w_rows, rank_A12B22)
    
    # A21*B11: 2x3 * 3x3 -> 2x3
    # A21: A[i*5+j] for i=3-4, j=0-2 -> indices [15,16,17, 20,21,22]
    # B11: B[j*5+k] for j=0-2, k=0-2 -> indices [0,1,2, 5,6,7, 10,11,12]
    # C21: C[k*5+i] for k=0-2, i=3-4 -> indices [3,4, 8,9, 13,14]
    U_A21B11, V_A21B11, W_A21B11 = _schoolbook_block(2, 3, 3)
    u_rows = [(3 + i) * 5 + j for i in range(2) for j in range(3)]
    v_rows = [j * 5 + k for j in range(3) for k in range(3)]
    w_rows = [k * 5 + (3 + i) for k in range(3) for i in range(2)]
    add_block(U_A21B11, V_A21B11, W_A21B11, u_rows, v_rows, w_rows, rank_A21B11)
    
    # A22*B21: 2x2 * 2x3 -> 2x3
    # A22: A[i*5+j] for i=3-4, j=3-4 -> indices [18,19, 23,24]
    # B21: B[j*5+k] for j=3-4, k=0-2 -> indices [15,16,17, 20,21,22]
    # C21: C[k*5+i] for k=0-2, i=3-4 -> indices [3,4, 8,9, 13,14]
    U_A22B21, V_A22B21, W_A22B21 = _schoolbook_block(2, 2, 3)
    u_rows = [(3 + i) * 5 + (3 + j) for i in range(2) for j in range(2)]
    v_rows = [(3 + j) * 5 + k for j in range(2) for k in range(3)]
    w_rows = [k * 5 + (3 + i) for k in range(3) for i in range(2)]
    add_block(U_A22B21, V_A22B21, W_A22B21, u_rows, v_rows, w_rows, rank_A22B21)
    
    # A21*B12: 2x3 * 3x2 -> 2x2
    # A21: A[i*5+j] for i=3-4, j=0-2 -> indices [15,16,17, 20,21,22]
    # B12: B[j*5+k] for j=0-2, k=3-4 -> indices [3,4, 8,9, 13,14]
    # C22: C[k*5+i] for k=3-4, i=3-4 -> indices [18,19, 23,24]
    U_A21B12, V_A21B12, W_A21B12 = _schoolbook_block(2, 3, 2)
    u_rows = [(3 + i) * 5 + j for i in range(2) for j in range(3)]
    v_rows = [j * 5 + (3 + k) for j in range(3) for k in range(2)]
    w_rows = [(3 + k) * 5 + (3 + i) for k in range(2) for i in range(2)]
    add_block(U_A21B12, V_A21B12, W_A21B12, u_rows, v_rows, w_rows, rank_A21B12)
    
    # A22*B22: 2x2 * 2x2 -> 2x2, use Strassen
    # A22: A[i*5+j] for i=3-4, j=3-4 -> indices [18,19, 23,24]
    # B22: B[j*5+k] for j=3-4, k=3-4 -> indices [18,19, 23,24]
    # C22: C[k*5+i] for k=3-4, i=3-4 -> indices [18,19, 23,24]
    u_rows = [(3 + i) * 5 + (3 + j) for i in range(2) for j in range(2)]
    v_rows = [(3 + j) * 5 + (3 + k) for j in range(2) for k in range(2)]
    w_rows = [(3 + k) * 5 + (3 + i) for k in range(2) for i in range(2)]
    add_block(U22, V22, W22, u_rows, v_rows, w_rows, rank_A22B22)
    
    # Verify correctness
    is_valid = _verify_decomposition(U, V, W, n, m, p, num_tests=5)
    print(f"Decomposition valid: {is_valid}, rank: {total_rank}")
    
    if not is_valid:
        # Fallback to schoolbook
        rank = n * m * p
        U, V, W = _schoolbook_block(n, m, p)
        total_rank = rank
        print(f"Fallback to schoolbook, rank: {total_rank}")
    
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    try:
        result = _block_decompose_5x5()
    except Exception as e:
        print(f"Error in block decomposition: {e}")
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        result = {
            "U": [[int(x) for x in row] for row in U],
            "V": [[int(x) for x in row] for row in V],
            "W": [[int(x) for x in row] for row in W]
        }
    
    if time.time() > deadline:
        print("Deadline exceeded, returning result")
    
    return result
# EVOLVE-BLOCK-END
