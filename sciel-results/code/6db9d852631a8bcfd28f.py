"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification using tensor contraction."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _laderman_2x2x2():
    """
    Laderman's rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as lists of lists with integer entries.
    
    The Laderman decomposition factors:
    C = A * B where A, B, C are 2x2 matrices.
    
    Using the standard Laderman identity:
    Let A = [[a00, a01], [a10, a11]]
    Let B = [[b00, b01], [b10, b11]]
    
    The 7 intermediate products are:
    p0 = (a00 + a11) * (b00 + b11)
    p1 = (a00 - a11) * (b00 - b11)
    p2 = (a00 + a10) * (b10 + b11)
    p3 = (a11 - a10) * (b00 - b10)
    p4 = (a01 + a11) * (b00 + b01)
    p5 = (a01 - a10) * (b10 - b11)
    p6 = (a00 + a01) * (b01 - b11)
    
    Then:
    c00 = p0 - p1 + p3 + p5 - p6
    c01 = p4 - p5 + p6
    c10 = p2 - p3 + p5
    c11 = p0 - p2 + p4
    """
    # U matrix: shape (n*m, rank) = (4, 7)
    # Indexing: U[i*m+j, r] where i=0,1 and j=0,1
    # For 2x2, n=2, m=2
    # U[i*2+j, r]
    
    # V matrix: shape (m*p, rank) = (4, 7)
    # Indexing: V[j*p+k, r] where j=0,1 and k=0,1
    # For 2x2, m=2, p=2
    # V[j*2+k, r]
    
    # W matrix: shape (p*n, rank) = (4, 7)
    # Indexing: W[k*n+i, r] where k=0,1 and i=0,1
    # For 2x2, p=2, n=2
    # W[k*2+i, r]
    
    # p0 = (a00 + a11) * (b00 + b11)
    # U contributions: a00 (U[0,0]=1), a11 (U[3,0]=1)
    # V contributions: b00 (V[0,0]=1), b11 (V[3,0]=1)
    # W contributions: c00 (W[0,0]=1), c11 (W[3,0]=1)
    
    # p1 = (a00 - a11) * (b00 - b11)
    # U: a00 (U[0,1]=1), a11 (U[3,1]=-1)
    # V: b00 (V[0,1]=1), b11 (V[3,1]=-1)
    # W: c00 (W[0,1]=-1), c11 (W[3,1]=-1)
    
    # p2 = (a00 + a10) * (b10 + b11)
    # U: a00 (U[0,2]=1), a10 (U[2,2]=1)
    # V: b10 (V[2,2]=1), b11 (V[3,2]=1)
    # W: c10 (W[2,2]=-1), c11 (W[3,2]=1)
    
    # p3 = (a11 - a10) * (b00 - b10)
    # U: a11 (U[3,3]=1), a10 (U[2,3]=-1)
    # V: b00 (V[0,3]=1), b10 (V[2,3]=-1)
    # W: c00 (W[0,3]=1), c10 (W[2,3]=-1)
    
    # p4 = (a01 + a11) * (b00 + b01)
    # U: a01 (U[1,4]=1), a11 (U[3,4]=1)
    # V: b00 (V[0,4]=1), b01 (V[1,4]=1)
    # W: c01 (W[1,4]=1), c11 (W[3,4]=1)
    
    # p5 = (a01 - a10) * (b10 - b11)
    # U: a01 (U[1,5]=1), a10 (U[2,5]=-1)
    # V: b10 (V[2,5]=1), b11 (V[3,5]=-1)
    # W: c00 (W[0,5]=1), c01 (W[1,5]=-1), c10 (W[2,5]=1)
    
    # p6 = (a00 + a01) * (b01 - b11)
    # U: a00 (U[0,6]=1), a01 (U[1,6]=1)
    # V: b01 (V[1,6]=1), b11 (V[3,6]=-1)
    # W: c00 (W[0,6]=-1), c01 (W[1,6]=1)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p1
    U[0][1] = 1; U[3][1] = -1
    V[0][1] = 1; V[3][1] = -1
    W[0][1] = -1; W[3][1] = -1
    
    # p2
    U[0][2] = 1; U[2][2] = 1
    V[2][2] = 1; V[3][2] = 1
    W[2][2] = -1; W[3][2] = 1
    
    # p3
    U[3][3] = 1; U[2][3] = -1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = -1
    
    # p4
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # p5
    U[1][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = -1
    W[0][5] = 1; W[1][5] = -1; W[2][5] = 1
    
    # p6
    U[0][6] = 1; U[1][6] = 1
    V[1][6] = 1; V[3][6] = -1
    W[0][6] = -1; W[1][6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """
    Build a decomposition using Laderman for 2x2x2 blocks and schoolbook for the remainder.
    
    For 2x2 * 2x7:
    - Split the 7 columns of B into three 2-column blocks (cols 0-1, 2-3, 4-5) and one 1-column block (col 6)
    - Each 2-column block: A (2x2) * B_block (2x2) = C_block (2x2)
      Use Laderman rank-7 for each
    - The 1-column block: A (2x2) * B_block (2x1) = C_block (2x1)
      Use schoolbook rank 4 (2*2*1)
    
    Total rank: 7 + 7 + 7 + 4 = 25
    
    Wait, let me reconsider the indexing.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    - U has shape (n*m, rank): indexed by (i, j) for row i, col j of A
    - V has shape (m*p, rank): indexed by (j, k) for row j, col k of B
    - W has shape (p*n, rank): indexed by (k, i) for row k, col i of C
    
    For a 2x2x2 block (A is 2x2, B is 2x2, C is 2x2):
    - i, j in {0, 1} for A
    - j, k in {0, 1} for B
    - k, i in {0, 1} for C
    
    For the full 2x2 * 2x7 problem:
    - A is 2x2: i in {0,1}, j in {0,1}
    - B is 2x7: j in {0,1}, k in {0,...,6}
    - C is 7x2: k in {0,...,6}, i in {0,1}
    
    We can decompose B into blocks:
    - Block 0: B[:, 0:2] (2x2)
    - Block 1: B[:, 2:4] (2x2)
    - Block 2: B[:, 4:6] (2x2)
    - Block 3: B[:, 6:7] (2x1)
    
    For each block b with columns k_start to k_end (exclusive):
    - The sub-multiplication is A (2x2) * B_b (2 x (k_end-k_start)) = C_b ((k_end-k_start) x 2)
    
    For a 2x2x2 block:
    - Local indices: i in {0,1}, j in {0,1}, k_local in {0,1}
    - Global indices: i in {0,1}, j in {0,1}, k = k_start + k_local
    
    For a 2x2x1 block:
    - Local indices: i in {0,1}, j in {0,1}, k_local = 0
    - Global indices: i in {0,1}, j in {0,1}, k = k_start
    
    Let's build this:
    """
    n, m, p = 2, 2, 7
    
    # Define blocks
    blocks = [
        (0, 2),  # cols 0-1: 2x2x2
        (2, 4),  # cols 2-3: 2x2x2
        (4, 6),  # cols 4-5: 2x2x2
        (6, 7),  # col 6: 2x2x1
    ]
    
    # Total rank
    total_rank = 0
    for k_start, k_end in blocks:
        width = k_end - k_start
        if width == 2:
            total_rank += 7  # Laderman
        else:
            total_rank += n * m * width  # Schoolbook
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    for k_start, k_end in blocks:
        width = k_end - k_start
        
        if width == 2:
            # Use Laderman
            U_l, V_l, W_l = _laderman_2x2x2()
            U_l = np.array(U_l, dtype=float)
            V_l = np.array(V_l, dtype=float)
            W_l = np.array(W_l, dtype=float)
            
            # Map local indices to global
            # U: local (i, j) -> global (i, j) for A
            # V: local (j, k_local) -> global (j, k_start + k_local) for B
            # W: local (k_local, i) -> global (k_start + k_local, i) for C
            
            for i in range(n):
                for j in range(m):
                    U[i*m + j, col_offset:col_offset+7] = U_l[i*m + j]
            
            for j in range(m):
                for k_local in range(width):
                    V[j*p + (k_start + k_local), col_offset:col_offset+7] = V_l[j*width + k_local]
            
            for k_local in range(width):
                for i in range(n):
                    W[(k_start + k_local)*n + i, col_offset:col_offset+7] = W_l[k_local*n + i]
            
            col_offset += 7
        
        else:
            # Schoolbook for 2x2x1
            # n=2, m=2, width=1
            # Rank = n*m*width = 4
            for i in range(n):
                for j in range(m):
                    for k_local in range(width):
                        k = k_start + k_local
                        U[i*m + j, col_offset] = 1.0
                        V[j*p + k, col_offset] = 1.0
                        W[k*n + i, col_offset] = 1.0
                        col_offset += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the block decomposition
    U, V, W = _build_block_decomposition(n, m, p)
    
    # Verify the decomposition
    if _verify_fast(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U_school = np.zeros((n*m, rank))
    V_school = np.zeros((m*p, rank))
    W_school = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
