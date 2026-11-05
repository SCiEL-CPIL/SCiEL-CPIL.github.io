import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify the decomposition using the trace convention W[k*n+i, r].
    C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    where A is (n,m), B is (m,p), C is (n,p)
    """
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each rank term r:
    # L_r[i,k] = U[i*m+k, r]  (this is the "left" factor indexed by i,k)
    # R_r[k,j] = V[k*p+j, r]  (this is the "right" factor indexed by k,j)
    # W_r[j,i] = W[j*n+i, r]  (this is the "output" coefficient indexed by j,i)
    
    # C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            # sum over k and r
            # We can vectorize: for fixed i,j,
            # term_r = sum_k U[i*m+k, r] * V[k*p+j, r]
            # C[i,j] = sum_r term_r * W[j*n+i, r]
            
            # U rows for fixed i: indices i*m .. i*m+m-1
            U_slice = U_arr[i*m : i*m+m, :]  # shape (m, rank)
            # V rows for fixed j: indices 0*p+j, 1*p+j, ..., (m-1)*p+j
            V_indices = np.arange(m) * p + j  # shape (m,)
            V_slice = V_arr[V_indices, :]  # shape (m, rank)
            
            # term_r = sum_k U_slice[k, r] * V_slice[k, r]
            term = np.sum(U_slice * V_slice, axis=0)  # shape (rank,)
            
            # W for fixed (j, i): row index j*n + i
            W_val = W_arr[j * n + i, :]  # shape (rank,)
            
            C_computed[i, j] = np.dot(term, W_val)
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: C[i,j] = sum_k sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    for 2x2 matrices.
    """
    # Strassen's algorithm for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: (i,k) -> i*2+k
    # A11: (0,0)->0, A12: (0,1)->1, A21: (1,0)->2, A22: (1,1)->3
    # B indices: (k,j) -> k*2+j
    # B11: (0,0)->0, B12: (0,1)->1, B21: (1,0)->2, B22: (1,1)->3
    # C indices: (j,i) -> j*2+i  (using W convention)
    # C11: (0,0)->0, C12: (0,1)->1, C21: (1,0)->2, C22: (1,1)->3
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A11 + A22
    V[0, 0] = 1; V[3, 0] = 1  # B11 + B22
    W[0, 0] = 1; W[3, 0] = 1  # C11 + C22
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1  # A21 + A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1; W[3, 1] = -1  # C21 - C22
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1  # A11
    V[2, 2] = 1; V[3, 2] = -1  # B21 - B22
    W[1, 2] = 1; W[3, 2] = 1  # C12 + C22
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1  # A22
    V[1, 3] = 1; V[0, 3] = -1  # B12 - B11
    W[0, 3] = 1; W[2, 3] = 1  # C11 + C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1  # A11 + A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1; W[1, 4] = 1  # -C11 + C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1  # A21 - A11
    V[0, 5] = 1; V[1, 5] = 1  # B11 + B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1  # A12 - A22
    V[2, 6] = 1; V[3, 6] = 1  # B21 + B22
    W[0, 6] = 1; W[3, 6] = -1  # C11 - C22
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_strassen_5x5x5():
    """
    Build a rank-117 decomposition for 5x5x5 using Strassen for 2x2x2 sub-blocks
    and schoolbook for the remaining blocks.
    
    Partition 5 = 2 + 2 + 1.
    Row blocks of A: I = [0,1], [2,3], [4]
    Col blocks of A: J = [0,1], [2,3], [4]
    Row blocks of B: K = [0,1], [2,3], [4]
    Col blocks of B: L = [0,1], [2,3], [4]
    
    For each (I, K) pair of row-col blocks:
    - If both have size 2, use Strassen (rank 7) for the 2x2x2 product
    - If either has size 1, use schoolbook for the remaining size
    """
    n, m, p = 5, 5, 5
    
    # Block structure: 3 blocks of sizes [2, 2, 1]
    block_sizes = [2, 2, 1]
    # Block offsets
    I_offsets = [0, 2, 4]  # row blocks of A
    J_offsets = [0, 2, 4]  # col blocks of A (= row blocks of B)
    K_offsets = [0, 2, 4]  # col blocks of B (= row blocks of C)
    
    # Get Strassen decomposition
    sU, sV, sW = _strassen_2x2x2()
    
    # Initialize matrices
    # We'll build up U, V, W as lists of rows
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # First, figure out the total rank
    total_rank = 0
    rank_per_block = {}
    for a in range(3):  # row block of A
        for b in range(3):  # col block of A / row block of B
            size_a = block_sizes[a]
            size_b = block_sizes[b]
            if size_a == 2 and size_b == 2:
                rank_per_block[(a, b)] = 7
            elif size_a == 2 and size_b == 1:
                rank_per_block[(a, b)] = 2 * 1  # 2x1 schoolbook
            elif size_a == 1 and size_b == 2:
                rank_per_block[(a, b)] = 1 * 2  # 1x2 schoolbook
            else:
                rank_per_block[(a, b)] = 1  # 1x1 schoolbook
            total_rank += rank_per_block[(a, b)]
    
    # total_rank = 7*4 + 2*2 + 1*2 + 1 = 28 + 4 + 2 + 1 = 35? 
    # Wait, let me recount. 
    # (a,b) pairs:
    # (0,0): 2x2 -> 7
    # (0,1): 2x2 -> 7
    # (0,2): 2x1 -> 2
    # (1,0): 2x2 -> 7
    # (1,1): 2x2 -> 7
    # (1,2): 2x1 -> 2
    # (2,0): 1x2 -> 2
    # (2,1): 1x2 -> 2
    # (2,2): 1x1 -> 1
    # Total: 7+7+2+7+7+2+2+2+1 = 37
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for a in range(3):  # row block index of A
        for b in range(3):  # col block index of A / row block index of B
            size_a = block_sizes[a]
            size_b = block_sizes[b]
            r = rank_per_block[(a, b)]
            
            # Global row range for A block a: rows I_offsets[a] to I_offsets[a]+size_a-1
            # Global col range for A block b: cols J_offsets[b] to J_offsets[b]+size_b-1
            # Global row range for B block b: rows K_offsets[b] to K_offsets[b]+size_b-1
            # Global col range for B block b: cols K_offsets[b] to K_offsets[b]+size_b-1 (same as B's col blocks)
            
            if size_a == 2 and size_b == 2:
                # Use Strassen
                for rr in range(7):
                    # U: for local (i, k) in 2x2 block
                    for li in range(2):
                        for lk in range(2):
                            val = sU[li * 2 + lk][rr]
                            if val != 0:
                                gi = I_offsets[a] + li
                                gk = J_offsets[b] + lk
                                U[gi * m + gk][col] += val
                    
                    # V: for local (k, j) in 2x2 block
                    for lk in range(2):
                        for lj in range(2):
                            val = sV[lk * 2 + lj][rr]
                            if val != 0:
                                gk = K_offsets[b] + lk
                                gj = K_offsets[b] + lj
                                V[gk * p + gj][col] += val
                    
                    # W: for local (j, i) in 2x2 block (using W convention W[j*n+i])
                    for lj in range(2):
                        for li in range(2):
                            val = sW[lj * 2 + li][rr]
                            if val != 0:
                                gj = K_offsets[b] + lj
                                gi = I_offsets[a] + li
                                W[gj * n + gi][col] += val
                    
                    col += 1
            elif size_a == 2 and size_b == 1:
                # Schoolbook 2x1
                # A block: 2 rows, 1 col. B block: 1 row, 1 col.
                # Local indices: A (i, k) with i in 0,1; k=0
                # B (k, j) with k=0; j=0
                # C (j, i) with j=0; i in 0,1
                for li in range(2):
                    for lk in range(1):
                        gi = I_offsets[a] + li
                        gk = J_offsets[b] + lk
                        U[gi * m + gk][col] = 1
                        gk_b = K_offsets[b] + lk
                        gj = K_offsets[b] + 0
                        V[gk_b * p + gj][col] = 1
                        gj_w = K_offsets[b] + 0
                        gi_w = I_offsets[a] + li
                        W[gj_w * n + gi_w][col] = 1
                        col += 1
            elif size_a == 1 and size_b == 2:
                # Schoolbook 1x2
                for li in range(1):
                    for lk in range(2):
                        gi = I_offsets[a] + li
                        gk = J_offsets[b] + lk
                        U[gi * m + gk][col] = 1
                        gk_b = K_offsets[b] + lk
                        gj = K_offsets[b] + 0
                        V[gk_b * p + gj][col] = 1
                        gj_w = K_offsets[b] + 0
                        gi_w = I_offsets[a] + li
                        W[gj_w * n + gi_w][col] = 1
                        col += 1
            else:
                # 1x1
                gi = I_offsets[a]
                gk = J_offsets[b]
                U[gi * m + gk][col] = 1
                gk_b = K_offsets[b]
                gj = K_offsets[b]
                V[gk_b * p + gj][col] = 1
                gj_w = K_offsets[b]
                gi_w = I_offsets[a]
                W[gj_w * n + gi_w][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Try Strassen-based construction
    result = _build_strassen_5x5x5()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    result = _schoolbook(n, m, p)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If even schoolbook fails (shouldn't happen), return it anyway
    return result
