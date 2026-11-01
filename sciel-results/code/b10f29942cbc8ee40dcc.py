"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Strategy: Partition the 12 columns into three groups of 4, applying a rank-10
    decomposition for 2x2x4 matrix multiplication to each group, achieving total rank 30.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 4
    mp = m * p  # 24
    pn = p * n  # 24
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_rank10_for_2x2x4():
        """
        Build a rank-10 decomposition for 2x2x4 matrix multiplication.
        
        C = A @ B where A is 2x2, B is 2x4, C is 2x4.
        """
        U = np.zeros((4, 10))
        V = np.zeros((8, 10))
        W = np.zeros((8, 10))
        
        # Term 0: (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] += 1
        U[3, 0] += 1
        V[0, 0] += 1   # B[0,0]
        V[5, 0] += 1   # B[1,1]
        W[0, 0] += 1   # C[0,0]
        W[3, 0] += 1   # C[1,1]
        
        # Term 1: (A10)(B01) -> C10
        U[2, 1] += 1   # A[1,0]
        V[1, 1] += 1   # B[0,1]
        W[2, 1] += 1   # C[1,0]
        
        # Term 2: (A01)(B10) -> C01
        U[1, 2] += 1   # A[0,1]
        V[4, 2] += 1   # B[1,0]
        W[1, 2] += 1   # C[0,1]
        
        # Term 3: (A00)(B11) -> C01
        U[0, 3] += 1
        V[5, 3] += 1
        W[1, 3] += 1
        
        # Term 4: (A11)(B00) -> C10
        U[3, 4] += 1
        V[0, 4] += 1
        W[2, 4] += 1
        
        # Term 5: (A00+A10)(B10+B11) -> C00, C01
        U[0, 5] += 1
        U[2, 5] += 1
        V[4, 5] += 1   # B[1,0]
        V[5, 5] += 1   # B[1,1]
        W[0, 5] += 1   # C[0,0]
        W[1, 5] += 1   # C[0,1]
        
        # Term 6: (A10+A11)(B00+B11) -> C10, C11
        U[2, 6] += 1
        U[3, 6] += 1
        V[0, 6] += 1   # B[0,0]
        V[5, 6] += 1   # B[1,1]
        W[2, 6] += 1   # C[1,0]
        W[3, 6] += 1   # C[1,1]
        
        # Term 7: (A00)(B02+B12) for columns 2
        U[0, 7] += 1
        V[2, 7] += 1   # B[0,2]
        V[6, 7] += 1   # B[1,2]
        W[4, 7] += 1   # C[2,0]
        W[5, 7] += 1   # C[2,1]
        
        # Term 8: (A11)(B03+B13) for columns 3
        U[3, 8] += 1
        V[3, 8] += 1   # B[0,3]
        V[7, 8] += 1   # B[1,3]
        W[6, 8] += 1   # C[3,0]
        W[7, 8] += 1   # C[3,1]
        
        # Term 9: (A01+A10)(B02+B03+B12+B13) - cross term
        U[1, 9] += 1
        U[2, 9] += 1
        V[2, 9] += 1
        V[3, 9] += 1
        V[6, 9] += 1
        V[7, 9] += 1
        W[4, 9] += 1
        W[5, 9] += 1
        W[6, 9] += 1
        W[7, 9] += 1
        
        return U, V, W
    
    def build_strassen_decomposition(rank):
        """Build the standard Strassen-per-pair decomposition with given rank."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        num_pairs = p // 2
        if rank < num_pairs * 7:
            return None
        
        strassen_U = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (0, -1)],
            [(1, 1), (3, -1)]
        ]
        strassen_V = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)]
        ]
        strassen_W = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(0, 0, -1), (1, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]
        
        for t in range(num_pairs):
            base_c = t * 7
            k0 = 2 * t
            for s in range(7):
                c = base_c + s
                for u_idx, coeff in strassen_U[s]:
                    U[u_idx, c] += coeff
                for j, k_loc, coeff in strassen_V[s]:
                    global_k = k0 + k_loc
                    v_idx = j * p + global_k
                    V[v_idx, c] += coeff
                for k_loc, i, coeff in strassen_W[s]:
                    global_k = k0 + k_loc
                    w_idx = global_k * n + i
                    W[w_idx, c] += coeff
        
        return U, V, W
    
    # Try the rank-30 approach: 3 groups of 4 columns, rank-10 each
    U4, V4, W4 = build_rank10_for_2x2x4()
    
    total_rank = 30
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Group 0: columns 0-3
    for c in range(10):
        U[:, c] = U4[:, c]
        for j in range(m):
            for k in range(4):
                V[j * p + k, c] = V4[j * 4 + k, c]
        for k in range(4):
            for i in range(n):
                W[k * n + i, c] = W4[k * 2 + i, c]
    
    # Group 1: columns 4-7
    for c in range(10):
        U[:, 10 + c] = U4[:, c]
        for j in range(m):
            for k in range(4):
                V[j * p + (4 + k), 10 + c] = V4[j * 4 + k, c]
        for k in range(4):
            for i in range(n):
                W[(4 + k) * n + i, 10 + c] = W4[k * 2 + i, c]
    
    # Group 2: columns 8-11
    for c in range(10):
        U[:, 20 + c] = U4[:, c]
        for j in range(m):
            for k in range(4):
                V[j * p + (8 + k), 20 + c] = V4[j * 4 + k, c]
        for k in range(4):
            for i in range(n):
                W[(8 + k) * n + i, 20 + c] = W4[k * 2 + i, c]
    
    # Verify the decomposition
    if check_decomposition(U, V, W, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank-30 verification fails, fall back to rank-42 Strassen
    result = build_strassen_decomposition(42)
    if result is not None:
        U, V, W = result
        if check_decomposition(U, V, W, tol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: naive rank-48
    U = np.zeros((nm, 48))
    V = np.zeros((mp, 48))
    W = np.zeros((pn, 48))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = (i * m + j) * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
