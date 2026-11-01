"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # 10s margin before 150s kill

def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly represents 2x2 * 2x6 matrix multiplication.
    
    The tensor T[i,j,k] = 1 if j == k%2 else 0. We verify by checking that
    for each block b (columns 2b, 2b+1 of B), the 2x2 multiplication is correct.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify using random integer tests
    import random
    rng = random.Random(12345)
    num_tests = 200
    for _ in range(num_tests):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B
        C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
        
        # Compute using decomposition: C[i,j] = sum_r U[i*m+j... wait
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # C[i,j] = sum_r U[i*m + ?]... 
        # Actually the tensor decomposition gives:
        # T[i,j,k] = sum_r U[i*m+k... no
        # T[i,j,k] where i is row of A (0..n-1), j is row of B (0..m-1), k is col of B (0..p-1)
        # U has shape (n*m, rank): indexed by (i,j) -> U[i*m+j]
        # V has shape (m*p, rank): indexed by (j,k) -> V[j*p+k]  
        # W has shape (p*n, rank): indexed by (k,i) -> W[k*n+i]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # Wait, that's not right either. Let me think again.
        
        # Standard tensor decomposition for matrix multiplication:
        # C = A * B where A is n×m, B is m×p, C is n×p
        # T[i,j,k] = 1 (the multiplication tensor)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor)
        
        # Wait no. For standard matrix multiplication C = AB:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = 1 for all i,j,k in their ranges.
        # But that's only if we're doing n×m times m×p.
        
        # Here n=2, m=2, p=6. So A is 2×2, B is 2×6, C is 2×6.
        # C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0,...,5}
        # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,5}
        
        # So the decomposition should satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Let me verify this way:
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U_arr[i * m + j, r]
                    if u_val != 0:
                        for k in range(p):
                            v_val = V_arr[j * p + k, r]
                            if v_val != 0:
                                w_val = W_arr[k * n + i, r]
                                if w_val != 0:
                                    T_decomp[i, j, k] += u_val * v_val * w_val
        
        T_target = np.ones((n, m, p))
        if not np.allclose(T_decomp, T_target, atol=1e-9):
            return False
    
    return True

def _strassen_decomposition():
    """Block-diagonal Strassen decomposition with rank 21.
    
    The tensor T[i,j,k] for 2×2 × 2×6 multiplication is:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,5}
    
    This decomposes into 3 independent 2×2 multiplications:
    - Block 0: columns 0,1 of B -> C columns 0,1
    - Block 1: columns 2,3 of B -> C columns 2,3  
    - Block 2: columns 4,5 of B -> C columns 4,5
    
    Each 2×2 multiplication uses Strassen's rank-7 algorithm.
    Total rank = 3 * 7 = 21.
    """
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # (4, 21)
    V = [[0] * rank for _ in range(m * p)]  # (12, 21)
    W = [[0] * rank for _ in range(p * n)]  # (12, 21)
    
    # Strassen 7 factors for 2x2 multiplication.
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # For each block b, B local indices: 0=B00, 1=B01, 2=B10, 3=B11
    #   Global B row j, col k: V[j*p + k]
    # For each block b, C local indices: 0=C00, 1=C01, 2=C10, 3=C11
    #   Global C row i, col k: W[k*n + i]
    
    # M1 = A00 + A11
    # M2 = A10 + A11
    # M3 = A00
    # M4 = A11
    # M5 = A00 + A01
    # M6 = A10 - A00
    # M7 = A01 - A11
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # V coefficients: for each M_k, list of (j_local, k_local, coeff)
    # j_local: row of B (0 or 1), k_local: col within block (0 or 1)
    # M1: B00 + B11 -> (0,0,1), (1,1,1)
    # M2: B00 -> (0,0,1)
    # M3: B01 - B11 -> (0,1,1), (1,1,-1)
    # M4: B10 - B00 -> (1,0,1), (0,0,-1)
    # M5: B11 -> (1,1,1)
    # M6: B00 + B01 -> (0,0,1), (0,1,1)
    # M7: B10 + B11 -> (1,0,1), (1,1,1)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(0, 0, 1)],               # M2
        [(0, 1, 1), (1, 1, -1)],  # M3
        [(1, 0, 1), (0, 0, -1)],  # M4
        [(1, 1, 1)],               # M5
        [(0, 0, 1), (0, 1, 1)],   # M6
        [(1, 0, 1), (1, 1, 1)],   # M7
    ]
    
    # W coefficients: for each M_k, list of (i, k_local, coeff)
    # i: row of C (0 or 1), k_local: col within block (0 or 1)
    # M1: C00(1), C11(1) -> (0,0,1), (1,1,1)
    # M2: C10(1), C11(-1) -> (1,0,1), (1,1,-1)
    # M3: C01(1), C11(1) -> (0,1,1), (1,1,1)
    # M4: C00(1), C10(1) -> (0,0,1), (1,0,1)
    # M5: C00(-1), C01(1) -> (0,0,-1), (0,1,1)
    # M6: C11(1) -> (1,1,1)
    # M7: C00(1) -> (0,0,1)
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(1, 0, 1), (1, 1, -1)],  # M2
        [(0, 1, 1), (1, 1, 1)],   # M3
        [(0, 0, 1), (1, 0, 1)],   # M4
        [(0, 0, -1), (0, 1, 1)],  # M5
        [(1, 1, 1)],              # M6
        [(0, 0, 1)]               # M7
    ]
    
    for b in range(3):  # block index
        for k in range(7):  # Strassen factor index
            col = b * 7 + k
            
            # Fill U
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
            
            # Fill V
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * b + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
            
            # Fill W
            for i, k_local, val in W_coeffs[k]:
                global_k = 2 * b + k_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank 24."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook_decomposition()
    
    result = _strassen_decomposition()
    
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        return _schoolbook_decomposition()
    
    return result

# EVOLVE-BLOCK-END
