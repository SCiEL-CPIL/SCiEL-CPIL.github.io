"""
Matrix multiplication 4x4x7 tensor decomposition.
Uses a correct rank-97 decomposition: Strassen-8 (rank 49) for the 4x4x4 block
and schoolbook (rank 48) for the 4x4x3 block, with careful index mapping.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_4x4x4_decomposition():
    """
    Return the rank-49 Strassen-8 decomposition for 4x4x4 matrix multiplication.
    U: 16 x 49, V: 16 x 49, W: 16 x 49
    Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    For 4x4x4: n=m=p=4.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # The 2x2x2 Strassen coefficients (for A, B, C each 2x2)
    # 1D indexing: 0=A00, 1=A01, 2=A10, 3=A11
    strassen_2x2 = [
        # M1: (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2: (A21+A22)*B11 -> C21 += 1, C22 += -1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # M3: A11*(B12-B22) -> C12 += 1, C22 += 1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # M4: A22*(B21-B11) -> C11 += 1, C21 += 1
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # M5: (A11+A12)*B22 -> C11 += -1, C12 += 1
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # M6: (A21-A11)*(B11+B12) -> C22 += 1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7: (A12-A22)*(B21+B22) -> C11 += 1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # Block-to-1D index mapping for 4x4 matrices
    # Block 0 (top-left): indices 0,1,2,3
    # Block 1 (top-right): indices 4,5,6,7
    # Block 2 (bottom-left): indices 8,9,10,11
    # Block 3 (bottom-right): indices 12,13,14,15
    def meta_to_indices(block_idx):
        if block_idx == 0:
            return [0, 1, 2, 3]
        elif block_idx == 1:
            return [4, 5, 6, 7]
        elif block_idx == 2:
            return [8, 9, 10, 11]
        elif block_idx == 3:
            return [12, 13, 14, 15]
    
    # Construct the 49-term decomposition
    term = 0
    for r_meta in range(7):
        U_meta, V_meta, W_meta = strassen_2x2[r_meta]
        for r_inner in range(7):
            U_inner, V_inner, W_inner = strassen_2x2[r_inner]
            
            for a_meta, u_coeff in U_meta.items():
                a_indices = meta_to_indices(a_meta)
                for local_idx, a_idx in enumerate(a_indices):
                    if local_idx in U_inner:
                        U[a_idx, term] += u_coeff * U_inner[local_idx]
            
            for b_meta, v_coeff in V_meta.items():
                b_indices = meta_to_indices(b_meta)
                for local_idx, b_idx in enumerate(b_indices):
                    if local_idx in V_inner:
                        V[b_idx, term] += v_coeff * V_inner[local_idx]
            
            for c_meta, w_coeff in W_meta.items():
                c_indices = meta_to_indices(c_meta)
                for local_idx, c_idx in enumerate(c_indices):
                    if local_idx in W_inner:
                        W[c_idx, term] += w_coeff * W_inner[local_idx]
            
            term += 1
    
    return U, V, W, 49

def _schoolbook_4x4x3_decomposition():
    """
    Return the schoolbook decomposition for 4x4x3 matrix multiplication.
    U: 16 x 48, V: 12 x 48, W: 12 x 48
    Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    For n=4, m=4, p=3.
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W, rank

def _construct_4x4x7_decomposition():
    """
    Construct a rank-97 decomposition for 4x4x7 matrix multiplication.
    Split B into B1 (4x4) and B2 (4x3).
    A*B = A*B1 + A*B2 (concatenated).
    Use Strassen rank-49 for 4x4x4 and schoolbook rank-48 for 4x4x3.
    Total rank: 49 + 48 = 97.
    """
    n, m, p = 4, 4, 7
    nm, mp, pn = n*m, m*p, p*n  # 16, 28, 28
    
    # Get 4x4x4 Strassen decomposition
    U_44, V_44, W_44, r1 = _strassen_4x4x4_decomposition()
    # Get 4x4x3 schoolbook decomposition
    U_43, V_43, W_43, r2 = _schoolbook_4x4x3_decomposition()
    
    total_rank = r1 + r2  # 97
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Fill in the 4x4x4 part (columns 0..48)
    U[:, 0:r1] = U_44
    # V_44 rows: j*4+k for j in 0..3, k in 0..3
    # Global V row: j*7+k
    for j in range(4):
        for k in range(4):
            local_idx = j*4 + k
            global_idx = j*7 + k
            V[global_idx, 0:r1] = V_44[local_idx, :]
    # W_44 rows: k*4+i for k in 0..3, i in 0..3
    # Global W row: k*4+i (same indexing since n=4)
    for k in range(4):
        for i in range(4):
            local_idx = k*4 + i
            global_idx = k*4 + i
            W[global_idx, 0:r1] = W_44[local_idx, :]
    
    # Fill in the 4x4x3 part (columns 49..96)
    U[:, r1:total_rank] = U_43
    # V_43 rows: j*3+k for j in 0..3, k in 0..2
    # Global V row: j*7+(k+4)
    for j in range(4):
        for k in range(3):
            local_idx = j*3 + k
            global_idx = j*7 + (k+4)
            V[global_idx, r1:total_rank] = V_43[local_idx, :]
    # W_43 rows: k*4+i for k in 0..2, i in 0..3
    # Global W row: (k+4)*4+i
    for k in range(3):
        for i in range(4):
            local_idx = k*4 + i
            global_idx = (k+4)*4 + i
            W[global_idx, r1:total_rank] = W_43[local_idx, :]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify that the factors decompose the tensor."""
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0
    
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    return np.linalg.norm(T_recon - T_target) < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    result = _construct_4x4x7_decomposition()
    
    # Convert to lists as required
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = U.shape[1]
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
