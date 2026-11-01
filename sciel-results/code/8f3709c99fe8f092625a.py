"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_decomp():
    """
    Return the rank-7 Strassen decomposition for 2x2 @ 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
    - U[i*2+j, c] for input A[i,j]
    - V[j*2+k, c] for input B[j,k]
    - W[k*2+i, c] for output C[i,k]
    """
    # Strassen's 7 products for 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4,         M1-M2+M3+M6]]
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7  (but in flat: j*2+k)
    #   B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx2, B[1,1]=idx3 in V's local indexing
    # Output C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 (k*2+i)
    #   C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3 in W's local indexing
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: contributes to C[0,0] and C[1,1]
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[3, 0] = 1  # e, h
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: C[1,0] and C[1,1]
    U[2, 1] = 1; U[3, 1] = 1  # c, d
    V[0, 1] = 1  # e
    W[2, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: C[0,1] and C[1,1]
    U[0, 2] = 1  # a
    V[1, 2] = 1; V[3, 2] = -1  # f, h
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: C[0,0] and C[1,0]
    U[3, 3] = 1  # d
    V[2, 3] = 1; V[0, 3] = -1  # g, e
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0], C[1,0]
    
    # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: C[0,0] (negative) and C[0,1]
    U[0, 4] = 1; U[1, 4] = 1  # a, b
    V[3, 4] = 1  # h
    W[0, 4] = -1; W[1, 4] = 1  # C[0,0], C[0,1]
    
    # M6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: C[1,1]
    U[2, 5] = 1; U[0, 5] = -1  # c, a
    V[0, 5] = 1; V[1, 5] = 1  # e, f
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: C[0,0]
    U[1, 6] = 1; U[3, 6] = -1  # b, d
    V[2, 6] = 1; V[3, 6] = 1  # g, h
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _construct_strassen_based_decomp(n, m, p):
    """
    Construct a decomposition for 2x3 @ 3x6 using Strassen for 2x2@2x2 blocks.
    
    Split middle index: 3 = 2 + 1
    A (2x3) = [A_left (2x2), A_right (2x1)]
    B (3x6) = [B_top (2x6); B_bottom (1x6)]
    
    Result = A_left @ B_top + A_right @ B_bottom
    
    A_left @ B_top: 2x2 @ 2x6 = [A_left@B1, A_left@B2, A_left@B3] where each Bi is 2x2
    Each 2x2@2x2 uses Strassen (rank 7). Total: 21.
    
    A_right @ B_bottom: 2x1 @ 1x6, rank 2.
    
    Total rank: 23.
    """
    rank = 23
    U = np.zeros((n * m, rank))  # 6 x 23
    V = np.zeros((m * p, rank))  # 18 x 23
    W = np.zeros((p * n, rank))  # 12 x 23
    
    col = 0
    
    # For each of the 3 Strassen blocks (covering columns 0-1, 2-3, 4-5 of B)
    for block in range(3):
        k_offset = block * 2  # starting column in B (and output)
        
        # Get Strassen decomposition
        U_s, V_s, W_s = _strassen_2x2x2_decomp()
        
        # Map Strassen indices to global indices
        # A_left is A[:, 0:2], so A[i, j] for i in {0,1}, j in {0,1}
        # Global U index: i * m + j = i * 3 + j
        # Strassen U local index: i * 2 + j
        for i in range(2):
            for j in range(2):
                local_idx = i * 2 + j
                global_idx = i * m + j  # i*3 + j
                U[global_idx, col:col+7] = U_s[local_idx, :]
        
        # B_top is B[0:2, k_offset:k_offset+2], so B[j, k] for j in {0,1}, k in {k_offset, k_offset+1}
        # Global V index: j * p + k = j * 6 + k
        # Strassen V local index: j * 2 + (k - k_offset)
        for j in range(2):
            for dk in range(2):
                local_idx = j * 2 + dk
                global_idx = j * p + (k_offset + dk)
                V[global_idx, col:col+7] = V_s[local_idx, :]
        
        # Output C[i, k] for i in {0,1}, k in {k_offset, k_offset+1}
        # Global W index: k * n + i = k * 2 + i
        # Strassen W local index: dk * 2 + i where dk = k - k_offset
        for i in range(2):
            for dk in range(2):
                local_idx = dk * 2 + i
                global_idx = (k_offset + dk) * n + i
                W[global_idx, col:col+7] = W_s[local_idx, :]
        
        col += 7
    
    # Now handle A_right @ B_bottom: 2x1 @ 1x6
    # A_right = A[:, 2] (2x1), B_bottom = B[2, :] (1x6)
    # Result: C[i, k] += A[i, 2] * B[2, k] for all i in {0,1}, k in {0..5}
    # This is rank 2: one term per row i
    for i in range(2):
        # U: A[i, 2] -> index i * m + 2 = i * 3 + 2
        U[i * m + 2, col] = 1
        # V: B[2, k] for all k -> index 2 * p + k = 12 + k
        for k in range(p):
            V[2 * p + k, col] = 1
        # W: C[i, k] for all k -> index k * n + i = k * 2 + i
        for k in range(p):
            W[k * n + i, col] = 1
        col += 1
    
    return U, V, W


def _tensor_entries(U, V, W, n, m, p):
    """
    Given U (n*m x rank), V (m*p x rank), W (p*n x rank),
    compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    as a 3D array.
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        Uc = U3[:, :, c]
        Vc = V3[:, :, c]
        Wc = W3[:, :, c]
        for j in range(m):
            T[:, j, :] += (Uc[:, j, np.newaxis] * Vc[j, np.newaxis, :] * Wc.T)
    return T


def _target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)


def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    T = _tensor_entries(U, V, W, n, m, p)
    T_target = _target_tensor(n, m, p)
    err = np.max(np.abs(T - T_target))
    return err < tol, err


def _is_half_integer(arr, tol=1e-9):
    doubled = 2 * arr
    rounded = np.round(doubled)
    return np.max(np.abs(doubled - rounded)) < tol


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140
    
    # Try Strassen-based decomposition first (rank 23)
    U, V, W = _construct_strassen_based_decomp(n, m, p)
    valid, err = _validate_decomposition(U, V, W, n, m, p)
    if valid:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    rank = n * m * p
    U_sb = np.zeros((n * m, rank))
    V_sb = np.zeros((m * p, rank))
    W_sb = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }

# EVOLVE-BLOCK-END
