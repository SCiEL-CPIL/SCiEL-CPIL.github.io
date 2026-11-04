import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W matrices of shapes (4,7), (4,7), (4,7)
    where the tensor T[a,b,c] = 1 iff a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}
    """
    U = np.array([
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1],  # M7: A12 - A22
    ], dtype=float).T  # shape (4, 7)
    
    V = np.array([
        [1, 1, 0, 0],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [-1, 0, 1, 0],  # M4: B21 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1],   # M7: B21 + B22
    ], dtype=float).T  # shape (4, 7)
    
    W = np.array([
        [1, 0, 0, 1],   # C11: M1 + M4 - M5 + M7
        [0, 0, 1, 0],   # C12: M3 + M5
        [0, 1, 0, 0],   # C21: M2 + M4
        [1, -1, 1, 1],  # C22: M1 - M2 + M3 + M6
    ], dtype=float).T  # shape (4, 7)
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Schoolbook rank-4 decomposition for 2x2 * 2x1 matrix multiplication.
    Returns U, V, W matrices of shapes (4,4), (2,4), (2,4)
    """
    # A is 2x2, B is 2x1, C is 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # a indices: 0=A00, 1=A01, 2=A10, 3=A11
    # b indices: 0=B0, 1=B1
    # c indices: 0=C0, 1=C1
    
    U = np.eye(4, dtype=float) # Each term uses one A element
    V = np.array([
        [1, 0, 0, 0], # B0
        [0, 1, 1, 1]  # B1
    ], dtype=float).T # shape (2, 4)
    
    W = np.array([
        [1, 1, 0, 0], # C0 gets A00*B0, A01*B1
        [0, 0, 1, 1]  # C1 gets A10*B0, A11*B1
    ], dtype=float).T # shape (2, 4)
    
    return U, V, W

def _embed_2x2x2_in_2x2x5(U_sub, V_sub, W_sub, j_offset, k_offset, p_full):
    """
    Embed a 2x2x2 decomposition into a 2x2x5 tensor.
    j_offset: 0 or 1 (which j index in the 2x2 block)
    k_offset: offset for k index (0, 1, 2, 3, 4)
    p_full: 5
    """
    # This function is not quite right because the sub-block 2x2x2 uses specific k values.
    # Let's just construct the full 2x2x5 decomposition directly.
    pass

def _construct_2x2x5():
    """
    Construct decomposition for 2x2 * 2x5.
    Split B into B1 (2x2) and B2 (2x3).
    Rank = Rank(2x2x2) + Rank(2x2x3).
    Rank(2x2x2) = 7.
    Rank(2x2x3) = Rank(2x2x2) + Rank(2x2x1) = 7 + 4 = 11.
    Total rank = 18.
    """
    n, m, p = 2, 2, 5
    rank = 18
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Part 1: 2x2x2 (k=0,1)
    U1, V1, W1 = _strassen_2x2x2()
    # Map indices
    # a = i*m + j -> i*2 + j. Same as sub-block.
    # b = j*p + k -> j*5 + k.
    #    For j=0, k=0,1 -> b=0,1.
    #    For j=1, k=0,1 -> b=5,6.
    # c = k*n + i -> k*2 + i.
    #    For k=0, i=0,1 -> c=0,1.
    #    For k=1, i=0,1 -> c=2,3.
    
    for s in range(7):
        # U
        for i in range(2):
            for j in range(2):
                a = i*2 + j
                U[a, s] = U1[i*2+j, s]
        
        # V
        for j in range(2):
            for k in range(2):
                b = j*5 + k
                V[b, s] = V1[j*2+k, s]
        
        # W
        for k in range(2):
            for i in range(2):
                c = k*2 + i
                W[c, s] = W1[k*2+i, s]
                
    # Part 2: 2x2x3 (k=2,3,4)
    # Split into 2x2x2 (k=2,3) and 2x2x1 (k=4)
    
    # Sub-part 2a: 2x2x2 (k=2,3)
    U2a, V2a, W2a = _strassen_2x2x2()
    # Map indices
    # a = i*2 + j. Same.
    # b = j*5 + k.
    #    For j=0, k=2,3 -> b=2,3.
    #    For j=1, k=2,3 -> b=7,8.
    # c = k*2 + i.
    #    For k=2, i=0,1 -> c=4,5.
    #    For k=3, i=0,1 -> c=6,7.
    
    for s in range(7):
        for i in range(2):
            for j in range(2):
                a = i*2 + j
                U[a, 7+s] = U2a[i*2+j, s]
        
        for j in range(2):
            for k in range(2):
                k_global = 2 + k
                b = j*5 + k_global
                V[b, 7+s] = V2a[j*2+k, s]
        
        for k in range(2):
            k_global = 2 + k
            for i in range(2):
                c = k_global*2 + i
                W[c, 7+s] = W2a[k*2+i, s]
                
    # Sub-part 2b: 2x2x1 (k=4)
    U2b, V2b, W2b = _schoolbook_2x2x1()
    # Map indices
    # a = i*2 + j. Same.
    # b = j*5 + k.
    #    For j=0, k=4 -> b=4.
    #    For j=1, k=4 -> b=9.
    # c = k*2 + i.
    #    For k=4, i=0,1 -> c=8,9.
    
    for s in range(4):
        for i in range(2):
            for j in range(2):
                a = i*2 + j
                U[a, 14+s] = U2b[i*2+j, s]
        
        for j in range(2):
            k_global = 4
            b = j*5 + k_global
            V[b, 14+s] = V2b[j, s] # V2b is (2,4), j is 0 or 1
            
        for i in range(2):
            k_global = 4
            c = k_global*2 + i
            W[c, 14+s] = W2b[i, s] # W2b is (2,4), i is 0 or 1

    return U, V, W

def _construct_2x4x5():
    """
    Construct decomposition for 2x4 * 4x5.
    Split into two 2x2x5 blocks.
    """
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Block 1: j=0,1
    U1, V1, W1 = _construct_2x2x5()
    # a = i*4 + j. j in 0,1.
    # b = j*5 + k. j in 0,1.
    # c = k*2 + i.
    
    for s in range(18):
        for i in range(2):
            for j in range(2):
                a = i*4 + j
                U[a, s] = U1[i*2+j, s]
        
        for j in range(2):
            for k in range(5):
                b = j*5 + k
                V[b, s] = V1[j*5+k, s]
        
        for k in range(5):
            for i in range(2):
                c = k*2 + i
                W[c, s] = W1[k*2+i, s]
                
    # Block 2: j=2,3
    U2, V2, W2 = _construct_2x2x5()
    # a = i*4 + j. j in 2,3.
    # b = j*5 + k. j in 2,3.
    # c = k*2 + i.
    
    for s in range(18):
        for i in range(2):
            for j in range(2):
                j_global = 2 + j
                a = i*4 + j_global
                U[a, 18+s] = U2[i*2+j, s]
        
        for j in range(2):
            j_global = 2 + j
            for k in range(5):
                b = j_global*5 + k
                V[b, 18+s] = V2[j*5+k, s]
        
        for k in range(5):
            for i in range(2):
                c = k*2 + i
                W[c, 18+s] = W2[k*2+i, s]
                
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    # Check the n*m*p "one" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                if abs(val - 1.0) > tol:
                    return False

    # Check that all other positions are zero
    one_positions = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                one_positions.add((i * m + j, j * p + k, k * n + i))

    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                if (a, b, c) in one_positions:
                    continue
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    U, V, W = _construct_2x4x5()
    
    if _check_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_sb = np.zeros((n*m, rank), dtype=float)
        V_sb = np.zeros((m*p, rank), dtype=float)
        W_sb = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1.0
                    V_sb[j*p+k, col] = 1.0
                    W_sb[k*n+i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
