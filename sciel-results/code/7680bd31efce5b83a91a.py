"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition of the matmul tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _strassen_2x2_2x2():
    """Return the 7 Strassen products for 2x2x2 matmul.
    Each product is (U_coeff, V_coeff, W_coeff) where each coeff is a dict mapping
    (offset_i, offset_j) -> value etc.
    
    Strassen formulas for C = A*B where A,B are 2x2:
    Let A = [[a11, a12],[a21, a22]], B = [[b11, b12],[b21, b22]]
    
    M1 = (a11+a22)*(b11+b22)
    M2 = (a21+a22)*b11
    M3 = a11*(b12-b22)
    M4 = a22*(b21-b11)
    M5 = (a11+a12)*b22
    M6 = (a21-a11)*(b11+b12)
    M7 = (a12-a22)*(b21+b22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    products = []
    
    # M1 = (a11+a22)*(b11+b22)
    # U: {(0,0):1, (1,1):1}, V: {(0,0):1, (0,1):1}, W: {(0,0):1, (1,1):1}  -- wait, W is for output
    # Actually let me think in terms of the tensor indices.
    # For a 2x2x2 block, the tensor entry T[i,j,k] = 1 for i,j,k in {0,1}
    # U[i,j,c] contributes to input A[i,j], V[j,k,c] to B[j,k], W[k,i,c] to output C[i,k]
    
    # M1: A part = a11+a22 = A[0,0]+A[1,1], B part = b11+b22 = B[0,0]+B[1,1]
    # Output: C11 += 1, C22 += 1 (i.e., C[0,0] and C[1,1])
    # U: {(0,0):1, (1,1):1}, V: {(0,0):1, (1,1):1}, W: {(0,0):1, (1,1):1}
    products.append((
        {(0,0):1, (1,1):1},   # U
        {(0,0):1, (1,1):1},   # V
        {(0,0):1, (1,1):1}    # W
    ))
    
    # M2: A part = a21+a22 = A[1,0]+A[1,1], B part = b11 = B[0,0]
    # Output: C21 += 1 (C[1,0]), C22 += -1 (C[1,1])
    products.append((
        {(1,0):1, (1,1):1},   # U
        {(0,0):1},            # V
        {(0,1):1, (1,1):-1}   # W
    ))
    
    # M3: A part = a11 = A[0,0], B part = b12-b22 = B[0,1]-B[1,1]
    # Output: C12 += 1 (C[0,1]), C22 += 1 (C[1,1])
    products.append((
        {(0,0):1},            # U
        {(0,1):1, (1,1):-1},  # V
        {(1,0):1, (1,1):1}    # W
    ))
    
    # M4: A part = a22 = A[1,1], B part = b21-b11 = B[1,0]-B[0,0]
    # Output: C11 += 1 (C[0,0]), C21 += 1 (C[1,0])
    products.append((
        {(1,1):1},            # U
        {(1,0):1, (0,0):-1},  # V
        {(0,0):1, (0,1):1}    # W
    ))
    
    # M5: A part = a11+a12 = A[0,0]+A[0,1], B part = b22 = B[1,1]
    # Output: C11 += -1 (C[0,0]), C12 += 1 (C[0,1])
    products.append((
        {(0,0):1, (0,1):1},   # U
        {(1,1):1},            # V
        {(0,0):-1, (1,0):1}   # W
    ))
    
    # M6: A part = a21-a11 = A[1,0]-A[0,0], B part = b11+b12 = B[0,0]+B[0,1]
    # Output: C22 += 1 (C[1,1])
    products.append((
        {(1,0):1, (0,0):-1},  # U
        {(0,0):1, (0,1):1},   # V
        {(1,1):1}             # W
    ))
    
    # M7: A part = a12-a22 = A[0,1]-A[1,1], B part = b21+b22 = B[1,0]+B[1,1]
    # Output: C11 += 1 (C[0,0])
    products.append((
        {(0,1):1, (1,1):-1},  # U
        {(1,0):1, (1,1):1},   # V
        {(0,0):1}             # W
    ))
    
    return products

def _construct_strassen_blocks(n, m, p):
    """Construct the decomposition using Strassen for 2x2x2 sub-blocks where possible."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    strassen_products = _strassen_2x2_2x2()
    
    # Determine which 2x2x2 blocks fit
    # Block at (i0, j0, k0) covers i0..i0+1, j0..j0+1, k0..k0+1
    # We need i0+1 < n, j0+1 < m, k0+1 < p
    blocks = []
    for i0 in range(0, n - 1):
        for j0 in range(0, m - 1):
            for k0 in range(0, p - 1):
                blocks.append((i0, j0, k0))
    
    # Track which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    # Remaining schoolbook terms
    remaining = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    remaining.append((i, j, k))
    
    total_rank = len(blocks) * 7 + len(remaining)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    # Add Strassen blocks
    for (i0, j0, k0) in blocks:
        for r in range(7):
            u_coeff, v_coeff, w_coeff = strassen_products[r]
            for (di, dj), val in u_coeff.items():
                i = i0 + di
                j = j0 + dj
                U[i * m + j, col] += val
            for (dj, dk), val in v_coeff.items():
                j = j0 + dj
                k = k0 + dk
                V[j * p + k, col] += val
            for (dk, di), val in w_coeff.items():
                k = k0 + dk
                i = i0 + di
                W[k * n + i, col] += val
            col += 1
    
    # Add schoolbook terms
    for (i, j, k) in remaining:
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    schoolbook_rank = n * m * p
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    return _construct_strassen_blocks(n, m, p)
# EVOLVE-BLOCK-END
