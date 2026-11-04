import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    # Rank 7 decomposition for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indexing: A[i,j], B[j,k], C[k,i]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # M1
    U[0*2+0, 0] += 1; U[1*2+1, 0] += 1
    V[0*2+0, 0] += 1; V[1*2+1, 0] += 1
    W[0*2+0, 0] += 1; W[1*2+1, 0] += 1
    
    # M2
    U[1*2+0, 1] += 1; U[1*2+1, 1] += 1
    V[0*2+0, 1] += 1
    W[0*2+1, 1] += 1; W[1*2+1, 1] += 1
    
    # M3
    U[0*2+0, 2] += 1
    V[0*2+1, 2] += 1; V[1*2+1, 2] -= 1
    W[0*2+0, 2] += 1; W[1*2+0, 2] += 1
    
    # M4
    U[1*2+1, 3] += 1
    V[1*2+0, 3] += 1; V[0*2+0, 3] -= 1
    W[0*2+0, 3] += 1; W[1*2+1, 3] += 1
    
    # M5
    U[0*2+0, 4] += 1; U[0*2+1, 4] += 1
    V[1*2+1, 4] += 1
    W[0*2+0, 4] -= 1; W[1*2+0, 4] += 1
    
    # M6
    U[1*2+0, 5] += 1; U[0*2+0, 5] -= 1
    V[0*2+0, 5] += 1; V[0*2+1, 5] += 1
    W[1*2+1, 5] += 1
    
    # M7
    U[0*2+1, 6] += 1; U[1*2+1, 6] -= 1
    V[1*2+0, 6] += 1; V[1*2+1, 6] += 1
    W[0*2+0, 6] += 1; W[1*2+1, 6] += 1
    
    return U, V, W

def _get_decomp(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    else:
        return _schoolbook(n, m, p)

def _combine_decomps(decomps, offsets):
    # decomps: list of (U, V, W)
    # offsets: list of (i0, j0, k0)
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    n_total = 5
    m_total = 5
    p_total = 5
    
    U = np.zeros((n_total * m_total, total_rank))
    V = np.zeros((m_total * p_total, total_rank))
    W = np.zeros((p_total * n_total, total_rank))
    
    col_offset = 0
    for (U_sub, V_sub, W_sub), (i0, j0, k0) in zip(decomps, offsets):
        n_sub, m_sub, p_sub = U_sub.shape[0] // m_sub, m_sub, V_sub.shape[0] // m_sub
        # Actually, we need to know the dimensions of the sub-problem
        # U_sub is (n_sub * m_sub, rank_sub)
        # We need to map local indices (i, j) to global (i0+i, j0+j)
        
        # Determine n_sub, m_sub, p_sub from the shapes
        # This is tricky if we don't pass them. 
        # Let's assume we pass them or infer.
        # For now, we will hardcode the call structure.
        pass

def construct():
    # 5x5 split into 3+2
    # Blocks:
    # (0,0) 3x3 * 3x3 -> 3x3
    # (0,1) 3x2 * 2x3 -> 3x3
    # (1,0) 2x3 * 3x2 -> 2x2
    # (1,1) 2x2 * 2x2 -> 2x2
    
    # Wait, the split is:
    # A = [[A33, A32], [A23, A22]]
    # B = [[B33, B32], [B23, B22]]
    # C = [[C33, C32], [C23, C22]]
    
    # C33 = A33 B33 + A32 B23
    # C32 = A33 B32 + A32 B22
    # C23 = A23 B33 + A22 B23
    # C22 = A23 B32 + A22 B22
    
    decomps = []
    offsets = []
    
    # 1. A33 B33
    U1, V1, W1 = _get_decomp(3, 3, 3)
    decomps.append((U1, V1, W1))
    offsets.append((0, 0, 0))
    
    # 2. A32 B23
    U2, V2, W2 = _get_decomp(3, 2, 3)
    decomps.append((U2, V2, W2))
    offsets.append((0, 3, 0)) # i0=0, j0=3, k0=0
    
    # 3. A33 B32
    U3, V3, W3 = _get_decomp(3, 3, 2)
    decomps.append((U3, V3, W3))
    offsets.append((0, 0, 3))
    
    # 4. A32 B22
    U4, V4, W4 = _get_decomp(3, 2, 2)
    decomps.append((U4, V4, W4))
    offsets.append((0, 3, 3))
    
    # 5. A23 B33
    U5, V5, W5 = _get_decomp(2, 3, 3)
    decomps.append((U5, V5, W5))
    offsets.append((3, 0, 0))
    
    # 6. A22 B23
    U6, V6, W6 = _get_decomp(2, 2, 3)
    decomps.append((U6, V6, W6))
    offsets.append((3, 3, 0))
    
    # 7. A23 B32
    U7, V7, W7 = _get_decomp(2, 3, 2)
    decomps.append((U7, V7, W7))
    offsets.append((3, 0, 3))
    
    # 8. A22 B22
    U8, V8, W8 = _get_decomp(2, 2, 2)
    decomps.append((U8, V8, W8))
    offsets.append((3, 3, 3))
    
    # Combine
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U_total = np.zeros((25, total_rank))
    V_total = np.zeros((25, total_rank))
    W_total = np.zeros((25, total_rank))
    
    col = 0
    for (U_sub, V_sub, W_sub), (i0, j0, k0) in zip(decomps, offsets):
        # Determine sub-dimensions
        # U_sub shape (n*m, r). We need n, m.
        # We can infer n, m, p from the offsets and the fact that it's a block.
        # Or we can just pass them.
        # Let's modify _get_decomp to return dims or store them.
        # For simplicity, I will hardcode the mapping in the loop.
        
        # We need to know n_sub, m_sub, p_sub.
        # From the calls:
        # 1: 3,3,3
        # 2: 3,2,3
        # 3: 3,3,2
        # 4: 3,2,2
        # 5: 2,3,3
        # 6: 2,2,3
        # 7: 2,3,2
        # 8: 2,2,2
        
        dims = [
            (3,3,3), (3,2,3), (3,3,2), (3,2,2),
            (2,3,3), (2,2,3), (2,3,2), (2,2,2)
        ]
        
        n_sub, m_sub, p_sub = dims[i]
        
        for i in range(n_sub):
            for j in range(m_sub):
                U_total[(i0+i) * 5 + (j0+j), col:col+U_sub.shape[1]] = U_sub[i * m_sub + j, :]
        
        for j in range(m_sub):
            for k in range(p_sub):
                V_total[(j0+j) * 5 + (k0+k), col:col+V_sub.shape[1]] = V_sub[j * p_sub + k, :]
        
        for k in range(p_sub):
            for i in range(n_sub):
                W_total[(k0+k) * 5 + (i0+i), col:col+W_sub.shape[1]] = W_sub[k * n_sub + i, :]
        
        col += U_sub.shape[1]
        
    return {"U": U_total, "V": V_total, "W": W_total}

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END
