import numpy as np

DIMENSIONS = [3, 4, 6]

def _decomp_1_mp(n, m, p):
    """Decomposition for 1 x m x p or n x 1 x p or n x m x 1."""
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

def _decomp_strassen_222():
    """Strassen decomposition for 2x2x2."""
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # i,j in 0..1, k in 0..1
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1: (a+d)(e+h)
    # a=U[0], d=U[3], e=V[0], h=V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C[0,0] += m1, C[1,1] += m1
    # Wait, W[k*n+i]. C[0,0] is k=0,i=0 -> W[0]. C[1,1] is k=1,i=1 -> W[3].
    
    # m2: (c+d)e
    # c=U[2], d=U[3], e=V[0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C[1,0] += m2 (k=0, i=1 -> W[1*2+0]=W[2])
    
    # m3: a(f-h)
    # a=U[0], f=V[1], h=V[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C[0,1] += m3 (k=1, i=0 -> W[1*2+0]=W[2]?)
    # W index: k*n+i. n=2.
    # C[0,1]: i=0, k=1. Index = 1*2+0 = 2.
    # C[1,1]: i=1, k=1. Index = 1*2+1 = 3.
    # Strassen: C[0,1] = m3 + m5. C[1,1] = m1 - m2 + m3 + m6.
    # So W[2] should get m3. W[3] should get m3.
    W[2, 2] = 1
    W[3, 2] = 1
    
    # m4: d(g-e)
    # d=U[3], g=V[2], e=V[0]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C[0,0] += m4
    W[2, 3] = 1 # C[1,0] += m4
    
    # m5: (a+b)h
    # a=U[0], b=U[1], h=V[3]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C[0,0] -= m5
    W[1, 4] = 1  # C[0,1] += m5
    
    # m6: (c-a)(e+f)
    # c=U[2], a=U[0], e=V[0], f=V[1]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1] += m6
    
    # m7: (b-d)(g+h)
    # b=U[1], d=U[3], g=V[2], h=V[3]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C[0,0] += m7
    
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, n, m, p, split_dim, split_idx):
    """
    Combine two decompositions.
    split_dim: 0 for n, 1 for m, 2 for p.
    split_idx: the index where the split occurs.
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Copy first part
    U[:, :rank1] = U1
    V[:, :rank1] = V1
    W[:, :rank1] = W1
    
    # Map second part
    # We need to map indices from (n2, m2, p2) to (n, m, p)
    n1, n2 = 0, 0
    m1, m2 = 0, 0
    p1, p2 = 0, 0
    
    if split_dim == 0:
        n1 = split_idx
        n2 = n - n1
    elif split_dim == 1:
        m1 = split_idx
        m2 = m - m1
    elif split_dim == 2:
        p1 = split_idx
        p2 = p - p1
        
    # U mapping
    for i2 in range(n2):
        for j2 in range(m2):
            if split_dim == 0:
                i = i2 + n1
            else:
                i = i2
            if split_dim == 1:
                j = j2 + m1
            else:
                j = j2
            idx2 = i2 * m2 + j2
            idx1 = i * m + j
            U[idx1, rank1:rank1+rank2] = U2[idx2, :]
            
    # V mapping
    for j2 in range(m2):
        for k2 in range(p2):
            if split_dim == 1:
                j = j2 + m1
            else:
                j = j2
            if split_dim == 2:
                k = k2 + p1
            else:
                k = k2
            idx2 = j2 * p2 + k2
            idx1 = j * p + k
            V[idx1, rank1:rank1+rank2] = V2[idx2, :]
            
    # W mapping
    for k2 in range(p2):
        for i2 in range(n2):
            if split_dim == 2:
                k = k2 + p1
            else:
                k = k2
            if split_dim == 0:
                i = i2 + n1
            else:
                i = i2
            idx2 = k2 * n2 + i2
            idx1 = k * n + i
            W[idx1, rank1:rank1+rank2] = W2[idx2, :]
            
    return U, V, W

def _get_decomp(n, m, p):
    if n == 1 or m == 1 or p == 1:
        return _decomp_1_mp(n, m, p)
    if n == 2 and m == 2 and p == 2:
        return _decomp_strassen_222()
        
    best_rank = float('inf')
    best_decomp = None
    best_params = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _get_decomp(n1, m, p)
        U2, V2, W2 = _get_decomp(n2, m, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            U, V, W = _combine(U1, V1, W1, U2, V2, W2, n, m, p, 0, n1)
            best_decomp = (U, V, W)
            best_params = (0, n1)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _get_decomp(n, m1, p)
        U2, V2, W2 = _get_decomp(n, m2, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            U, V, W = _combine(U1, V1, W1, U2, V2, W2, n, m, p, 1, m1)
            best_decomp = (U, V, W)
            best_params = (1, m1)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _get_decomp(n, m, p1)
        U2, V2, W2 = _get_decomp(n, m, p2)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            U, V, W = _combine(U1, V1, W1, U2, V2, W2, n, m, p, 2, p1)
            best_decomp = (U, V, W)
            best_params = (2, p1)
            
    return best_decomp

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = _get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
