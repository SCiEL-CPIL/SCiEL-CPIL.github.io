import numpy as np
import time

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def check_time():
    return time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN

def strassen_222():
    # Returns U, V, W for 2x2x2 with rank 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j,k in 0..1
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas for C = AB
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(g+h)
    # M7 = (b-c)(e+f)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition.
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
    # Rank 1 term c: U_{i,j} V_{j,k} W_{k,i}
    # Sum over c: \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = C_{i,k}
    
    # Let's define the 7 terms.
    # Term 1: (a+d)(e+h)
    # A part: a+d -> U_{0,0}=1, U_{1,1}=1
    # B part: e+h -> V_{0,0}=1, V_{1,1}=1
    # C part: C11+=1, C22+=1 -> W_{0,0}=1, W_{1,1}=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: (c+d)e
    # A: c+d -> U_{1,0}=1, U_{1,1}=1
    # B: e -> V_{0,0}=1
    # C: C21+=1, C22-=1? No.
    # C21 = M2 + M4. C22 = M1 - M2 + ...
    # So M2 contributes +1 to C21, -1 to C22.
    # C21 is index (i=1, k=0) -> W_{0,1}
    # C22 is index (i=1, k=1) -> W_{1,1}
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1; W[1, 1] = -1
    
    # Term 3: a(f-h)
    # A: a -> U_{0,0}=1
    # B: f-h -> V_{0,1}=1, V_{1,1}=-1
    # C: C11+=1, C12+=1
    # C11 (0,0) -> W_{0,0}
    # C12 (0,1) -> W_{1,0}
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # Term 4: d(g-e)
    # A: d -> U_{1,1}=1
    # B: g-e -> V_{1,0}=1, V_{0,0}=-1
    # C: C21+=1, C22+=1
    # C21 (1,0) -> W_{0,1}
    # C22 (1,1) -> W_{1,1}
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: (a+b)h
    # A: a+b -> U_{0,0}=1, U_{0,1}=1
    # B: h -> V_{1,1}=1
    # C: C11-=1, C12+=1
    # C11 (0,0) -> W_{0,0}
    # C12 (0,1) -> W_{1,0}
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6: (c-d)(g+h)
    # A: c-d -> U_{1,0}=1, U_{1,1}=-1
    # B: g+h -> V_{1,0}=1, V_{1,1}=1
    # C: C22+=1
    # C22 (1,1) -> W_{1,1}
    U[2, 5] = 1; U[3, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[1, 5] = 1
    
    # Term 7: (b-c)(e+f)
    # A: b-c -> U_{0,1}=1, U_{1,0}=-1
    # B: e+f -> V_{0,0}=1, V_{0,1}=1
    # C: C11+=1
    # C11 (0,0) -> W_{0,0}
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def decompose(n, m, p):
    if check_time():
        return schoolbook(n, m, p)
        
    if n == 1 and m == 1 and p == 1:
        return schoolbook(1, 1, 1)
        
    if n == 2 and m == 2 and p == 2:
        return strassen_222()
        
    best_rank = float('inf')
    best_split = None
    best_type = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1 = decompose(n1, m, p)
        r2 = decompose(n2, m, p)
        rank1 = r1[0].shape[1]
        rank2 = r2[0].shape[1]
        if rank1 + rank2 < best_rank:
            best_rank = rank1 + rank2
            best_split = (n1, n2)
            best_type = 'n'
            best_decomp = (r1, r2)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r1 = decompose(n, m1, p)
        r2 = decompose(n, m2, p)
        rank1 = r1[0].shape[1]
        rank2 = r2[0].shape[1]
        if rank1 + rank2 < best_rank:
            best_rank = rank1 + rank2
            best_split = (m1, m2)
            best_type = 'm'
            best_decomp = (r1, r2)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1 = decompose(n, m, p1)
        r2 = decompose(n, m, p2)
        rank1 = r1[0].shape[1]
        rank2 = r2[0].shape[1]
        if rank1 + rank2 < best_rank:
            best_rank = rank1 + rank2
            best_split = (p1, p2)
            best_type = 'p'
            best_decomp = (r1, r2)
            
    if best_type is None:
        return schoolbook(n, m, p)
        
    r1, r2 = best_decomp
    U1, V1, W1 = r1
    U2, V2, W2 = r2
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    if best_type == 'n':
        n1, n2 = best_split
        # U: split rows
        U[:n1 * m, :rank1] = U1
        U[n1 * m:, rank1:] = U2
        # V: same rows, concat cols
        V[:, :rank1] = V1
        V[:, rank1:] = V2
        # W: split rows
        W[:p * n1, :rank1] = W1
        W[p * n1:, rank1:] = W2
        
    elif best_type == 'm':
        m1, m2 = best_split
        # U: split rows
        U[:n * m1, :rank1] = U1
        U[n * m1:, rank1:] = U2
        # V: split rows
        V[:m1 * p, :rank1] = V1
        V[m1 * p:, rank1:] = V2
        # W: same rows, concat cols
        W[:, :rank1] = W1
        W[:, rank1:] = W2
        
    else: # p
        p1, p2 = best_split
        # U: same rows, concat cols
        U[:, :rank1] = U1
        U[:, rank1:] = U2
        # V: split rows
        V[:m * p1, :rank1] = V1
        V[m * p1:, rank1:] = V2
        # W: split rows
        W[:p1 * n, :rank1] = W1
        W[p1 * n:, rank1:] = W2
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
