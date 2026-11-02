"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
@lru_cache(maxsize=None)
def get_decomp(n, m, p):
    """
    Returns U, V, W matrices for n x m x p matrix multiplication.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    # Base Cases
    if n == 1 and m == 1 and p == 1:
        return np.ones((1, 1), dtype=int), np.ones((1, 1), dtype=int), np.ones((1, 1), dtype=int)
    
    if m == 1:
        rank = n * p
        U = np.zeros((n, rank), dtype=int)
        V = np.zeros((p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, col] = 1
                V[k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        return U, V, W
        
    if n == 1:
        rank = m * p
        U = np.zeros((m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p, rank), dtype=int)
        col = 0
        for j in range(m):
            for k in range(p):
                U[j, col] = 1
                V[j * p + k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W
                
    if p == 1:
        rank = n * m
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m, rank), dtype=int)
        W = np.zeros((n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
        
    # Special Case: Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=-1; W[2,0]=-1; W[3,0]=1
        
        # M2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        
        # M3
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=-1
        
        # M4
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        W[2,3]=1; W[0,3]=-1
        
        # M5
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[1,4]=1; W[3,4]=-1
        
        # M6
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[2,5]=1; W[0,5]=-1
        
        # M7
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[1,6]=1; W[3,6]=-1
        
        return U, V, W

    # Recursive Split
    best_rank = float('inf')
    best_split = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = get_decomp(n1, m, p)
        U2, V2, W2 = get_decomp(n2, m, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            best_split = ('n', n1, n2, U1, V1, W1, U2, V2, W2)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = get_decomp(n, m1, p)
        U2, V2, W2 = get_decomp(n, m2, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            best_split = ('m', m1, m2, U1, V1, W1, U2, V2, W2)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = get_decomp(n, m, p1)
        U2, V2, W2 = get_decomp(n, m, p2)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            best_split = ('p', p1, p2, U1, V1, W1, U2, V2, W2)
            
    if best_split is None:
        # Fallback to schoolbook if no split found (should not happen for n,m,p > 1)
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

    # Construct result
    split_type, d1, d2, U1, V1, W1, U2, V2, W2 = best_split
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    if split_type == 'n':
        n1 = d1
        n2 = d2
        # U1 maps to first n1*m rows
        U[:n1*m, :r1] = U1
        # U2 maps to last n2*m rows
        U[n1*m:, r1:] = U2
        
        # V is concatenated
        V[:, :r1] = V1
        V[:, r1:] = V2
        
        # W1 maps to rows where i < n1
        # W2 maps to rows where i >= n1
        for k in range(p):
            for i in range(n1):
                W[k*n + i, :r1] = W1[k*n1 + i, :]
            for i2 in range(n2):
                i = n1 + i2
                W[k*n + i, r1:] = W2[k*n2 + i2, :]
                
    elif split_type == 'm':
        m1 = d1
        m2 = d2
        # U1 maps to columns j < m1
        # U2 maps to columns j >= m1
        for i in range(n):
            for j in range(m1):
                U[i*m + j, :r1] = U1[i*m1 + j, :]
            for j2 in range(m2):
                j = m1 + j2
                U[i*m + j, r1:] = U2[i*m2 + j2, :]
                
        # V1 maps to rows j < m1
        # V2 maps to rows j >= m1
        for j in range(m1):
            for k in range(p):
                V[j*p + k, :r1] = V1[j*p + k, :] # Wait, V1 shape is (m1*p, r1)
                # V1 index: j*p + k? No, V1 is for m1 x p.
                # V1 row index is j*p + k.
                # V2 row index is j2*p + k.
                V[j*p + k, :r1] = V1[j*p + k, :]
        for j2 in range(m2):
            for k in range(p):
                j = m1 + j2
                V[j*p + k, r1:] = V2[j2*p + k, :]
                
        # W is concatenated
        W[:, :r1] = W1
        W[:, r1:] = W2

    elif split_type == 'p':
        p1 = d1
        p2 = d2
        # U is concatenated
        U[:, :r1] = U1
        U[:, r1:] = U2
        
        # V1 maps to columns k < p1
        # V2 maps to columns k >= p1
        for j in range(m):
            for k in range(p1):
                V[j*p + k, :r1] = V1[j*p1 + k, :]
            for k2 in range(p2):
                k = p1 + k2
                V[j*p + k, r1:] = V2[j*p2 + k2, :]
                
        # W1 maps to rows where k < p1
        # W2 maps to rows where k >= p1
        for k in range(p1):
            for i in range(n):
                W[k*n + i, :r1] = W1[k*n + i, :]
        for k2 in range(p2):
            for i in range(n):
                k = p1 + k2
                W[k*n + i, r1:] = W2[k2*n + i, :]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = get_decomp(n, m, p)
    # Convert to list of lists if required, but numpy arrays are usually accepted
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is standard for this interface.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
