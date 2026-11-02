# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _combine_decomps(decomp1, decomp2, n, m1, m2, p, offset1, offset2):
    """
    Combine two decompositions for A1 B1 and A2 B2.
    decomp1: U1, V1, W1 for n x m1 x p
    decomp2: U2, V2, W2 for n x m2 x p
    A is n x (m1+m2). V1 is m1 x r1, V2 is m2 x r2.
    We need to construct V of size (m1+m2) x (r1+r2).
    V[:m1, :r1] = V1
    V[m1:, r1:] = V2
    U and W are concatenated horizontally.
    """
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    U = np.hstack([U1, U2])
    W = np.hstack([W1, W2])
    
    V = np.zeros((m1 + m2, r1 + r2), dtype=int)
    V[:m1, :r1] = V1
    V[m1:, r1:] = V2
    
    return U, V, W

def _get_decomp(n, m, p):
    """
    Get rank-minimal decomposition for n x m x p matrix multiplication.
    Returns U, V, W as numpy arrays.
    """
    # Base case: m = 1
    if m == 1:
        # C_ij = A_i1 B_1j
        # Rank 1
        U = np.ones((n, 1), dtype=int)
        V = np.ones((1, 1), dtype=int)
        W = np.ones((p, 1), dtype=int)
        return U, V, W
        
    # Base case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        # Strassen decomposition
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # U, V, W for each M
        # Indices: U[n*m, r], V[m*p, r], W[p*n, r]
        # A indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        # B indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        # C indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # A11=0, A22=3. B11=0, B22=3.
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21+A22)B11 -> C21 += M2
        # A21=2, A22=3. B11=0.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3: A11(B12-B22) -> C12 += M3
        # A11=0. B12=1, B22=3.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4: A22(B21-B11) -> C22 += M4
        # A22=3. B21=2, B11=0.
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5: (A11+A12)B22 -> C12 += M5
        # A11=0, A12=1. B22=3.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6: (A21-A11)(B11+B12) -> C21 += M6
        # A21=2, A11=0. B11=0, B12=1.
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        # A12=1, A22=3. B21=2, B22=3.
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Recursive step: split m into m1, m2
    # Choose m1, m2 to minimize rank
    # Try m1 = m // 2, m2 = m - m1
    # Or m1 = 1, m2 = m-1
    
    best_rank = float('inf')
    best_decomp = None
    
    # Try splitting m into 1 and m-1
    if m > 1:
        m1, m2 = 1, m - 1
        d1 = _get_decomp(n, m1, p)
        d2 = _get_decomp(n, m2, p)
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            best_decomp = _combine_decomps(d1, d2, n, m1, m2, p, 0, 0)
            
        # Try splitting m into m//2 and m - m//2
        if m >= 2:
            m1, m2 = m // 2, m - m // 2
            if m1 != 1 or m2 != m - 1: # Avoid duplicate if m=2
                d1 = _get_decomp(n, m1, p)
                d2 = _get_decomp(n, m2, p)
                U1, V1, W1 = d1
                U2, V2, W2 = d2
                r1 = U1.shape[1]
                r2 = U2.shape[1]
                if r1 + r2 < best_rank:
                    best_rank = r1 + r2
                    best_decomp = _combine_decomps(d1, d2, n, m1, m2, p, 0, 0)
                    
    if best_decomp is None:
        # Fallback to schoolbook if recursion fails (should not happen)
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
        
    return best_decomp

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 110
    
    # We split the 4x5x5 problem into two 2x5x5 problems
    # A (4x5) = [A1 (2x5); A2 (2x5)]
    # C (4x5) = [C1 (2x5); C2 (2x5)]
    # C1 = A1 B, C2 = A2 B
    
    d1 = _get_decomp(2, 5, 5)
    d2 = _get_decomp(2, 5, 5)
    
    U1, V1, W1 = d1
    U2, V2, W2 = d2
    
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    # Combine U, V, W
    # U1 is 2*5 x r1. U2 is 2*5 x r2.
    # We need U of size 4*5 x (r1+r2).
    # U[:10, :r1] = U1
    # U[10:, r1:] = U2
    
    U = np.zeros((20, r1 + r2), dtype=int)
    V = np.zeros((25, r1 + r2), dtype=int)
    W = np.zeros((20, r1 + r2), dtype=int)
    
    U[:10, :r1] = U1
    U[10:, r1:] = U2
    
    V[:25, :r1] = V1 # V1 is 5*5 x r1
    V[:25, r1:] = V2 # V2 is 5*5 x r2
    
    # W1 is 5*2 x r1? No, W is p*n x r.
    # For 2x5x5, p=5, n=2. W is 10 x r.
    # C1 is 2x5. Indices 0..9.
    # C2 is 2x5. Indices 10..19.
    W[:10, :r1] = W1
    W[10:, r1:] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
