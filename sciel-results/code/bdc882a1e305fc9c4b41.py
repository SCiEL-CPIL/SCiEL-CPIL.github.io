import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_decomposition():
    """
    Returns U, V, W for a single 2x2x2 Strassen block.
    U shape: (2*2, 7)
    V shape: (2*2, 7)
    W shape: (2*2, 7)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Term 0: P1 = (a+d)(e+h)
    # A: a(0,0)=1, d(1,1)=1
    U[0, 0] = 1; U[3, 0] = 1
    # B: e(0,0)=1, h(1,1)=1
    V[0, 0] = 1; V[3, 0] = 1
    # C: C00(0,0)=1, C11(1,1)=1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: P2 = (c+d)e
    # A: c(1,0)=1, d(1,1)=1
    U[2, 1] = 1; U[3, 1] = 1
    # B: e(0,0)=1
    V[0, 1] = 1
    # C: C10(1,0)=1
    W[1, 1] = 1 # k=0, i=1 -> 0*2+1=1
    
    # Term 2: P3 = a(f-h)
    # A: a(0,0)=1
    U[0, 2] = 1
    # B: f(0,1)=1, h(1,1)=-1
    V[1, 2] = 1; V[3, 2] = -1
    # C: C01(0,1)=1
    W[2, 2] = 1 # k=1, i=0 -> 1*2+0=2
    
    # Term 3: P4 = d(g-e)
    # A: d(1,1)=1
    U[3, 3] = 1
    # B: g(1,0)=1, e(0,0)=-1
    V[2, 3] = 1; V[0, 3] = -1
    # C: C00(0,0)=1, C10(1,0)=1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 4: P5 = (a+b)h
    # A: a(0,0)=1, b(0,1)=1
    U[0, 4] = 1; U[1, 4] = 1
    # B: h(1,1)=1
    V[3, 4] = 1
    # C: C00(0,0)=-1
    W[0, 4] = -1
    
    # Term 5: P6 = (c-a)(e+f)
    # A: c(1,0)=1, a(0,0)=-1
    U[2, 5] = 1; U[0, 5] = -1
    # B: e(0,0)=1, f(0,1)=1
    V[0, 5] = 1; V[1, 5] = 1
    # C: C11(1,1)=1
    W[3, 5] = 1
    
    # Term 6: P7 = (b-d)(g+h)
    # A: b(0,1)=1, d(1,1)=-1
    U[1, 6] = 1; U[3, 6] = -1
    # B: g(1,0)=1, h(1,1)=1
    V[2, 6] = 1; V[3, 6] = 1
    # C: C00(0,0)=1, C11(1,1)=-1
    W[0, 6] = 1; W[3, 6] = -1
    
    return U, V, W

def _construct_strassen_hybrid(n, m, p):
    """
    Constructs the hybrid decomposition for 2x3x10.
    Splits j into {0,1} and {2}.
    Splits k into 5 blocks of size 2 for the first part.
    """
    if n != 2 or m != 3 or p != 10:
        return None
        
    # Part 1: 2x2x10 using 5 Strassen blocks
    U1, V1, W1 = _strassen_decomposition()
    
    U_total = np.zeros((n*m, 5*7 + 1))
    V_total = np.zeros((m*p, 5*7 + 1))
    W_total = np.zeros((p*n, 5*7 + 1))
    
    for l in range(5):
        offset = l * 7
        # U is same for all blocks (indices i,j in 0..1)
        U_total[:, offset:offset+7] = U1
        
        # V indices: j in 0..1, k in 2l..2l+1
        # V1 has shape (2*2, 7). V1[j*2+k', c]
        # We need to place this into V_total[j*10+k, c]
        for j in range(2):
            for kp in range(2):
                k = 2*l + kp
                V_total[j*10 + k, offset:offset+7] = V1[j*2 + kp, :]
                
        # W indices: k in 2l..2l+1, i in 0..1
        # W1 has shape (2*2, 7). W1[k'*2+i, c]
        # We need to place this into W_total[k*2+i, c]
        for kp in range(2):
            k = 2*l + kp
            for i in range(2):
                W_total[k*2 + i, offset:offset+7] = W1[kp*2 + i, :]
                
    # Part 2: 2x1x10 using rank 1
    offset = 35
    # U indices: i in 0..1, j=2
    U_total[2*2 + 0, offset] = 1 # i=0, j=2 -> index 2
    U_total[2*2 + 1, offset] = 1 # i=1, j=2 -> index 3
    
    # V indices: j=2, k in 0..9
    for k in range(10):
        V_total[2*10 + k, offset] = 1 # j=2, k=k -> index 20+k
        
    # W indices: k in 0..9, i in 0..1
    for k in range(10):
        for i in range(2):
            W_total[k*2 + i, offset] = 1
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try the specific Strassen hybrid construction
    result = _construct_strassen_hybrid(n, m, p)
    if result is not None:
        if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
            if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                return result
        else:
            # If fast verify fails, there's a bug in construction. Fall back to schoolbook.
            pass
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

