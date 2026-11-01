"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen decompositions.
    """
    n, m, p = DIMENSIONS  # A: (n, m) = (2,4), B: (m, p) = (4,8), C: (n, p) = (2,8)
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    # The 8 block products:
    # For d in {0,1,2,3} (B column pair) and h in {0,1} (A half / B row half):
    #   h=0: A[:, 0:2] (2x2) * B[0:2, 2d:2d+2] (2x2) -> contributes to C[:, 2d:2d+2]
    #   h=1: A[:, 2:4] (2x2) * B[2:4, 2d:2d+2] (2x2) -> contributes to C[:, 2d:2d+2]
    
    for d in range(4):
        for h in range(2):
            base = (d * 2 + h) * 7
            
            # Index functions
            if h == 0:
                def a_idx(r, c):
                    return r * m + c
                def b_idx(r, c):
                    return r * p + (2 * d + c)
            else:
                def a_idx(r, c):
                    return r * m + (2 + c)
                def b_idx(r, c):
                    return (2 + r) * p + (2 * d + c)
            
            def c_idx(r, c):
                return (2 * d + c) * n + r
            
            # Strassen's 7 products for 2x2:
            # M1 = (A00+A11)*(B00+B11)
            col = base + 0
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M2 = (A10+A11)*B00
            col = base + 1
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            W[c_idx(1, 1)][col] -= 1.0
            
            # M3 = A00*(B01-B11)
            col = base + 2
            U[a_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M4 = A11*(B10-B00)
            col = base + 3
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            
            # M5 = (A00+A01)*B11
            col = base + 4
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            
            # M6 = (A10-A00)*(B00+B01)
            col = base + 5
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(0, 0)][col] -= 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M7 = (A01-A11)*(B10+B11)
            col = base + 6
            U[a_idx(0, 1)][col] += 1.0
            U[a_idx(1, 1)][col] -= 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[a_idx, b_idx, c_idx] = sum_s U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
    # Should equal 1 if (a,b,c) corresponds to valid (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # and 0 otherwise.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = 0.0
                for s in range(rank):
                    val += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that other entries are 0 (sample a few)
    import random
    random.seed(42)
    for _ in range(100):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        # Check if this is a valid (i,j,k) combination
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or i1 != i2 or k1 != k2:
            val = 0.0
            for s in range(rank):
                val += U[a, s] * V[b, s] * W[c, s]
            if abs(val) > 1e-9:
                return False
    
    return True
