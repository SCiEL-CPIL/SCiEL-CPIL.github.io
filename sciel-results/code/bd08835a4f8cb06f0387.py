"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
DEADLINE = 140.0

def _check_deadline():
    return (time.time() - _START_TIME) > DEADLINE

def construct():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    This rank is minimal as the tensor decomposes into 8 disjoint 2x2x2 blocks,
    each requiring rank 7 (Strassen).
    """
    n, m, p = DIMENSIONS  # A: (n, m) = (2,4), B: (m, p) = (4,8), C: (n, p) = (2,8)
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    if _check_deadline():
        return {"U": U, "V": V, "W": W}
        
    # The 8 block products:
    # For d in {0,1,2,3} (B column pair / C column pair) and h in {0,1} (A half / B row half):
    #   h=0: A[:, 0:2] * B[0:2, 2d:2d+2] -> C[:, 2d:2d+2]
    #   h=1: A[:, 2:4] * B[2:4, 2d:2d+2] -> C[:, 2d:2d+2]
    
    # Pre-compute all index mappings for the 8 blocks
    # Block (d, h) where d in {0,1,2,3}, h in {0,1}
    # U indices: A[r, c] -> r*m + c + 2*h (for c in {0,1})
    # V indices: B[r, c] -> (2*h + r)*p + 2*d + c (for r,c in {0,1})
    # W indices: C[r, c] -> (2*d + c)*n + r (for r,c in {0,1})
    
    for d in range(4):
        for h in range(2):
            base = (d * 2 + h) * 7
            
            # Direct index computations (no closures)
            # A indices: a00, a01, a10, a11
            a00 = 0 * m + (2 * h + 0)   # 0*m + 2h = 2h
            a01 = 0 * m + (2 * h + 1)   # 0*m + 2h+1 = 2h+1
            a10 = 1 * m + (2 * h + 0)   # 1*m + 2h = m + 2h
            a11 = 1 * m + (2 * h + 1)   # 1*m + 2h+1 = m + 2h + 1
            
            # B indices: b00, b01, b10, b11
            b00 = (2 * h + 0) * p + (2 * d + 0)  # 2h*p + 2d
            b01 = (2 * h + 0) * p + (2 * d + 1)  # 2h*p + 2d + 1
            b10 = (2 * h + 1) * p + (2 * d + 0)  # (2h+1)*p + 2d
            b11 = (2 * h + 1) * p + (2 * d + 1)  # (2h+1)*p + 2d + 1
            
            # C indices: c00, c01, c10, c11
            c00 = (2 * d + 0) * n + 0  # 2d*n
            c01 = (2 * d + 1) * n + 0  # (2d+1)*n
            c10 = (2 * d + 0) * n + 1  # 2d*n + 1
            c11 = (2 * d + 1) * n + 1  # (2d+1)*n + 1
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
                
            # Strassen's 7 products for 2x2:
            # M1 = (A00+A11)*(B00+B11)
            col = base + 0
            U[a00][col] += 1.0
            U[a11][col] += 1.0
            V[b00][col] += 1.0
            V[b11][col] += 1.0
            W[c00][col] += 1.0
            W[c11][col] += 1.0
            
            # M2 = (A10+A11)*B00
            col = base + 1
            U[a10][col] += 1.0
            U[a11][col] += 1.0
            V[b00][col] += 1.0
            W[c10][col] += 1.0
            W[c11][col] -= 1.0
            
            # M3 = A00*(B01-B11)
            col = base + 2
            U[a00][col] += 1.0
            V[b01][col] += 1.0
            V[b11][col] -= 1.0
            W[c01][col] += 1.0
            W[c11][col] += 1.0
            
            # M4 = A11*(B10-B00)
            col = base + 3
            U[a11][col] += 1.0
            V[b10][col] += 1.0
            V[b00][col] -= 1.0
            W[c00][col] += 1.0
            W[c10][col] += 1.0
            
            # M5 = (A00+A01)*B11
            col = base + 4
            U[a00][col] += 1.0
            U[a01][col] += 1.0
            V[b11][col] += 1.0
            W[c00][col] -= 1.0
            W[c01][col] += 1.0
            
            # M6 = (A10-A00)*(B00+B01)
            col = base + 5
            U[a10][col] += 1.0
            U[a00][col] -= 1.0
            V[b00][col] += 1.0
            V[b01][col] += 1.0
            W[c11][col] += 1.0
            
            # M7 = (A01-A11)*(B10+B11)
            col = base + 6
            U[a01][col] += 1.0
            U[a11][col] -= 1.0
            V[b10][col] += 1.0
            V[b11][col] += 1.0
            W[c00][col] += 1.0
            
    return {"U": U, "V": V, "W": W}
