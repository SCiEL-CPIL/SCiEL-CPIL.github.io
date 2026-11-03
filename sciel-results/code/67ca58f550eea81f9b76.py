import time
import numpy as np

_DIMS = (4, 5, 5)
_DEADLINE = 145.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_halfint(x):
    """Convert a float entry to an exact int/half-int (as a float multiple of 0.5)."""
    hx = round(x * 2.0)
    if abs(x * 2.0 - hx) < 1e-9:
        return hx * 0.5
    return x

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm (rank 7).
    Dimensions: n=2, m=2, p=2.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to express these as sums of rank-1 tensors.
    # A rank-1 tensor for 2x2x2 is defined by u (4x1), v (4x1), w (4x1).
    # The contribution to C[i,k] is sum_r U[i,2+j,r] V[j,2+k,r] W[k,2+i,r].
    
    # Let's define the 7 terms explicitly.
    # Term 1: M1
    # (A00 + A11) * (B00 + B11)
    # Expansion: A00B00 + A00B11 + A11B00 + A11B11
    # This is not a single rank-1 term in the original A, B basis.
    # Strassen computes M1 by first forming X = A00+A11 and Y = B00+B11, then multiplying X*Y.
    # The multiplication X*Y is a 2x2 matrix multiplication.
    # So we need the decomposition of 2x2 multiplication for X and Y.
    # But X and Y are linear combinations of A and B.
    # So the total decomposition is a composition.
    
    # This is getting complex to hardcode manually.
    # I will implement a generic tensor composition.
    
    pass

def construct():
    n, m, p = _DIMS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    if _remaining() < 20.0:
        return best

    try:
        # Construct rank 69 decomposition
        # Part 1: 4x4x4 using Strassen (Rank 49)
        # Part 2: 4x5x1 using Schoolbook (Rank 20)
        
        # We will build the matrices U, V, W directly.
        # Total Rank = 49 + 20 = 69.
        
        rank = 69
        U = [[0] * rank for _ in range(n * m)] # 20 x 69
        V = [[0] * rank for _ in range(m * p)] # 25 x 69
        W = [[0] * rank for _ in range(p * n)] # 20 x 69
        
        # --- Part 1: 4x4x4 (cols 0-3 of B, rows 0-3 of C) ---
        # This part uses A[0:4, 0:4] and B[0:4, 0:4] to produce C[0:4, 0:4].
        # We use Strassen for 4x4.
        # Strassen 4x4 is composed of 7 Strassen 2x2 multiplications.
        # Each 2x2 multiplication has rank 7.
        # Total 49 terms.
        
        # We need the 2x2 Strassen decomposition coefficients.
        # Let's define the 7 terms for 2x2 multiplication (indices 0..3 for U, V, W).
        # U[i*2+j], V[j*2+k], W[k*2+i].
        
        # The 7 rank-1 terms for 2x2x2 (Strassen):
        # We can find these in literature or derive.
        # Actually, the standard 7 terms are:
        # t1: (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        # This is not a single rank-1 term in A, B.
        # Strassen's algorithm reduces the number of multiplications, but each "multiplication" is a bilinear form.
        # To get the tensor decomposition, we need to expand each of the 7 products into rank-1 terms.
        # But wait, the 7 products are of 2x2 matrices.
        # So each product requires a rank-7 decomposition itself?
        # No, Strassen's algorithm *is* the rank-7 decomposition.
        # The 7 "multiplications" in Strassen are scalar multiplications if we view the blocks as scalars?
        # No, the blocks are 2x2 matrices.
        # So for 4x4, we have 7 blocks of 2x2 matrices to multiply.
        # Each 2x2 multiplication has rank 7.
        # So total rank is 7 * 7 = 49.
        
        # So we need to:
        # 1. Define the 7 Strassen combinations for 4x4 blocks.
        # 2. For each combination, apply the 2x2 rank-7 decomposition.
        
        # Let's define the 2x2 rank-7 decomposition first.
        # Indices for 2x2:
        # A: 00, 01, 10, 11 -> idx 0,1,2,3
        # B: 00, 01, 10, 11 -> idx 0,1,2,3
        # C: 00, 01, 10, 11 -> idx 0,1,2,3 (k*2+i)
        
        # The 7 terms of Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # These M's are 2x2 matrices.
        # C = f(M1..M7).
        
        # To get the rank-1 decomposition, we need to express M1..M7 as rank-1 tensors?
        # No, M1 is a product of two linear combinations.
        # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # This product itself has rank 7? No, it's just a matrix.
        # The "multiplication" is the bilinear operation.
        
        # I will use a known explicit rank-49 decomposition for 4x4.
        # If I can't find it, I will use schoolbook 4x4 (rank 64).
        # Rank 64 + 20 = 84.
        
        # Let's use schoolbook 4x4 to be safe and fast.
        # Schoolbook 4x4:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i * 16 + j * 4 + k # 0..63
                    U[i * 5 + j][r] = 1
                    V[j * 5 + k][r] = 1
                    W[k * 4 + i][r] = 1
        
        # Part 2: 4x5x1 (col 4 of B, all rows of C)
        # C[i, 4] = sum_j A[i, j] B[j, 4]
        for i in range(4):
            for j in range(5):
                r = 64 + i * 5 + j # 64..83
                U[i * 5 + j][r] = 1
                V[j * 5 + 4][r] = 1
                W[4 * 4 + i][r] = 1
        
        best_rank = 84
        best = {"U": U, "V": V, "W": W}
        
        if _verify(best["U"], best["V"], best["W"], n, m, p):
            pass
        else:
            best = _schoolbook(n, m, p)
            best_rank = 100
            
    except Exception:
        best = _schoolbook(n, m, p)
        best_rank = 100
    
    for mat in (best["U"], best["V"], best["W"]):
        for row in mat:
            for c in range(len(row)):
                row[c] = _to_halfint(row[c])
    
    return best
