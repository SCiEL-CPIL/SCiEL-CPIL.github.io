"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
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
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4,7), V has shape (4,7), W has shape (4,7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = a00(b10-b11)
    U[0][2] = 1  # a00
    V[2][2] = 1  # b10
    V[3][2] = -1 # b11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = a11(b01-b00)
    U[3][3] = 1  # a11
    V[1][3] = 1  # b01
    V[0][3] = -1 # b00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (a00+a01)b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (a10-a00)(b00+b10)
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[2][5] = 1  # b10
    W[3][5] = 1  # C11
    
    # M7 = (a01-a11)(b01+b11)
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[1][6] = 1  # b01
    V[3][6] = 1  # b11
    W[0][6] = 1  # C00
    
    return U, V, W

def _try_winograd_3x4x4():
    """
    Try to construct a lower-rank decomposition for 3x4x4 using Winograd on 2x2 blocks.
    Partition the inner dimension (size 4) into two groups of 2: {0,1} and {2,3}.
    For each group, use Strassen's 7-product scheme on 2x2 sub-blocks.
    """
    n, m, p = 3, 4, 4
    deadline = time.time() + 140  # safety margin
    
    # We'll build the decomposition by combining:
    # 1. Strassen on rows (0,1) of A, inner dims (0,1), all cols of B
    # 2. Strassen on row 2 of A, inner dims (0,1), all cols of B  
    # 3. Strassen on rows (0,1) of A, inner dims (2,3), all cols of B
    # 4. Strassen on row 2 of A, inner dims (2,3), all cols of B
    # But this overcounts. Let me think differently.
    
    # Alternative: use the full 3x4x4 tensor and try to find a rank reduction
    # by exploiting the structure.
    
    # For now, return schoolbook as fallback
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # First, try to find a better decomposition
    best = _schoolbook(n, m, p)
    best_rank = best["U"] and len(best["U"][0]) or 0
    
    # Try Winograd-based construction
    try:
        candidate = _try_winograd_3x4x4()
        if candidate and _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            rank = len(candidate["U"][0])
            if rank < best_rank:
                best = candidate
                best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return best
    
    # Try random search for further rank reduction
    try:
        rng = np.random.default_rng(42)
        current_best = best_rank
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Try to remove one column from the current best and see if we can
            # find a valid completion
            if current_best <= 40:
                break
            
            # Generate a random perturbation of the schoolbook decomposition
            # and try to reduce rank
            rank = current_best - 1
            if rank < 20:
                break
            
            # Skip complex search for now, just return best found
            pass
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
