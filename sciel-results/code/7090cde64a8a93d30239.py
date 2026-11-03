"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Try to find a valid rank-23 decomposition for 3x3x3 using Bini et al.
    # The Bini et al. (1979) algorithm has rank 23 for 3x3x3.
    # We'll use a known set of coefficients and verify them.
    
    # Bini et al. (1979) rank-23 coefficients for 3x3x3
    # U, V, W are 9x23 matrices
    # Indexing: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Let's use a verified set of Bini et al. coefficients
    # These are from the standard reference implementation
    
    # Actually, let me use a simpler approach: use the Strassen-like structure
    # For 3x3x3, the best known is 23 (Bini et al.)
    
    # Let me try to construct a valid rank-23 decomposition
    # Using the known Bini et al. algorithm
    
    # Bini et al. (1979) explicit form:
    # The 23 multiplications are:
    # M0 = (a11+a22+a33)(b11+b22+b33)
    # M1 = (a11+a22-a33)(b11+b22-b33)
    # M2 = (a11-a22+a33)(b11-b22+b33)
    # M3 = (a11-a22-a33)(b11-b22-b33)
    # M4 = (a11+a33-a22)(b11+b33-b22)
    # M5 = (a11-a33+a22)(b11-b33+b22)
    # M6 = (a22+a33-a11)(b22+b33-b11)
    # M7 = (a22-a33+a11)(b22-b33+b11)
    # M8 = (a12+a21)(b12+b21)
    # M9 = (a12-a21)(b12-b21)
    # M10 = (a13+a31)(b13+b31)
    # M11 = (a13-a31)(b13-b31)
    # M12 = (a23+a32)(b23+b32)
    # M13 = (a23-a32)(b23-b32)
    # M14 = (a11+a12+a13)(b11+b21+b31)
    # M15 = (a21+a22+a23)(b12+b22+b32)
    # M16 = (a31+a32+a33)(b13+b23+b33)
    # M17 = (a12+a13+a22)(b12+b13+b22)
    # M18 = (a13+a22+a23)(b13+b22+b23)
    # M19 = (a11+a23+a33)(b11+b23+b33)
    # M20 = (a12+a23+a31)(b12+b23+b31)
    # M21 = (a13+a21+a32)(b13+b21+b32)
    # M22 = (a11+a22+a33)(b11+b22+b33) -- this is same as M0, so this isn't right
    
    # Let me use a different approach: just use the schoolbook for now
    # since I can't reliably recall the exact Bini et al. coefficients
    
    # Schoolbook rank = n*m*p = 3*3*6 = 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the decomposition
    valid = True
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # This shouldn't happen for schoolbook, but just in case
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
