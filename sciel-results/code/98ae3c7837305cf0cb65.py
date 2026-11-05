"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _winograd_3x3x3():
    """Winograd's rank-23 decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W as lists of lists."""
    n, m, p = 3, 3, 3
    rank = 23
    # Winograd 3x3x3 from Strassen 1969 / Winograd 1972
    # U has 9 rows, V has 9 rows, W has 9 rows
    # Coefficients are small integers or halves
    # Using known rank-23 scheme:
    
    # The Winograd 3x3x3 uses 23 multiplications
    # Let's use the standard formulation
    # F2 = [[1,0,1,0,1],[-1,2,-1,2,-1],[1,2,1,2,1]]  (5x3 transform for A columns)
    # F1^T = [[1,0,1,0,1],[-1,2,-1,2,-1],[1,2,1,2,1]]  (3x5 transform for B rows)
    # This gives rank 5*5 = 25, not 23.
    
    # The actual rank-23 is different. Let me use the known explicit scheme.
    # From "Optimal tensor decompositions" literature, rank-23 for 3x3x3:
    
    # I'll construct it using the known factor matrices.
    # Actually, let me use a simpler approach: try rank 25 (Winograd) and validate.
    # If rank 25 works, that's still better than 36.
    
    # Winograd 5x5 scheme for 3x3:
    # F2 (5x3) transforms each column of A
    # F1 (5x3) transforms each row of B (transpose)
    # Then 25 multiplications
    
    F2 = [
        [1, 0, 1],
        [-1, 2, -1],
        [1, 2, 1],
        [0, 1, 0],
        [0, 1, 0]
    ]
    # Wait, this isn't right. Let me use the correct Winograd transforms.
    # For 3x3, the standard Winograd uses 5 multiplications per "block"
    # F2 = [[1,0,1],[-1,2,-1],[1,2,1]] would be 3x3, but we need 5x3
    
    # Let me just use the known rank-25 Winograd scheme explicitly.
    # Actually, the simplest correct approach: use the bilinear algorithm.
    
    # Let me construct a valid rank-25 decomposition using Winograd's method.
    # A is 3x3, B is 3x3, C = A*B is 3x3.
    # Transform A columns: a' = F2 * a, where F2 is 5x3
    # Transform B rows: b' = F1 * b, where F1 is 5x3
    # C = (F2^{-1} if exists...) 
    # This is getting complex. Let me just return schoolbook for safety.
    
    return None  # Signal to use schoolbook

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # First, try to return a known good decomposition
    # For 3x3x4, we can split into 3x3x3 + 3x3x1
    # 3x3x3 has rank 23 (Winograd) or 25
    # 3x3x1 has rank 9 (each output element needs one multiplication with each input)
    # Actually 3x3x1: C is 3x1, A is 3x3, B is 3x1.
    # C[i] = sum_j A[i,j]*B[j]. Rank is 9 (schoolbook) but can be less?
    # Actually for n x m x p with p=1, it's just n*m dot products... 
    # No, it's n outputs, each is a dot product of length m.
    # The tensor is sum_{i,j} e_{i,j} tensor e_j tensor e_i
    # This has rank n*m = 9 for 3x3x1.
    
    # So if 3x3x3 has rank R, then 3x3x4 has rank at most R + 9.
    # With R=25 (Winograd 5x5), total = 34. Better than 36.
    # With R=23, total = 32. Even better.
    
    # Let me try to construct a rank-25 Winograd for 3x3x3.
    # The Winograd F2 and F1 transforms:
    # For multiplying two 3x3 matrices:
    # F2 = [[1,0,1],[1,1,1],[-1,2,-1],[1,1,1],[1,0,1]] -- no, this is wrong.
    
    # Let me use a different approach: construct a valid low-rank decomposition
    # by solving a system of polynomial equations.
    
    # Actually, the safest bet: just return schoolbook rank 36.
    # The previous attempts to use Winograd were not validated.
    
    # But let me try one more thing: can we do rank 34?
    # 3x3x4 = 3x3x3 + 3x3x1
    # If I can find a rank-25 decomposition for 3x3x3, I get rank 34.
    
    # Let me construct the Winograd 3x3x3 rank-25 decomposition explicitly.
    # Standard Winograd for 3x3:
    # The 5-point transform:
    # F2 = [[1, 0, 1],
    #       [1, 1, 1], 
    #       [-1, 2, -1],
    #       [1, 1, 1],
    #       [1, 0, 1]]
    # No, this doesn't work because F2 needs to be invertible in some sense.
    
    # The correct Winograd transforms for 3x3:
    # a' = F2 * a where F2 is 5x3
    # b' = F1 * b where F1 is 5x3  
    # F2 = [[1, 0, 1],
    #       [-1, 2, -1],
    #       [1, 2, 1],
    #       [0, 1, 0],
    #       [0, 1, 0]]
    # Hmm, rows 3 and 4 are identical, so this doesn't span.
    
    # I think the correct Winograd uses:
    # F2 = [[1, 0, 1],
    #       [-1, 2, -1],
    #       [1, 2, 1]]
    # which is 3x3, giving rank 9, not helpful.
    
    # Let me just go with schoolbook. It's guaranteed correct.
    
    # Actually, wait. Let me try a different strategy.
    # Use random initialization and gradient descent to find a low-rank decomposition.
    # But that's risky and might not converge in time.
    
    # The safest and most reliable approach: return schoolbook rank 36.
    result = _schoolbook(n, m, p)
    
    # Validate before returning
    if _validate(result['U'], result['V'], result['W'], n, m, p):
        return result
    
    # This should never happen, but just in case
    return result
# EVOLVE-BLOCK-END
