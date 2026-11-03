"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Try DFT-based rank-15 decomposition: 3 independent 1x5x5 products, each rank 5
    # For 1x5x5: compute y_k = sum_j x_j * M_{j,k}
    # Using 5-point DFT with omega = exp(2*pi*i/5)
    # We use the fact that over Z_5, we can use the DFT to reduce multiplications
    
    # The DFT-based algorithm for 1x5x5:
    # Let F be the 5x5 DFT matrix: F[j,k] = omega^(j*k) where omega = e^(2*pi*i/5)
    # Then y = (1/5) * (F * x) ⊙ (F * M) where ⊙ is column-wise element-wise product
    # This requires 5 multiplications per output column, but we can share the F*x computation
    # Actually, this gives rank 5 for the 1x5x5 case if we structure it correctly
    
    # For the 1x5x5 subproblem with DFT:
    # U rows: 5 (for the 5 components of x)
    # V rows: 25 (for the 25 entries of M)
    # W rows: 5 (for the 5 outputs)
    # Rank: 5
    
    # The DFT matrix entries are complex, but we need integer/half-integer entries.
    # This is a problem. Let me use a different approach.
    
    # Alternative: Use the fact that 5 is prime and use the number theoretic transform
    # or simply use a structured bilinear algorithm.
    
    # Actually, let's use the known result: for 1xm x p, the rank is at most m*p,
    # but for specific cases we can do better.
    
    # Let me try a different structural approach: block decomposition
    # 3x5 = [3x2 | 3x3], 5x5 = [2x2 2x3; 3x2 3x3]
    # Use Strassen on the 2x2 block, schoolbook elsewhere
    
    # Actually, let's just use schoolbook but verify it works, then try to improve
    
    # Schoolbook: rank = n*m*p = 75
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
    # C[k,i] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j... 
    # Let's verify: C[k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook, each column r corresponds to a unique (i,j,k) triple
    # U[i*m+j, r] = 1 only for the matching i,j
    # V[j*p+k, r] = 1 only for the matching j,k
    # W[k*n+i, r] = 1 only for the matching k,i
    # So the product is 1 only when all three match, giving the correct sum
    
    if time.time() - _start_time > _TIMEOUT:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
