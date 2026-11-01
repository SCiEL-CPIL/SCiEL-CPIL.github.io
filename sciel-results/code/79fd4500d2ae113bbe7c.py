"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(W[0])
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _try_rank_10():
    """Try to find a rank-10 decomposition via randomized search."""
    n, m, p = DIMENSIONS
    rank = 10
    # The tensor has n*m*p = 12 entries.
    # We need to find U (4x10), V (6x10), W (6x10) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # where i in [0,1], j in [0,1], k in [0,2]
    
    # This is a system of 12 equations with 4*10 + 6*10 + 6*10 = 160 unknowns.
    # The equations are cubic (trilinear). We can use a randomized approach.
    
    # Strategy: Fix W to be simple (e.g., W[k*n+i, r] = 1 for all), then solve for U and V.
    # But that gives sum_r U[i*m+j, r] * V[j*p+k, r] = 1, which is a system of quadratic equations.
    
    # Alternative: Use the fact that we can set some entries to 0 to simplify.
    # Let's try a structured approach: for each rank term r, assign it to contribute
    # to a specific subset of the 12 tensor entries.
    
    # Actually, let's try a more direct approach: 
    # The tensor M[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try to find a solution where W is diagonal in some sense.
    # Suppose W[k*n+i, r] = w_r for all i,k (i.e., W has rank 1 as a matrix).
    # Then we need: w_r * sum_r' U[i*m+j, r'] * V[j*p+k, r'] = 1 for all i,j,k
    # This doesn't quite work.
    
    # Let's try a different approach: use the known rank-11 solution and try to merge two terms.
    # In the rank-11 solution, the schoolbook terms for column 2 are:
    # M8: U[0,7]=1, V[2,7]=1, W[4,7]=1 (contributes to C[0,2] via A[0,0]*B[0,2])
    # M9: U[1,8]=1, V[5,8]=1, W[4,8]=1 (contributes to C[0,2] via A[0,1]*B[1,2])
    # M10: U[2,9]=1, V[2,9]=1, W[5,9]=1 (contributes to C[1,2] via A[1,0]*B[0,2])
    # M11: U[3,10]=1, V[5,10]=1, W[5,10]=1 (contributes to C[1,2] via A[1,1]*B[1,2])
    
    # Can we merge M8 and M10? They both involve V[2,*] (B[0,2]).
    # M8+M10: U[0,7]=1, U[2,9]=1, V[2,7]=1, V[2,9]=1
    # If we combine: U[0,r]=1, U[2,r]=1, V[2,r]=1, then this gives A[0,0]*B[0,2] + A[1,0]*B[0,2]
    # But we need separate contributions to C[0,2] and C[1,2].
    # W[4,r] would give (A[0,0]+A[1,0])*B[0,2] to C[0,2], which is wrong.
    
    # Let's try randomized search for rank 10.
    start_time = time.time()
    deadline = start_time + 80  # Leave time for verification and fallback
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(attempt * 1000 + 42)
        
        # Randomly initialize U, V, W with small integers or simple patterns
        # Try to find a solution using iterative refinement
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with random sparse patterns
        for r in range(rank):
            # Each rank term should involve a small number of non-zero entries
            # Pick random positions
            u_idx = rng.choice(n*m, size=rng.integers(1, 3), replace=False)
            v_idx = rng.choice(m*p, size=rng.integers(1, 3), replace=False)
            w_idx = rng.choice(p*n, size=rng.integers(1, 3), replace=False)
            
            for idx in u_idx:
                U[idx, r] = rng.choice([-1, 1])
            for idx in v_idx:
                V[idx, r] = rng.choice([-1, 1])
            for idx in w_idx:
                W[idx, r] = rng.choice([-1, 1])
        
        # Check if this random assignment works
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # First try rank 10
    result_10 = _try_rank_10()
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 11 (Strassen + schoolbook)
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen-style part for the 2x2 block (columns 0 and 1 of B)
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    # Verify the decomposition
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
