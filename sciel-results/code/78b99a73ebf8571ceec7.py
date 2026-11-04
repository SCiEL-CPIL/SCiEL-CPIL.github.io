"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIME_LIMIT = 130  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 44 construction for 3x4x4 matrix multiplication tensor.
    # 
    # Approach: Use the structure C = A*B where A is 3x4, B is 4x4, C is 3x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Decompose B into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # A row i: [a0, a1, a2, a3] where a0,a1 are first half, a2,a3 second half.
    #
    # C[i,0:2] = [a0,a1]*B00 + [a2,a3]*B10  (2x1 = 2x2*2x1 + 2x2*2x1)
    # C[i,2:4] = [a0,a1]*B01 + [a2,a3]*B11
    #
    # For each 2x2 * 2x1 multiplication, we can use rank 2 (schoolbook) or
    # find better. But the key is sharing across the 3 rows.
    #
    # Better approach: Group by the 2x2 blocks of B.
    # For B00 (2x2): contributes to C[i,0] and C[i,1] via A[i,0:2]*B00
    #   C[i,0] += A[i,0]*B00[0,0] + A[i,1]*B00[1,0]
    #   C[i,1] += A[i,0]*B00[0,1] + A[i,1]*B00[1,1]
    # This is a 3x2x2 sub-problem (3 rows, 2 cols of A, 2 cols of output).
    #
    # The 3x2x2 tensor has schoolbook rank 12. We can potentially reduce this.
    #
    # Total schoolbook: 4 blocks * 12 = 48.
    # 
    # Improvement: For the 3x2x2 sub-problem, note that it's computing
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in 0..2, k in 0..1.
    # This is 3 independent 1x2 * 2x2 = 1x2 computations.
    # Each 1x2*2x2 has rank 2. Total: 3*2 = 6? No, 3*2*2 = 12 schoolbook.
    # 
    # Actually 1x2 * 2x2: C[k] = A[0]*B[0,k] + A[1]*B[1,k]. 
    # This requires 2 multiplications (A[0]*B[0,k] and A[1]*B[1,k]) for each k.
    # So rank 2 per k, total rank 4 for 1x2*2x2? No, the tensor T_{1,2,2} has
    # T[0,j,k] = 1 for j,k in 0..1. Rank = 2 (since it's a 2x2 all-ones matrix, rank 1? No)
    # 
    # I'm going in circles. Let me just implement a verified rank 44 solution.
    #
    # Known: 4x4x4 has rank 19. For 3x4x4, we can use the following:
    # The 3x4x4 tensor can be embedded in 4x4x4 by adding a zero row.
    # If 4x4x4 has rank 19, then 3x4x4 has rank <= 19 + (extra terms for the 3rd row structure).
    # This doesn't directly give 44.
    #
    # Let me use a different known result: 3x4x4 has rank at most 44.
    # Construction via merging 4 pairs of schoolbook terms.
    #
    # Schoolbook term (i,j,k): U[i*4+j]=1, V[j*4+k]=1, W[k*3+i]=1
    # 
    # Merge terms (i, j1, k) and (i, j2, k) for same i,k:
    # Combined: U[i*4+j1]=1, U[i*4+j2]=1, V[j1*4+k]=1, V[j2*4+k]=1, W[k*3+i]=1
    # This is NOT a single rank-1 term. It's 2 terms.
    #
    # To merge, we need: sum of 2 rank-1 = 1 rank-1, which requires the 2x2x2 subcube
    # to have rank 1, which it doesn't.
    #
    # I'll implement the safest option: return schoolbook rank 48, which is verified correct.
    # The knowledge says rank 44 is achievable, but I cannot derive the construction
    # with confidence in this session.
    
    if time.time() - _start_time > _TIME_LIMIT - 10:
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
    
    # Return verified schoolbook decomposition (rank 48)
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
    
    # Verify correctness
    assert _verify(U, V, W, n, m, p), "Schoolbook verification failed!"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
