"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Use Kronecker product structure to decompose 2x3x6 into scalar multiplications.
    # Each element of A (2x3) and B (3x6) contributes to specific elements of C (2x6).
    # We can represent this as a sum of rank-1 terms where each term corresponds to
    # one (i,j,k) triple in the tensor C[i,k] = sum_j A[i,j]*B[j,k].
    
    # The total number of non-zero terms in the naive decomposition is n*m*p = 2*3*6 = 36.
    # But we can be smarter: for each j, we have a 2x6 outer product of A[:,j] and B[j,:].
    # A[:,j] is a column vector (2x1), B[j,:] is a row vector (1x6).
    # Their outer product is 2x6, which can be decomposed into 2 rank-1 terms (using 2x1 * 1x6).
    # Wait, that's not right. The outer product of a 2-vector and a 6-vector is already rank 1 
    # if we think of it as a matrix, but in the tensor decomposition sense, we need to think differently.
    
    # Actually, let's think about this more carefully. The tensor T has dimensions 2x3x6.
    # T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k], but actually the tensor is defined by
    # C[i,k] = sum_j A[i,j] * B[j,k].
    
    # The standard way to decompose matrix multiplication is:
    # For each j, we have contributions from A[:,j] and B[j,:].
    # The product A[:,j] * B[j,:] is a 2x6 matrix that is rank 1 (outer product).
    # So for each j, we can use 1 rank-1 term to represent the contribution from that j.
    # Total rank = m = 3? No, that's not right either.
    
    # Let me reconsider. In tensor decomposition, a rank-1 term is a product of three vectors:
    # u (length n*m), v (length m*p), w (length p*n)
    # such that T[i,j,k] = sum_r u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # For matrix multiplication C = A*B, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (the tensor is all ones in the appropriate sense).
    
    # The naive decomposition uses n*m*p terms, but we can do better.
    # For each j, define:
    # u_j: has 1's at positions i*m+j for all i (i.e., column j of A)
    # v_j: has 1's at positions j*p+k for all k (i.e., row j of B)
    # w_j: has 1's at positions k*n+i for all i,k (i.e., all of C)
    
    # But this doesn't work because w_j would be the same for all j, and we'd be summing
    # the same C matrix multiple times.
    
    # The correct approach: for each j, we need a separate w vector that picks out
    # only the contributions from that j. But in matrix multiplication, all j's contribute
    # to all (i,k) positions, so we can't separate them in w.
    
    # Actually, the standard decomposition for matrix multiplication A (m1 x m2) * B (m2 x m3)
    # has rank at most m1*m2*m3, but can be lower using algorithms like Strassen.
    
    # For 2x3x6, let's use the fact that we can group operations.
    # One approach: treat it as 3 independent 2x1x2 multiplications (for each j),
    # but that's not quite right either.
    
    # Let me use a different strategy: decompose based on the structure.
    # We can write C = A * B where A is 2x3 and B is 3x6.
    # We can partition B into 3 columns of size 2: B = [B1, B2, B3] where each Bj is 3x2.
    # Then C = [A*B1, A*B2, A*B3], and each A*Bj is 2x2.
    
    # For each 2x2 multiplication, we can use Strassen's algorithm with rank 7.
    # So total rank = 3 * 7 = 21.
    
    # Let's implement this.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 6 x 21
    V = [[0] * rank for _ in range(m * p)]  # 18 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen terms for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1-M4+M6+M7, M3+M5], [M2+M4, -M1+M3+M5+M6]]
    
    # Indices for A (2x2): a=0, b=1, c=2, d=3
    # Indices for B (2x2): e=0, f=1, g=2, h=3
    # Indices for C (2x2): c00=0, c01=1, c10=2, c11=3
    
    strassen_terms = [
        # M1
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        # M2
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        # M3
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        # M4
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: -1, 2: 1}},
        # M5
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: 1, 1: 1}},
        # M6
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {0: 1, 3: 1}},
        # M7
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1, 2: 1}},
    ]
    
    # Fill the matrices for 3 blocks (each block is a 2x2 multiplication)
    for block in range(3):
        if time.time() > _deadline:
            break
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            
            # U: A indices for block. A is 2x3, so for block, we use columns 0 and 1 of A?
            # Wait, I need to reconsider the block structure.
            # A is 2x3, B is 3x6. If I partition B into 3 blocks of 2 columns each,
            # then for block b, I'm computing A * B[:, 2b:2b+2].
            # But A is the same for all blocks, so I can't just use the same A indices.
            
            # Actually, the issue is that A is 2x3, not 2x2. So I can't directly apply
            # Strassen to a 2x2 multiplication unless I reduce the problem.
            
            # Let me try a different approach: use the naive decomposition but with
            # some optimization.
            
            # For now, let's just fill in the Strassen terms assuming A is 2x2 for each block,
            # but this is incorrect because A is 2x3.
            
            # I need to rethink this. Let me use a simpler decomposition.
            
            # Approach: decompose into n*m*p = 2*3*6 = 36 rank-1 terms.
            # For each (i,j,k), define:
            # u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1
            # This gives a valid decomposition with rank 36.
            
            # But we can do better. Let's use the fact that for fixed j, the contributions
            # from A[:,j] and B[j,:] can be combined.
            
            # Actually, let's just use the naive decomposition for now and see if we can improve.
            pass
    
    # Let me restart with a cleaner approach.
    # Use the naive decomposition: rank = n*m*p = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u[i*m+j] = 1
                U[i * m + j][col] = 1
                # v[j*p+k] = 1
                V[j * p + k][col] = 1
                # w[k*n+i] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
