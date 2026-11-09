"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n+i, r] corresponds to output element C[i,k] from A[i,j]*B[j,k]
    The decomposition requires: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    # For each (i,j,k) triple, check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2x2():
    """Classic Strassen 2x2x2 decomposition, rank 7.
    Returns U (4x7), V (4x7), W (4x7) for multiplying 2x2 * 2x2.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U[i*2+j, r] = coefficient for A[i,j] in multiplication r
    # V[j*2+k, r] = coefficient for B[j,k] in multiplication r
    # W[k*2+i, r] = coefficient for C[i,k] from multiplication r
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1   # A00, A11
    V[0][0] = 1; V[3][0] = 1   # B00, B11
    W[0][0] = 1; W[3][0] = 1   # C00, C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1   # A10, A11
    V[0][1] = 1                # B00
    W[2][1] = 1; W[3][1] = -1  # C10, C11
    
    # M3 = A00(B10-B11)
    U[0][2] = 1                # A00
    V[2][2] = 1; V[3][2] = -1  # B10, B11
    W[1][2] = 1; W[3][2] = 1   # C01, C11
    
    # M4 = A11(B01-B10)
    U[3][3] = 1                # A11
    V[1][3] = 1; V[2][3] = -1  # B01, B10
    W[0][3] = 1; W[2][3] = 1   # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1   # A00, A01
    V[3][4] = 1                # B11
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[1][5] = 1   # B00, B01
    W[3][5] = 1                # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, A11
    V[2][6] = 1; V[3][6] = 1   # B10, B11
    W[0][6] = 1                # C00
    
    return U, V, W

def _pad_and_decompose(n, m, p, N, M, P, U2, V2, W2):
    """
    Decompose n x m * m x p by padding to N x M * M x P and using the 2x2x2 Strassen
    recursively, then extracting the relevant sub-tensor.
    """
    # For now, we'll try a direct approach: use Strassen on blocks
    # This is complex, so let's try a different strategy.
    pass

def _strassen_pad(n, m, p):
    """
    Try to use Strassen by padding n,m,p to powers of 2.
    4x5*5x6 -> pad to 4x8*8x8 (m=5->8, p=6->8)
    Then 4x8*8x8 can be decomposed using Strassen on 2x2 blocks.
    4 = 2*2, 8 = 2*2*2, so we need 3 levels of Strassen in the 8-dimension.
    Actually, let's think more carefully.
    
    We want to multiply A (4x5) by B (5x6) to get C (4x6).
    Pad A to 4x8, B to 8x8, compute C' = A' * B' (4x8), extract C (4x6).
    
    4x8 * 8x8: We can view this as 2x(2*4) * (2*4)*8... this gets complicated.
    
    Alternative: pad to 8x8 * 8x8 and use 3 levels of Strassen (rank 7^3 = 343).
    That's worse than 120.
    
    Better: 4x8 * 8x8. View as (2x2) blocks:
    A is 2x4 blocks of 2x2, B is 4x4 blocks of 2x2.
    C is 2x4 blocks of 2x2.
    C[i,j] = sum_k A[i,k] * B[k,j] for block indices.
    This is a 2x4 * 4x4 block matmul, which has 2*4*4 = 32 block multiplications,
    each of rank 7, giving total rank 224. Worse.
    
    Let me try a different approach: direct construction for small cases.
    """
    pass

def _try_bilinear_search(n, m, p, max_rank, time_limit):
    """
    Try to find a decomposition of rank <= max_rank using a greedy/greedy-random approach.
    This is a heuristic search.
    """
    start = time.time()
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # For small ranks, we can try to verify candidate decompositions
    # But constructing them is hard. Let's try a different approach.
    
    # Use the fact that we can combine Strassen with schoolbook for sub-blocks.
    # For 4x5*5x6:
    # Split A into 2x2 blocks: A = [[A00, A01, A02, A03, A04], [A10, A11, A12, A13, A14]]
    # where each block is 2x1 (since 4 rows = 2 blocks of 2, 5 cols = 5 blocks of 1)
    # This doesn't work well.
    
    # Alternative: use the "laser method" or known results.
    # For 4x5*5x6, the best known upper bound is likely around 100-110.
    # Let's try a structured decomposition.
    
    # Strategy: pad 4x5*5x6 to 4x8*8x8, use a 2-level Strassen-like approach.
    # 4x8*8x8: split into 2x4 blocks of 2x2.
    # A has 2 row-blocks, 4 col-blocks (each 2x2)
    # B has 4 row-blocks, 4 col-blocks (each 2x2)
    # C has 2 row-blocks, 4 col-blocks (each 2x2)
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # This is a 2x4 * 4x4 matrix multiplication of 2x2 blocks.
    # Rank of 2x4*4x4 block matmul: schoolbook is 2*4*4 = 32, each rank 7 => 224.
    # Too high.
    
    # Let me try: 4x5*5x6 directly with a smarter decomposition.
    # Known: M(4,5,6) <= M(4,8,8) and M(4,8,8) can be computed via:
    # Pad to 8x8*8x8, use 3-level Strassen: rank 343. Too high.
    
    # Actually, let's just return the schoolbook for now and try to improve.
    # The key insight: we can use Strassen on the 5x6 part by padding 5->8, 6->8.
    # 4x8 * 8x8: this is a rectangular matmul.
    # We can compute it as: view A as 2x2 blocks (2 row-blocks, 4 col-blocks of 2x2)
    # B as 4x4 blocks of 2x2.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # For fixed i (2 choices), j (4 choices): C[i,j] = sum_{k=0}^{3} A[i,k]*B[k,j]
    # This is 2*4 = 8 independent "vector" multiplications of length 4.
    # Each can be done with rank... well, it's a sum of 4 products.
    # Schoolbook: 8*4 = 32 multiplications of 2x2, each rank 7 => 224.
    
    # I think for this problem size, the best we can do with known methods is still
    # close to schoolbook. Let me just return schoolbook.
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based improvements
    # For 4x5*5x6, pad to 4x8*8x8
    # Use a hybrid: Strassen on 2x2 blocks where possible
    
    # Let's try a direct construction:
    # Split the 5-dimension into 2+2+1 and 6-dimension into 2+2+2
    # A: 4x5 = 4x(2+2+1)
    # B: 5x6 = (2+2+1)x(2+2+2)
    
    # This gives block structure:
    # A = [A1 | A2 | A3] where A1:4x2, A2:4x2, A3:4x1
    # B = [[B11,B12,B13],[B21,B22,B23],[B31,B32,B33]]
    # C = A1*B1 + A2*B2 + A3*B3 (block-wise)
    # C: 4x6
    
    # C = A1*B1 + A2*B2 + A3*B3
    # A1*B1: 4x2 * 2x2 = 4x2, schoolbook rank 4*2*2=16
    # A2*B2: 4x2 * 2x2 = 4x2, schoolbook rank 16
    # A3*B3: 4x1 * 1x2 = 4x2, schoolbook rank 4*1*2=8
    # Total schoolbook: 16+16+8 = 40 for the block parts
    # But we also need to handle the cross terms... no, this is just a sum.
    # Wait, C = A*B where A is 4x5 and B is 5x6.
    # If we partition the middle dimension as 2+2+1:
    # C = A[:,0:2]*B[0:2,:] + A[:,2:4]*B[2:4,:] + A[:,4:5]*B[4:5,:]
    # Each term is a matrix product.
    # Term 1: 4x2 * 2x6 = 4x6, schoolbook rank 4*2*6 = 48
    # Term 2: 4x2 * 2x6 = 4x6, schoolbook rank 48
    # Term 3: 4x1 * 1x6 = 4x6, schoolbook rank 24
    # Total: 48+48+24 = 120. Same as schoolbook.
    
    # To improve, we need to use Strassen on some blocks.
    # For term 1: 4x2 * 2x6. Pad 2->4, 6->8: 4x4*4x8.
    # 4x4*4x8: use 2-level Strassen on the 4x4 part.
    # 4x4*4x8: split 4x4 into 2x2 blocks (2x2 block grid), 4x8 into 2x4 block grid of 2x2.
    # C[i,j] = sum_k A[i,k]*B[k,j] for 2x2 blocks.
    # 2x2 * 2x4 block matmul: schoolbook 2*2*4=16 block mults, each rank 7 => 112.
    # Worse than 48.
    
    # I think the fundamental issue is that Strassen helps when you have square blocks.
    # For 4x5*5x6, the dimensions are awkward.
    
    # Let's try: use Strassen on a 2x2x2 sub-problem embedded in the larger one.
    # This is very hard to do optimally.
    
    # For now, return schoolbook. The search space for better decompositions
    # is large and requires more sophisticated algorithms.
    
    if time.time() > deadline:
        return result
    
    return result

# EVOLVE-BLOCK-END
