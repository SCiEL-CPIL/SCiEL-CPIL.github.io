"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Strategy:
    Use the Kronecker product structure: T_{3,3,6} = T_{3,3,3} ⊗ T_{1,3,2}
    where T_{1,3,2} is the 1x3 by 3x2 multiplication tensor.
    
    We use the known rank-7 Strassen decomposition for 3x3x3 and a rank-3 decomposition for 1x3x2.
    This gives a rank-21 decomposition.
    
    Actually, let me reconsider. The 1x3 by 3x2 multiplication:
    C[0,k] = sum_j A[0,j] * B[j,k] for k=0,1
    So the tensor T_{i,j,k} where i=0 only, j in [0,3), k in [0,2): T[0,j,k] = 1
    This is a 1x3x2 tensor that is all ones.
    
    Rank of 1x3x2 all-ones tensor: since it's a matrix (flattened), the rank is min(1, 3*2) = 1?
    No, it's a 3-way tensor. Let me think...
    T[0,j,k] = 1 for all j,k.
    We need sum_r u_r[0] * v_r[j] * w_r[k] = 1 for all j in [0,3), k in [0,2).
    
    If we set u_r[0] = 1, then we need sum_r v_r[j] * w_r[k] = 1 for all j,k.
    This is a rank-1 matrix (all ones) of size 3x2, which has rank 1.
    So we can use rank 1: v[j] = 1, w[k] = 1 for all j,k.
    
    So T_{1,3,2} has rank 1!
    
    Then T_{3,3,6} = T_{3,3,3} ⊗ T_{1,3,2} has rank 7 * 1 = 7?
    
    Wait, that's not right. Let me reconsider the tensor product.
    
    The tensor T_{n,m,p} ⊗ T_{n',m',p'} has dimensions (n*n', m*m', p*p') and 
    T[i,i',j,j',k,k'] = T[i,j,k] * T[i',j',k']
    
    For matrix multiplication, T_{n,m,p}[i,j,k] = 1 (all ones).
    So T_{3,3,6} should have all entries equal to 1 as well.
    
    If T_{3,3,6} is the all-ones tensor of size (3,3,6), and T_{3,3,3} is all-ones of size (3,3,3),
    and T_{1,3,2} is all-ones of size (1,3,2), then T_{3,3,3} ⊗ T_{1,3,2} is the all-ones tensor 
    of size (3*1, 3*3, 3*2) = (3, 9, 6), not (3,3,6).
    
    So the Kronecker product doesn't directly work this way.
    
    Let me think differently. For 3x3 by 3x6 multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can split the 6 columns into two groups of 3:
    C[:, 0:3] = A * B[:, 0:3]  (3x3 times 3x3)
    C[:, 3:6] = A * B[:, 3:6]  (3x3 times 3x3)
    
    Each 3x3 by 3x3 multiplication can be done with rank 7 (Strassen).
    So total rank = 7 + 7 = 14.
    
    Let me construct this:
    For the first block (columns 0-2): use Strassen on A and B1 = B[:, 0:3]
    For the second block (columns 3-5): use Strassen on A and B2 = B[:, 3:6]
    
    The rank-7 Strassen decomposition for 3x3 matrix multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using the standard Strassen-like rank-7 algorithm for 3x3:
    The 7 intermediate products are:
    M1 = (A[0,0] + A[1,1] + A[2,2]) * (B[0,0] + B[1,1] + B[2,2])
    M2 = (A[0,1] + A[1,0] - A[2,2]) * (B[0,1] + B[1,0] - B[2,2])
    M3 = (A[0,2] + A[1,2] - A[2,1]) * (B[0,2] + B[1,2] - B[2,1])
    M4 = (A[2,0] + A[2,1] - A[1,0]) * (B[2,0] + B[2,1] - B[1,0])
    M5 = (A[2,1] + A[2,2] - A[1,2]) * (B[2,1] + B[2,2] - B[1,2])
    M6 = (A[1,0] + A[1,2] - A[0,2]) * (B[1,0] + B[1,2] - B[0,2])
    M7 = (A[1,2] + A[0,0] - A[0,1]) * (B[1,2] + B[0,0] - B[0,1])
    
    And the output is:
    C[0,0] = M1 - M2 - M3 + M4
    C[0,1] = M2 + M3 - M4 - M7
    C[0,2] = M3 + M7 - M1 - M6
    C[1,0] = M4 - M5 - M2 + M6
    C[1,1] = M5 + M2 - M6 - M3
    C[1,2] = M6 - M3 + M5 - M7  (need to verify)
    C[2,0] = M1 - M4 - M5 + M3  (need to verify)
    C[2,1] = M5 - M7 + M1 - M6  (need to verify)
    C[2,2] = M7 - M6 + M4 - M2  (need to verify)
    
    Actually, I should use a known correct rank-7 decomposition. Let me use the standard one.
    
    For now, let me just use the schoolbook rank-54 decomposition to ensure correctness, then try to optimize.
    """
    n, m, p = DIMENSIONS
    
    # First, let's try rank 14 using two Strassen decompositions
    # If that fails, fall back to rank 54 schoolbook
    
    # For now, use schoolbook to guarantee correctness
    rank = n * m * p
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
