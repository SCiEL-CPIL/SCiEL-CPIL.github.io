"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x2 matrix by a 2x10 matrix.
    
    Strategy: Use Strassen's algorithm for 2x2 matrix multiplication, which has rank 7
    instead of 8. Since the second matrix is 2x10, we can decompose it into 5 independent
    2x2 blocks. However, we need to be careful about how the blocks interact.
    
    Actually, let's think more carefully. We're computing C = A * B where A is 2x2 and B is 2x10.
    C is 2x10. We can view B as 5 blocks of 2x2: B = [B0, B1, B2, B3, B4] where each Bi is 2x2.
    Then C = [C0, C1, C2, C3, C4] where Ci = A * Bi.
    
    If we use Strassen for each block independently, we'd get rank 7*5 = 35.
    But can we do better by sharing some computations across blocks?
    
    Let's try the straightforward approach first: apply Strassen to each 2x2 block.
    This gives rank 35, which is better than the schoolbook rank 40.
    
    Strassen's algorithm for 2x2 matrices:
    Given A = [[a,b],[c,d]] and B = [[e,f],[g,h]], the product C = A*B can be computed
    with 7 multiplications:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For the tensor decomposition, we need to express this as a sum of rank-1 tensors.
    Each "multiplication" in Strassen corresponds to a rank-1 term in the tensor decomposition.
    
    Let's define the indices:
    - U has shape (n*m, rank) = (4, rank)
    - V has shape (m*p, rank) = (20, rank)  
    - W has shape (p*n, rank) = (20, rank)
    
    The tensor T[i,j,k] represents the coefficient for A[i,j] * B[j,k] in the output C[i,k].
    Wait, let me re-read the convention.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W has shape (p*n, rank) = (20, 20) if rank were 20, but rank is what we're finding.
    W[k*n + i, r] where k is 0..p-1=0..9, i is 0..n-1=0..1.
    
    The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that doesn't seem right. Let me think about the standard matrix multiplication tensor.
    
    For C = A * B, C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], else 0.
    Actually, T[i,j,k] = delta_{i,k}... no.
    
    Let me reconsider. The matrix multiplication tensor T_{ijk} satisfies:
    (A * B)_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    So T_{ijk} = 1 for all i,j,k? No, that's not right either.
    
    Actually, (A*B)_{ik} = sum_j A_{ij} B_{jk}, so the tensor is T_{ijk} = 1 for all valid i,j,k,
    and the contraction is over j. Wait, no. The tensor T has three indices, and the multiplication
    is a 3-tensor contraction.
    
    Let me look at this differently. The standard decomposition is:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    where (u_r)_{ij} = U[i*m+j, r], (v_r)_{jk} = V[j*p+k, r], (w_r)_{ki} = W[k*n+i, r]
    
    And T_{ijk} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For matrix multiplication, T_{ijk} should equal 1 if the indices form a valid chain i->j->k,
    but actually for the standard matrix multiplication, T_{ijk} = 1 for all i,j,k in range,
    and the contraction gives (A*B)_{ik} = sum_j T_{ijk} A_{ij} B_{jk}... 
    
    Hmm, I think I'm overcomplicating this. Let me just implement the Strassen-based decomposition
    and verify it works.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # We'll build the decomposition using Strassen for each 2x2 block of B.
    # B is 2x10, so we have 5 blocks of 2x2.
    # For each block b (0..4), B[:, 2b:2b+2] is a 2x2 matrix.
    
    # Strassen's 7 multiplications for 2x2:
    # Given A = [[a,b],[c,d]] and X = [[x00,x01],[x10,x11]] (a 2x2 block of B)
    # M1 = (a+d)(x00+x11)
    # M2 = (c+d)x00
    # M3 = a(x01-x11)
    # M4 = d(x10-x00)
    # M5 = (a+b)x11
    # M6 = (c-a)(x00+x01)
    # M7 = (b-d)(x10+x11)
    
    # Output C_block = [[C00,C01],[C10,C11]]:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the full computation, C is 2x10, and C[:, 2b:2b+2] = A * B[:, 2b:2b+2]
    
    # Let's enumerate all the rank-1 terms.
    # For each block b, we have 7 terms.
    # Total rank = 5 * 7 = 35.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    col = 0
    for b in range(5):  # 5 blocks
        # Indices in B for this block: rows 0,1; cols 2b, 2b+1
        # In V, index is j*p + k, so for block b:
        # V[0*p + 2b] = B[0, 2b]
        # V[0*p + 2b+1] = B[0, 2b+1]
        # V[1*p + 2b] = B[1, 2b]
        # V[1*p + 2b+1] = B[1, 2b+1]
        
        # In W, index is k*n + i, so for output C[i, 2b] and C[i, 2b+1]:
        # W[(2b)*n + 0] = C[0, 2b]
        # W[(2b)*n + 1] = C[1, 2b]
        # W[(2b+1)*n + 0] = C[0, 2b+1]
        # W[(2b+1)*n + 1] = C[1, 2b+1]
        
        # U indices: i*m + j
        # U[0*2+0] = A[0,0] = a
        # U[0*2+1] = A[0,1] = b
        # U[1*2+0] = A[1,0] = c
        # U[1*2+1] = A[1,1] = d
        
        # M1 = (a+d)(x00+x11)
        # U: a=1, d=1
        # V: x00=1, x11=1
        # W: contributes to C00 (+1) and C11 (+1)
        U[0][col] = 1  # a
        U[3][col] = 1  # d
        V[0 * p + 2*b][col] = 1    # x00
        V[1 * p + (2*b+1)][col] = 1 # x11
        W[(2*b) * n + 0][col] = 1   # C[0, 2b] += M1
        W[(2*b+1) * n + 1][col] = 1 # C[1, 2b+1] += M1
        col += 1
        
        # M2 = (c+d)x00
        U[2][col] = 1  # c
        U[3][col] = 1  # d
        V[0 * p + 2*b][col] = 1    # x00
        W[(2*b) * n + 1][col] = 1  # C[1, 2b] += M2
        col += 1
        
        # M3 = a(x01-x11)
        U[0][col] = 1  # a
        V[0 * p + (2*b+1)][col] = 1  # x01
        V[1 * p + (2*b+1)][col] = -1 # -x11
        W[(2*b+1) * n + 0][col] = 1  # C[0, 2b+1] += M3
        W[(2*b+1) * n + 1][col] = 1  # C[1, 2b+1] += M3
        col += 1
        
        # M4 = d(x10-x00)
        U[3][col] = 1  # d
        V[1 * p + 2*b][col] = 1    # x10
        V[0 * p + 2*b][col] = -1   # -x00
        W[(2*b) * n + 0][col] = 1  # C[0, 2b] += M4
        W[(2*b) * n + 1][col] = 1  # C[1, 2b] += M4
        col += 1
        
        # M5 = (a+b)x11
        U[0][col] = 1  # a
        U[1][col] = 1  # b
        V[1 * p + (2*b+1)][col] = 1 # x11
        W[(2*b) * n + 0][col] = -1  # C[0, 2b] -= M5
        W[(2*b+1) * n + 0][col] = 1 # C[0, 2b+1] += M5
        col += 1
        
        # M6 = (c-a)(x00+x01)
        U[2][col] = 1  # c
        U[0][col] = -1 # -a
        V[0 * p + 2*b][col] = 1    # x00
        V[0 * p + (2*b+1)][col] = 1 # x01
        W[(2*b+1) * n + 1][col] = 1 # C[1, 2b+1] += M6
        col += 1
        
        # M7 = (b-d)(x10+x11)
        U[1][col] = 1  # b
        U[3][col] = -1 # -d
        V[1 * p + 2*b][col] = 1    # x10
        V[1 * p + (2*b+1)][col] = 1 # x11
        W[(2*b) * n + 0][col] = 1  # C[0, 2b] += M7
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
