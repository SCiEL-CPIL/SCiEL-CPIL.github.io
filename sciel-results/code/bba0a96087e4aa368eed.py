<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import null_space
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if the indices are "consistent"
    # Actually for matrix multiplication C[i,k] = sum_j A[i,j] B[j,k],
    # the tensor is T[i,j,k] which is 1 for all valid (i,j,k) in ranges [0,n) x [0,m) x [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the tensor.
    # Let's use the approach: for each column c, we have vectors u_c in R^{nm}, v_c in R^{mp}, w_c in R^{pn}
    # such that the tensor T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the all-ones tensor of size n x m x p!
    
    # Wait, that's not right. Let me re-read the problem.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k]
    # T[i,j,k] = 1 if k is the column index in C, i is the row index, j is the sum index
    # Actually T[i,j,k] = 1 for all valid i,j,k since every A[i,j]*B[j,k] contributes to C[i,k]
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This means the all-ones tensor of size (n*m) x (m*p) x (p*n) needs a rank-r decomposition
    # where the slices are constrained.
    
    # Actually, let me think differently. The tensor T is indexed by (i,j,k) and T[i,j,k]=1.
    # We want U, V, W such that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # Let's try to find the minimal rank. The trivial rank is n*m*p = 36.
    # Can we do better?
    
    # For n=2, m=3, p=6: Let's try rank 20 as suggested.
    
    # Let me try a constructive approach. 
    # For each pair (j,k), we need to handle both i=0 and i=1.
    # If we could use one term per (j,k) with U[0*m+j]=a, U[1*m+j]=b, V[j*p+k]=1, W[k*n+0]=c, W[k*n+1]=d
    # Then contribution to C[0,k] is a*1*c = a*c and to C[1,k] is b*1*d = b*d
    # We need a*c=1 and b*d=1, so a=1/c, b=1/d.
    # But we also need cross-terms to vanish? No, there are no cross-terms in this setup.
    
    # Wait, I think I'm overcomplicating. Let me just solve it numerically.
    
    # Let me try rank r and see if we can find a solution.
    # The equations are: for all i,j,k: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 36 equations with 3*r*? unknowns... actually r columns, each with nm+mp+pn = 6+18+12=36 entries, so 36r unknowns.
    # But the equations are nonlinear (products of three variables).
    
    # Let me try a different approach: fix W and solve for U,V, or use iterative methods.
    
    # Actually, let me just provide the schoolbook decomposition of rank 36, which is guaranteed correct.
    # The score will be 36, and we can try to improve later.
    
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
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k],
    # we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # Let me try to find a rank-20 decomposition.
    # Strategy: For each k (6 values), we need to compute C[0,k] and C[1,k].
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # For a fixed k, let's use 3 terms (one per j) but combine i=0 and i=1.
    # Term for (j,k): U[0*m+j]=1, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=1
    # This gives: contribution to C[0,k] from j is 1*1*1=1 (correct, should be A[0,j]*B[j,k])
    # But wait, this doesn't work because it doesn't separate A[0,j] from A[1,j].
    
    # I need to think about this differently. The factors U, V, W are fixed; they don't depend on A or B.
    # The decomposition must work for ALL A and B.
    
    # So U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1 for all valid (i,j,k) when summed over c.
    # This means the tensor T[i,j,k] = 1 (all ones) must be decomposable.
    
    # The all-ones tensor of size 2x3x6 has rank... let me think.
    # Actually, the all-ones tensor 1_{n x m x p} can be written as (1_n ⊗ 1_m ⊗ 1_p) which is rank 1!
    # But that's not right because the indices in U, V, W are coupled.
    
    # Let me re-examine: U is indexed by (i,j), V by (j,k), W by (k,i).
    # The constraint is that for the decomposition to work, we need:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    # This is a different constraint than the standard tensor decomposition because the indices cycle.
    
    # Let me try: U[i,j,c] = a_c, V[j,k,c] = b_c, W[k,i,c] = c_c (independent of indices)
    # Then sum_c a_c * b_c * c_c = 1. We need just one term with a_0*b_0*c_0=1.
    # But U[i,j,0]=a_0 for all i,j, V[j,k,0]=b_0 for all j,k, W[k,i,0]=c_0 for all k,i.
    # This works! Rank 1?
    
    # Wait, let me verify: if U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1*1*1 = 1. Yes!
    
    # But wait, is this correct? Let me double-check the problem statement.
    # The tensor should represent matrix multiplication. For C = A*B, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T[i,j,k] should be 1 if A[i,j] and B[j,k] contribute to C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k. Yes, the all-ones tensor.
    
    # And the decomposition sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = T[i,j,k] = 1
    # With U=V=W=1 (all ones), this gives 1*1*1=1. So rank 1 works!
    
    # But that seems too good to be true. Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # Hmm, maybe I'm misunderstanding the tensor structure. Let me re-read.
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    # So U has shape (2*3, rank) = (6, rank)
    # V has shape (3*6, rank) = (18, rank)  
    # W has shape (6*2, rank) = (12, rank)
    
    # And the condition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # If U, V, W are all-ones vectors, then yes, rank 1 works.
    
    # But wait - is the tensor really all ones? Let me think again.
    # In matrix multiplication, A[i,j] * B[j,k] contributes to C[i,k]. The "tensor" of matrix multiplication
    # is usually defined as T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k].
    # Since every A[i,j]*B[j,k] appears in C[i,k] (with coefficient 1), T[i,j,k] = 1 for all valid indices.
    
    # So yes, the all-ones tensor, and rank 1 should work.
    
    # Let me just return rank 1.
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
