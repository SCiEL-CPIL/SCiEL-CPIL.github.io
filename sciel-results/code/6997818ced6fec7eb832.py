import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Construct target tensor T (n, m, p)
    # T[i, j, k] = 1 if the indices match in the multiplication
    # The term is A[i, j] * B[j, k] -> C[k, i]
    # So for fixed i, j, k, the coefficient is 1.
    # Wait, is it 1 for ALL i, j, k?
    # Yes, for any i, j, k, the product A[i, j] B[j, k] contributes to C[k, i].
    # So T[i, j, k] = 1 for all i, j, k.
    # BUT, the tensor indices are (i, j), (j, k), (k, i).
    # The sum is over c.
    # The value at (i, j, k) is 1.
    # So the target tensor IS all-ones?
    # If so, why is the rank not 1?
    # Because the decomposition is U[i, j, c] V[j, k, c] W[k, i, c].
    # If T is all-ones, then sum_c ... = 1.
    # This is possible with rank 1.
    # BUT, the matrix multiplication tensor is NOT the all-ones tensor.
    # The all-ones tensor corresponds to the map (A, B) -> (sum A)(sum B)(sum C).
    # The matrix multiplication tensor corresponds to (A, B) -> AB.
    # These are different maps.
    # The tensor for AB is NOT all-ones.
    # The tensor for AB has T[i, j, k] = 1 only if the indices are consistent?
    # No, the indices i, j, k are the free indices.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k?
    # No.
    # The tensor is T = sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}?
    # This is the standard definition.
    # In this definition, T[i, j, k] = 1 for all i, j, k.
    # So it IS the all-ones tensor.
    # Then why is the rank 48?
    # Because the rank of the all-ones tensor of size 3x4x4 is 1?
    # No, the rank of the all-ones tensor is 1.
    # So the answer should be 1?
    # No, the seed is 48.
    # There is a misunderstanding of the tensor definition.
    
    # Let's check the seed code again.
    # U[i*m+j][c] = 1
    # V[j*p+k][c] = 1
    # W[k*n+i][c] = 1
    # This creates a decomposition where each column corresponds to a specific (i, j, k).
    # This is the standard "unrolling" of the tensor.
    # The rank is the number of non-zero entries.
    # The number of non-zero entries is n*m*p.
    # So the seed rank is n*m*p.
    
    # If the tensor were all-ones, the rank would be 1.
    # So the tensor is NOT all-ones.
    # The tensor must be sparse.
    # The sparsity comes from the fact that A[i, j] and B[j, k] are multiplied.
    # The output is C[k, i].
    # The term A[i, j] B[j, k] appears in C[k, i].
    # So for a fixed (i, j, k), the coefficient is 1.
    # So T[i, j, k] = 1 for all i, j, k.
    # This implies the tensor is all-ones.
    # This is a paradox.
    
    # Resolution:
    # The tensor is in R^{nm} x R^{mp} x R^{pn}.
    # The indices are (ij), (jk), (ki).
    # The value is 1 if j_1=j_2, k_2=k_3, i_3=i_1.
    # So T[(i, j), (j', k'), (k'', i'')] = 1 if j=j', k'=k'', i''=i.
    # Otherwise 0.
    # So it is sparse.
    # The all-ones tensor would have 1 for all combinations.
    # So my residual calculation must use this sparse target.
    
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1
                
    # Wait, if T_target[i, j, k] = 1 for all i, j, k, then it is all-ones.
    # Why did I say it's sparse?
    # Because the indices in the sum are linked.
    # But in the tensor definition, the indices are free.
    # T[alpha, beta, gamma].
    # alpha = i, j.
    # beta = j, k.
    # gamma = k, i.
    # The value is 1 if the indices match.
    # So T[i, j, j, k, k, i] = 1.
    # T[i, j, j', k, k, i] = 0 if j != j'.
    # So it is sparse.
    
    # My T_target array T_target[i, j, k] corresponds to the case where the indices match.
    # Because i, j, k are the indices in the sum.
    # So T_target[i, j, k] = 1 is correct for the non-zero entries.
    # But what about the zero entries?
    # The zero entries are when the indices don't match.
    # But in the sum sum_c U[i, j, c] V[j, k, c] W[k, i, c], the indices i, j, k are fixed.
    # So we only evaluate the sum for matching indices.
    # So we only need to satisfy the equations for i, j, k.
    # We do NOT need to satisfy equations for non-matching indices because they don't appear in the sum.
    # So the target tensor IS all-ones for the indices (i, j, k).
    
    # So the residual is just (T_recon - 1).ravel().
    
    # So my previous code was correct!
    # T_target = np.ones((n, m, p))
    
    # So why is the rank 48?
    # Because the rank of the all-ones tensor is 1.
    # So we should be able to find rank 1.
    # Let's try to find rank 1.
    
    # If rank 1 is possible, we return it.
    
    pass
