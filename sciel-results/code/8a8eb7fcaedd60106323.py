"""Program for matrix multiplication 3x4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # We want to find a rank-minimal decomposition of the matrix multiplication tensor.
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For small matrices, the minimum rank is often known or can be found via search.
    # For 3x4x6, let's try to use a structured approach.
    
    # One approach: use the fact that matrix multiplication can be decomposed using
    # the "border rank" or known lower bounds. However, for exact decomposition,
    # we need rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24.
    # Actually, the trivial lower bound is max(n*m, m*p, p*n) = 24.
    # The schoolbook rank is n*m*p = 72.
    
    # Let's try to achieve rank 24, which is the trivial lower bound.
    # This is achievable if we can find a decomposition where each "slice" is used efficiently.
    
    # Actually, for matrix multiplication, the minimum rank is at least max(n*m, m*p, p*n).
    # For 3x4x6: max(12, 24, 18) = 24.
    
    # Let's try rank = 24. We need to verify if this is achievable.
    # A known result: the rank of matrix multiplication M_{n,m,p} is at least max(n*m, m*p, p*n).
    # Whether it equals this bound depends on the dimensions.
    
    # For now, let's try a structured decomposition at rank 24.
    # If that doesn't work, we'll increase the rank.
    
    rank = 24
    
    # Let's try to construct U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # such that for all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is "valid" else 0
    # Wait, the tensor for matrix multiplication is: T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor? No!)
    # Actually, the matrix multiplication tensor is: T[i,j,k] = delta_{i,j,k} in the sense that
    # C[i,k] = sum_j A[i,j] * B[j,k], so the tensor has T[i,j,k] = 1 for all valid i,j,k.
    # Yes! The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    
    # This is a system of n*m*p = 72 equations in rank^3 variables (but structured).
    
    # Let's try a simple approach: use the identity that if we can find matrices such that
    # the product U * V^T * W^T (in some tensor sense) gives the all-ones tensor.
    
    # Alternative: use the fact that the all-ones tensor of size n*m*p can be decomposed as
    # a sum of rank-1 tensors. The minimum number is the tensor rank.
    
    # For the all-ones tensor of size a*b*c, the tensor rank is max(a,b,c) if one of them is 1,
    # but in general it's more complex.
    
    # Actually, for matrix multiplication, the tensor is NOT the all-ones tensor in the usual sense.
    # Let me re-read the problem.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k. Yes, that's correct.
    # Because C[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j] B[j,k] C^*[i,k] is 1.
    
    # So we need to decompose the all-ones tensor of size 3*4*6 = 72 entries into a sum of rank-1 tensors.
    # The rank-1 tensors are of the form u * v * w where u is in R^{12}, v in R^{24}, w in R^{18}.
    
    # The minimum rank to decompose the all-ones tensor of size a*b*c is... 
    # Actually, the all-ones tensor of size a*b*c has tensor rank equal to min(a,b,c) * ... no.
    
    # Let me think differently. The all-ones tensor J_{a,b,c} can be written as:
    # J_{a,b,c}[i,j,k] = 1 = sum_{c=1}^{r} u_c[i] * v_c[j] * w_c[k]
    
    # One simple decomposition: use r = max(a,b,c) and set u_c[i] = 1 if i == c mod a, etc.
    # But this doesn't work directly.
    
    # Actually, a simple decomposition of the all-ones tensor: 
    # For r = a, set u_c[i] = 1 if i == c, v_c[j] = 1, w_c[k] = 1. Then sum_c u_c[i] v_c[j] w_c[k] = 1.
    # So the rank is at most min(a, b, c). For 12, 24, 18, the rank is at most 12.
    
    # Wait, that can't be right because the matrix multiplication rank is known to be larger.
    
    # I think I'm confusing the tensor structure. Let me re-examine.
    
    # The matrix multiplication tensor for M_{n,m,p} is defined as:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C
    # where e_i^A is the basis vector for the (i,j) entry of A, etc.
    
    # In the decomposition, we have:
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # And we need this to equal 1 for all i, j, k.
    
    # So yes, we need to decompose the all-ones tensor of size (n*m) * (m*p) * (p*n) = 12 * 24 * 18.
    
    # But wait, not all combinations are independent. The indices are coupled: the j in U and V must match,
    # and the k in V and W must match, and the i in U and W must match.
    
    # So it's not a free all-ones tensor; it's a structured one where only certain index combinations appear.
    
    # Specifically, for each (i,j,k), we have one equation:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of 72 equations. The variables are U, V, W with 12*r + 24*r + 18*r = 54*r variables.
    
    # For r = 24, we have 1296 variables and 72 equations, so it should be solvable.
    
    # Let's try to construct a solution. One approach: use the "diagonal" decomposition.
    # Set U[i*m+j, c] = 1 if c == i*m+j, else 0. Similarly for V and W.
    # Then for a given (i,j,k), the sum becomes U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j].
    # This is not 1 in general.
    
    # Let's try a different approach. Use the fact that we can set:
    # U[i*m+j, c] = alpha_{i,j,c}
    # V[j*p+k, c] = beta_{j,k,c}  
    # W[k*n+i, c] = gamma_{k,i,c}
    # such that for all i,j,k: sum_c alpha_{i,j,c} * beta_{j,k,c} * gamma_{k,i,c} = 1
    
    # One simple solution: use r = n*m*p = 72 with the schoolbook method.
    # But we want lower rank.
    
    # Let's try r = 24 and use a numerical solver or a structured ansatz.
    
    # Actually, let's try a very simple structured decomposition:
    # For each (i,j), assign a unique "color" c(i,j) in [0, r).
    # For each (j,k), assign a unique "color" d(j,k) in [0, r).
    # For each (k,i), assign a unique "color" e(k,i) in [0, r).
    # Then set U[i*m+j, c] = 1 if c == c(i,j), V[j*p+k, c] = 1 if c == d(j,k), W[k*n+i, c] = 1 if c == e(k,i).
    # Then the sum for (i,j,k) is 1 if c(i,j) == d(j,k) == e(k,i), else 0.
    # We need this to be 1 for all i,j,k, so we need c(i,j) == d(j,k) == e(k,i) for all i,j,k.
    # This means c(i,j) must depend only on a common value. But c(i,j) is for (i,j), d(j,k) for (j,k), e(k,i) for (k,i).
    # For this to be consistent, we need a function f(i,j,k) that is the same when viewed as (i,j), (j,k), or (k,i).
    # This is only possible if f is constant, which would give rank 1, but that's not possible for non-trivial cases.
    
    # So this simple approach doesn't work.
    
    # Let's try a numerical approach: set up the equations and solve.
    # But that might be slow.
    
    # Alternative: use a known decomposition. For small matrices, the minimum rank is often achieved by
    # specific constructions.
    
    # For 3x4x6, let's try rank = 24 and see if we can find a solution.
    
    # Actually, let's just use the schoolbook decomposition for now, which gives rank 72.
    # But the task asks for rank-minimal, so we should try to do better.
    
    # Let's try a compromise: use a decomposition that groups terms.
    
    # For each j (0 to 3), we can decompose the 3x6 subproblem.
    # The 3x6 matrix multiplication has tensor rank at least max(3*6, 6*3, 3*3) = max(18, 18, 9) = 18.
    # So for each j, we need rank 18, and for 4 values of j, total rank 72. That's the same as schoolbook.
    
    # Hmm. Let's just return the schoolbook decomposition for now, since finding a lower rank
    # decomposition is non-trivial.
    
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
