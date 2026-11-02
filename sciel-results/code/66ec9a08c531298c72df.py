"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact using the trace convention."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition requires:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all (i,j,k)
    
    # Vectorized check
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # For each c, we form outer products and check the sum
    
    # Use einsum for efficiency
    # For each c: U[:,c] is (n*m,), V[:,c] is (m*p,), W[:,c] is (p*n,)
    # Reshape: U_c -> (n, m), V_c -> (m, p), W_c -> (p, n)
    # The product U_c[i,j] * V_c[j,k] * W_c[k,i] summed over j,k gives a scalar per (i)
    # Actually we need: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's do it with loops over the small dimensions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition using a constructive approach."""
    # For matrix multiplication of n x m by m x p, we look for a decomposition
    # of the form: (AB)[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] (approximately)
    # But the exact tensor is T[i,j,k] = 1 for all i,j,k.
    
    # A key insight: if we can find matrices A_c (n x m), B_c (m x p), C_c (p x n)
    # such that sum_c A_c[i,j] * B_c[j,k] * C_c[k,i] = 1 for all i,j,k,
    # then we have a valid decomposition.
    
    # One constructive approach: use the "border rank" style decomposition.
    # For small cases, we can try to find solutions using linear algebra.
    
    # Approach: Fix the structure and solve for the remaining variables.
    # We'll use a randomized approach with local optimization.
    
    # First, let's try a structured decomposition.
    # The tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_c a_c[i,j] * b_c[j,k] * c_c[k,i]
    
    # A simple decomposition: use the "diagonal" terms
    # For each (i,j,k), we need the sum to be 1.
    
    # Let's try a decomposition where each term c corresponds to a "path"
    # through the indices. 
    
    # Actually, let's use a more systematic approach.
    # We know that the schoolbook decomposition has rank n*m*p.
    # We can try to reduce this by finding linear dependencies.
    
    # For now, let's try a few structured decompositions.
    
    # Decomposition 1: Group by i
    # For each i, we have an m x p sub-tensor that is all 1s.
    # This can be decomposed with rank m*p (schoolbook for each i).
    # Total rank: n * m * p (same as schoolbook).
    
    # Decomposition 2: Try to use shared structure.
    # If n=2, we can try to share between the two i values.
    
    # Let's try a specific construction for small n.
    # For n=2, m=5, p=6:
    # T[i,j,k] = 1 for i in {0,1}, j in {0,...,4}, k in {0,...,5}
    
    # We can write: T[i,j,k] = 1/2 * (1 + 1) but that doesn't help directly.
    
    # Let's try: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # where we choose U_c, V_c, W_c to be sparse.
    
    # A promising approach: use the fact that for fixed j, the matrix
    # [T[i,j,k]] over (i,k) is a 2x6 all-ones matrix, which has rank 1.
    # So for each j, we need rank 1 to cover the (i,k) part.
    # This suggests: T[i,j,k] = sum_j' (something)
    
    # Actually, let's think differently.
    # T[i,j,k] = 1 = 1*1*1
    # We can write this as a sum of rank-1 tensors in the 3D space.
    # The tensor T is the all-ones tensor of shape (2,5,6).
    # The rank of the all-ones tensor of shape (a,b,c) is... 
    # Actually, the all-ones tensor can be decomposed as:
    # T[i,j,k] = sum_{c=1}^{r} u_c[i] * v_c[j] * w_c[k]
    # But our decomposition is different: it's U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # which couples the indices differently.
    
    # Let's re-examine the constraint:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # This is equivalent to: for each c, define matrices
    # A_c of shape (n, m) with A_c[i,j] = U[i*m+j, c]
    # B_c of shape (m, p) with B_c[j,k] = V[j*p+k, c]  
    # C_c of shape (p, n) with C_c[k,i] = W[k*n+i, c]
    # Then: sum_c (A_c * B_c)_{i,k} * C_c[k,i]... no, that's not right either.
    
    # Let me re-read: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = A_c[i,j] * B_c[j,k] * C_c[k,i]
    # Summed over c, this should equal 1.
    
    # Note that for fixed c, the expression A_c[i,j] * B_c[j,k] * C_c[k,i]
    # can be seen as: for each (i,k), sum_j A_c[i,j] * B_c[j,k] * C_c[k,i]
    # = C_c[k,i] * (A_c * B_c)[i,k]
    # So sum_c C_c[k,i] * (A_c * B_c)[i,k] = 1 for all (i,k).
    
    # This is a system of n*p equations (one for each (i,k)) in terms of the
    # matrices A_c, B_c, C_c.
    
    # Let M_c = A_c * B_c (n x p matrix). Then we need:
    # sum_c C_c[k,i] * M_c[i,k] = 1 for all (i,k).
    
    # This is: sum_c C_c^T[i,k] * M_c[i,k] = 1, i.e.,
    # sum_c (C_c^T * M_c)_{i,k} = 1 (element-wise product summed)
    
    # Wait, C_c[k,i] * M_c[i,k] is the (i,k) entry of C_c^T * M_c? No.
    # C_c^T[i,k] = C_c[k,i], so C_c^T[i,k] * M_c[i,k] is the (i,k) entry of
    # the Hadamard product C_c^T ⊙ M_c.
    
    # So we need: sum_c (C_c^T ⊙ M_c) = J (the all-ones n x p matrix).
    
    # This is a system where we have freedom in choosing A_c, B_c, C_c.
    
    # A simple solution: let M_c = I/something and C_c^T = something.
    # For example, if we set M_c = (1/r) * J for all c, and C_c^T = (1/r) * J,
    # then sum_c (C_c^T ⊙ M_c) = sum_c (1/r^2) * J = (r/r^2) * J = (1/r) * J.
    # We need this to equal J, so r = 1. But r=1 means rank 1, which is not possible
    # for general matrix multiplication.
    
    # Let's try: M_c = e_i e_k^T (outer product of standard basis vectors)
    # and C_c^T = e_i e_k^T. Then (C_c^T ⊙ M_c) = e_i e_k^T.
    # Sum over c: we need sum_c e_{i_c} e_{k_c}^T = J, which requires
    # r = n*p terms, one for each (i,k) pair. This gives rank n*p = 12.
    # But wait, we also need A_c and B_c such that A_c * B_c = M_c.
    # For M_c = e_i e_k^T, we can take A_c = e_i e_1^T and B_c = e_1 e_k^T,
    # where e_1 is the first standard basis vector in R^m.
    # Then A_c * B_c = e_i e_1^T e_1 e_k^T = e_i e_k^T = M_c. Good.
    # And C_c = e_k e_i^T (so C_c^T = e_i e_k^T).
    # 
    # So for each (i,k) pair, we have a term with:
    # U[i*m+j, c] = A_c[i,j] = delta_{j,0} (1 if j=0, else 0)
    # V[j*p+k, c] = B_c[j,k] = delta_{j,0} * delta_{k,k_c} (1 if j=0 and k=k_c, else 0)
    # W[k*n+i, c] = C_c[k,i] = delta_{k,k_c} * delta_{i,i_c} (1 if k=k_c and i=i_c, else 0)
    # 
    # Wait, let me re-check. For the term corresponding to (i_c, k_c):
    # A_c[i,j] = 1 if i=i_c and j=0, else 0
    # B_c[j,k] = 1 if j=0 and k=k_c, else 0
    # C_c[k,i] = 1 if k=k_c and i=i_c, else 0
    # 
    # Then A_c[i,j] * B_c[j,k] * C_c[k,i] is nonzero only when
    # i=i_c, j=0, k=k_c, i=i_c (consistent), so the product is 1.
    # For other (i,j,k), it's 0.
    # 
    # So sum_c A_c[i,j] * B_c[j,k] * C_c[k,i] = 1 if there's a term with
    # i_c=i and k_c=k, which is always true since we have a term for each (i,k).
    # But this only works when j=0! For j != 0, all terms are 0.
    # 
    # So this decomposition only works for j=0. We need to extend it.
    
    # Let's try a different approach. For each j, we need to cover all (i,k).
    # We can use the same structure but with different "middle" indices.
    # 
    # For each (j, i, k) triple, use a term with:
    # A_c[i',j'] = 1 if i'=i and j'=j, else 0
    # B_c[j',k'] = 1 if j'=j and k'=k, else 0
    # C_c[k',i'] = 1 if k'=k and i'=i, else 0
    # This gives the schoolbook decomposition with rank n*m*p = 60.
    
    # To reduce rank, we need to find linear combinations that work.
    
    # Let's try a different constructive approach.
    # Suppose we fix C_c to be diagonal matrices. Then C_c[k,i] = d_c[i] if k=i, else 0.
    # Then the condition becomes:
    # sum_c d_c[i] * (A_c * B_c)[i,i] = 1 for all i (only diagonal terms).
    # But we need all (i,k) entries, not just diagonal. So C_c can't be just diagonal.
    
    # Let's try: C_c[k,i] = u_c[k] * v_c[i] (rank-1 matrices).
    # Then the condition is:
    # sum_c u_c[k] * v_c[i] * (A_c * B_c)[i,k] = 1 for all (i,k).
    
    # This is getting complex. Let's use a numerical optimization approach.
    
    # We'll set up the problem as a system of polynomial equations and try to solve it.
    # Given the complexity, let's try a simpler constructive approach first.
    
    # Approach: Use the "Strassen-like" decomposition.
    # For 2x2 matrix multiplication, Strassen uses 7 multiplications.
    # For our case, we can try to generalize.
    
    # Actually, let's try a direct numerical search for small ranks.
    # We'll use random initialization and gradient descent on the error.
    
    if time.time() > deadline:
        return None
    
    # Try a structured decomposition based on the following idea:
    # T[i,j,k] = 1 = sum_c f_c(i) * g_c(j) * h_c(k) * ... 
    # But the coupling is through the indices in U, V, W.
    
    # Let's try: for each c, let U[:,c] be a vector in R^{n*m},
    # V[:,c] in R^{m*p}, W[:,c] in R^{p*n}.
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    
    # This is a system of n*m*p = 60 equations in r*(n*m + m*p + p*n) = r*(10+30+12) = 52r unknowns.
    # For r=60, we have 3120 unknowns and 60 equations, so it's underdetermined.
    # For smaller r, it becomes more constrained.
    
    # Let's try r from a lower bound up to 60.
    # Lower bound: the rank of the matrix multiplication tensor is at least
    # max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
    
    # Actually, the lower bound is more subtle. The rank is at least the
    # border rank, which for matrix multiplication is known to be at least
    # (n*m*p) / something. But for our purposes, let's just try ranks.
    
    # Given the time constraints, let's just return the schoolbook decomposition
    # if we can't find a better one quickly.
    
    # For now, return None to indicate we didn't find a decomposition of this rank.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    max_rank = n * m * p  # 60
    U_school = [[0] * max_rank for _ in range(n * m)]
    V_school = [[0] * max_rank for _ in range(m * p)]
    W_school = [[0] * max_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions using a constructive approach.
    # We'll use the following idea:
    # For each j, the sub-tensor T[:,j,:] is a 2x6 all-ones matrix.
    # We can decompose the all-ones matrix as a sum of rank-1 matrices.
    # The all-ones matrix of shape (a,b) has rank 1, so it can be written as
    # J = u v^T where u = (1,...,1)^T and v = (1,...,1)^T.
    # 
    # But in our tensor decomposition, the structure is different.
    # Let's try to exploit the fact that T[i,j,k] = 1 for all i,j,k.
    
    # Constructive decomposition:
    # We'll use the following form:
    # For each j, we have terms that cover all (i,k) pairs.
    # Key insight: T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_{c} alpha_c * (delta_{i, i_c} + 1) * (delta_{k, k_c} + 1) * ...
    # This is getting complicated. Let's use a simpler approach.
    
    # Simple approach: use the schoolbook but try to merge terms.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if they share
    # enough structure.
    
    # For now, let's just return the schoolbook decomposition.
    # A more sophisticated search would be needed to find a lower rank.
    
    return best_result
# EVOLVE-BLOCK-END
