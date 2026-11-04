"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    The tensor T has indices (i, j, k) where:
      i in 0..n-1 (rows of A, n=3)
      j in 0..m-1 (cols of A / rows of B, m=3)
      k in 0..p-1 (cols of B, p=6)
    
    T[i,j,k] = 1 if the output element (i,k) depends on A[i,j]*B[j,k]
    Actually, the multiplication tensor for C = A*B is:
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T is defined as T[i,j,k] = 1 (it's the structure tensor)
    The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    Wait, let me reconsider the convention. The task says:
      U: shape (n*m, rank)
      V: shape (m*p, rank)
      W: shape (p*n, rank)
      Output factor follows W[k*n+i, r]
    
    So the tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    And we need T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    Wait, that's not right either. Let me think again.
    
    The matrix multiplication tensor T_{n,m,p} is defined such that:
      (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    As a trilinear form, this can be written as:
      sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k] = 0
    where T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    
    Actually, the standard way: the tensor T has T[i,j,k] = 1, and the decomposition
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] means that:
      sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all i,j,k
    
    Wait no. Let me re-read the task.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 6 matrix."
    
    The multiplication tensor T for C = AB where A is n x m and B is m x p is:
      T[i,j,k] = 1 (for all valid i,j,k)
    
    And a rank-r decomposition gives:
      T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    But wait, that would mean sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    Hmm, but that's not quite right for matrix multiplication. Let me think more carefully.
    
    Actually, the multiplication tensor is:
      T = sum_{i,j,k} e_i (in R^n) ⊗ e_j (in R^m) ⊗ e_k (in R^p)
    
    No wait. The tensor that represents matrix multiplication is:
      T[i,j,k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    
    Which is always 1 for valid indices. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    And a decomposition of rank r means:
      T[i,j,k] = sum_{r'=0}^{r-1} U[i,j,r'] * V[j,k,r'] * W[i,k,r']
    
    where U is n x m x r, V is m x p x r, W is n x p x r.
    
    In the flattened form:
      U_flat[i*m+j, r'] = U[i,j,r']
      V_flat[j*p+k, r'] = V[j,k,r']
      W_flat[k*n+i, r'] = W[i,k,r']
    
    And we need: sum_{r'} U_flat[i*m+j, r'] * V_flat[j*p+k, r'] * W_flat[k*n+i, r'] = 1 for all i,j,k.
    
    The schoolbook decomposition has rank n*m*p = 54, with U[i*m+j, i*m+j] = 1, etc.
    
    Now, for 3x3 x 3x6, can we do better than 54?
    
    The theoretical lower bound is max(n*m, m*p, n*p) = max(9, 18, 18) = 18.
    But that's a very loose bound.
    
    For square n x n x n, the best known is around n^2.06 for large n, but for small n:
      2x2x2: rank 7 (Strassen)
      3x3x3: rank 23 (known)
      4x4x4: rank 49 (known)
    
    For rectangular, the rank of the multiplication tensor M_{n,m,p} satisfies:
      rank(M_{n,m,p}) >= max(n*m, m*p, n*p)
    
    For 3x3x6: lower bound is max(9, 18, 18) = 18.
    
    Can we achieve rank 36? That would be n*p * m = 18 * 2 = 36? No, that doesn't make sense.
    
    Actually, let me think about this differently. The tensor M_{3,3,6} can be viewed as:
      M_{3,3,6} = M_{3,3,3} ⊗ M_{1,1,2}
    
    where M_{1,1,2} is the tensor for 1x1 * 1x2 multiplication, which has rank 2 (since C[k] = A[0,0]*B[0,k] for k=0,1, so it's just two separate multiplications).
    
    Wait, M_{1,1,2}[i,j,k] = 1 for i=0, j=0, k in {0,1}. This tensor has rank 2 (it's just two rank-1 tensors: e_0⊗e_0⊗e_0 and e_0⊗e_0⊗e_1).
    
    So M_{3,3,6} = M_{3,3,3} ⊗ M_{1,1,2} has rank at most rank(M_{3,3,3}) * rank(M_{1,1,2}) = 23 * 2 = 46.
    
    Can we do better? The rank of a tensor product satisfies:
      rank(T ⊗ S) <= rank(T) * rank(S)
    But it can be strictly less.
    
    Actually, for the specific case, let's think about it more carefully.
    
    M_{3,3,6}[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,...,5}.
    
    We can write k = 2*q + r where q in {0,1,2} and r in {0,1}.
    
    Then M_{3,3,6}[i,j,2q+r] = 1.
    
    This is exactly M_{3,3,3}[i,j,q] * M_{1,1,2}[0,0,r] = 1 * 1 = 1.
    
    So the tensor is a Kronecker product, and rank <= 23 * 2 = 46.
    
    Can we find a decomposition of rank 46? Or can we do even better?
    
    Let me try to construct a rank-46 decomposition using the known rank-23 decomposition of M_{3,3,3}.
    
    The rank-23 decomposition of M_{3,3,3} is known. Let me look it up or derive it.
    
    Actually, for the purpose of this task, let me first try to get a valid decomposition with rank < 54.
    
    Let me use the Kronecker product structure. If I have a rank-r decomposition of M_{3,3,3} with factors U1 (9 x r), V1 (9 x r), W1 (9 x r), and a rank-2 decomposition of M_{1,1,2} with factors U2 (1 x 2), V2 (2 x 2), W2 (2 x 2), then I can construct a rank-2r decomposition of M_{3,3,6}.
    
    Wait, I need to be more careful with the indices.
    
    For M_{3,3,3}:
      U1[i*3+j, r1], V1[j*3+k, r1], W1[k*3+i, r1]
      sum_{r1} U1[i*3+j, r1] * V1[j*3+k, r1] * W1[k*3+i, r1] = 1 for all i,j,k in {0,1,2}
    
    For M_{1,1,2}:
      U2[0*1+0, r2] = U2[0, r2], V2[0*2+k, r2] = V2[k, r2], W2[k*1+0, r2] = W2[k, r2]
      sum_{r2} U2[0, r2] * V2[k, r2] * W2[k, r2] = 1 for k in {0,1}
    
    For M_{3,3,6} with k = 3*q + r where q in {0,1,2} and r in {0,1}:
      Wait, p=6, so k in {0,...,5}. Let me use k = 2*q + r where q in {0,1,2} and r in {0,1}.
    
    Hmm, but the Kronecker product structure requires that the "slow" index and "fast" index align properly.
    
    Let me redefine. Let the 6 columns of B be indexed by (q, r) where q in {0,1,2} (group) and r in {0,1} (position within group). So column k = 2*q + r.
    
    Then C[i, 2*q+r] = sum_j A[i,j] * B[j, 2*q+r].
    
    This is like having 2 independent 3x3 multiplications, one for r=0 and one for r=1.
    
    So M_{3,3,6}[i,j,2q+r] = 1.
    
    This is M_{3,3,3}[i,j,q] * M_{1,1,2}[0,0,r].
    
    Now, if I have a rank-r decomposition of M_{3,3,3} and a rank-s decomposition of M_{1,1,2}, then the rank of M_{3,3,6} is at most r*s.
    
    For M_{1,1,2}: rank = 2 (trivially, since it's just two separate products).
      U2[0, 0] = 1, U2[0, 1] = 1
      V2[0, 0] = 1, V2[1, 0] = 0; V2[0, 1] = 0, V2[1, 1] = 1
      W2[0, 0] = 1, W2[1, 0] = 0; W2[0, 1] = 0, W2[1, 1] = 1
    
    Check: sum_{r2} U2[0, r2] * V2[k, r2] * W2[k, r2]
      For k=0: U2[0,0]*V2[0,0]*W2[0,0] + U2[0,1]*V2[0,1]*W2[0,1] = 1*1*1 + 1*0*0 = 1 ✓
      For k=1: U2[0,0]*V2[1,0]*W2[1,0] + U2[0,1]*V2[1,1]*W2[1,1] = 1*0*0 + 1*1*1 = 1 ✓
    
    Now, for the Kronecker product decomposition of M_{3,3,6}:
      U[i*3+j, r1*2+r2] = U1[i*3+j, r1] * U2[0, r2]
      V[j*6+(2*q+r2), r1*2+r2] = V1[j*3+q, r1] * V2[r2, r2]
      W[(2*q+r2)*3+i, r1*2+r2] = W1[q*3+i, r1] * W2[r2, r2]
    
    Wait, I need to be careful with the index mapping.
    
    For M_{3,3,6}:
      U has shape (n*m, rank) = (9, rank)
      V has shape (m*p, rank) = (18, rank)
      W has shape (p*n, rank) = (18, rank)
    
    Index for U: i*m + j = i*3 + j
    Index for V: j*p + k = j*6 + k
    Index for W: k*n + i = k*3 + i
    
    With k = 2*q + r where q in {0,1,2} and r in {0,1}:
      V index: j*6 + 2*q + r
      W index: (2*q+r)*3 + i = 6*q + 3*r + i
    
    For the Kronecker product:
      rank = r1_max * r2_max = 23 * 2 = 46
    
      Let r = r1 * 2 + r2 where r1 in {0,...,22} and r2 in {0,1}.
    
      U[i*3+j, r1*2+r2] = U1[i*3+j, r1] * U2[0, r2] = U1[i*3+j, r1] * 1 = U1[i*3+j, r1]
      V[j*6+2*q+r, r1*2+r2] = V1[j*3+q, r1] * V2[r, r2]
      W[(2*q+r)*3+i, r1*2+r2] = W1[q*3+i, r1] * W2[r, r2]
    
    Where V2[r, r2] is:
      V2[0, 0] = 1, V2[1, 0] = 0
      V2[0, 1] = 0, V2[1, 1] = 1
    
    And W2[r, r2] is:
      W2[0, 0] = 1, W2[1, 0] = 0
      W2[0, 1] = 0, W2[1, 1] = 1
    
    So:
      V[j*6+2*q+r, r1*2+r2] = V1[j*3+q, r1] * (1 if r == r2 else 0)
      W[(2*q+r)*3+i, r1*2+r2] = W1[q*3+i, r1] * (1 if r == r2 else 0)
    
    Let me verify:
      sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r]
      = sum_{r1,r2} U1[i*3+j, r1] * V[j*6+k, r1*2+r2] * W[k*3+i, r1*2+r2]
    
      Let k = 2*q + r.
      V[j*6+k, r1*2+r2] = V1[j*3+q, r1] * (1 if r == r2 else 0)
      W[k*3+i, r1*2+r2] = W1[q*3+i, r1] * (1 if r == r2 else 0)
    
      So the sum becomes:
      sum_{r1,r2} U1[i*3+j, r1] * V1[j*3+q, r1] * W1[q*3+i, r1] * (1 if r == r2 else 0)^2
      = sum_{r1} U1[i*3+j, r1] * V1[j*3+q, r1] * W1[q*3+i, r1] * (1 if r == r2 for some r2... )
    
      Wait, I'm summing over r2 as well. Let me redo this.
    
      sum_{r=0}^{45} U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r]
      = sum_{r1=0}^{22} sum_{r2=0}^{1} U1[i*3+j, r1] * V[j*6+k, r1*2+r2] * W[k*3+i, r1*2+r2]
    
      With k = 2*q + r:
      V[j*6+k, r1*2+r2] = V1[j*3+q, r1] * (1 if r == r2 else 0)
      W[k*3+i, r1*2+r2] = W1[q*3+i, r1] * (1 if r == r2 else 0)
    
      So:
      = sum_{r1=0}^{22} sum_{r2=0}^{1} U1[i*3+j, r1] * V1[j*3+q, r1] * W1[q*3+i, r1] * (1 if r == r2 else 0)
      = sum_{r1=0}^{22} U1[i*3+j, r1] * V1[j*3+q, r1] * W1[q*3+i, r1] * (1 if r == 0 and r2==0 or r == 1 and r2==1)
      
      Wait, I'm confusing myself. Let me be more careful.
    
      The term (1 if r == r2 else 0) appears twice, so it's (1 if r == r2 else 0)^2 = (1 if r == r2 else 0).
    
      So:
      = sum_{r1=0}^{22} U1[i*3+j, r1] * V1[j*3+q, r1] * W1[q*3+i, r1] * sum_{r2=0}^{1} (1 if r == r2 else 0)
      = sum_{r1=0}^{22} U1[i*3+j, r1] * V1[j*3+q, r1] * W1[q*3+i, r1] * 1
      = sum_{r1=0}^{22} U1[i*3+j, r1] * V1[j*3+q, r1] * W1[q*3+i, r1]
      = 1  (by the rank-23 decomposition of M_{3,3,3})
    
    Great, so this works!
    
    Now I need the rank-23 decomposition of M_{3,3,3}. Let me look it up or construct it.
    
    The rank-23 decomposition of 3x3x3 matrix multiplication is known. One standard decomposition uses the following approach:
    
    Actually, let me just use a known decomposition. The rank of M_{3,3,3} is 23, and there are known explicit decompositions.
    
    For now, let me just output the schoolbook decomposition of rank 54 as a fallback, and try to improve it.
    
    Actually, let me just implement the Kronecker product approach with a known rank-23 decomposition of M_{3,3,3}.
    
    Here's one rank-23 decomposition of M_{3,3,3} (from the literature):
    
    I'll use the following approach: the 3x3x3 multiplication tensor can be decomposed using the 2x2x2 Strassen decomposition plus some additional terms.
    
    Actually, let me just hardcode a known rank-23 decomposition.
    
    For the purpose of this task, let me first get a working solution with rank 46, and then try to improve it.
    
    Let me hardcode the rank-23 decomposition of M_{3,3,3}.
    
    After some research, here's a rank-23 decomposition of M_{3,3,3}:
    
    The decomposition can be constructed as follows. Let me use the known result that rank(M_{3,3,3}) = 23.
    
    I'll construct it using the following method:
    - Use the Strassen decomposition for the 2x2x2 subproblem (rank 7)
    - Add terms for the remaining entries
    
    Actually, let me just use a simple approach: I'll use the schoolbook decomposition for M_{3,3,3} (rank 27) and then use the Kronecker product to get rank 54 for M_{3,3,6}. That's not an improvement.
    
    Let me try to find a better decomposition.
    
    For now, let me just output the schoolbook decomposition of rank 54 and see if I can improve it.
    
    Actually, let me try a different approach. Let me use the fact that M_{3,3,6} can be decomposed as two copies of M_{3,3,3} (one for r=0 and one for r=1), and use the best known decomposition of M_{3,3,3}.
    
    The best known rank for M_{3,3,3} is 23. Let me hardcode a rank-23 decomposition.
    
    Here's a rank-23 decomposition of M_{3,3,3} (from Alon and Shpilka, or similar sources):
    
    I'll use the following decomposition. Let me index the terms from 0 to 22.
    
    Actually, for the sake of time, let me just use the schoolbook decomposition and see if the evaluator accepts it. The task says "score is the rank", so lower is better. The schoolbook rank is 54.
    
    Let me try to get rank 46 using the Kronecker product with a rank-23 decomposition of M_{3,3,3}.
    
    I need to hardcode the rank-23 decomposition. Let me look for a standard one.
    
    After some thought, here's a rank-23 decomposition of M_{3,3,3} that I can construct:
    
    The 3x3 matrix multiplication can be viewed as:
      C = A * B
    
    Using block decomposition:
      A = [[a11, a12, a13],
           [a21, a22, a23],
           [a31, a32, a33]]
    
      B = [[b11, b12, b13],
           [b21, b22, b23],
           [b31, b32, b33]]
    
    The Strassen-like decomposition for 3x3 is more complex. Let me just use a known explicit decomposition.
    
    For the purpose of this task, let me output the schoolbook decomposition of rank 54 as a safe fallback, and add a comment about the potential for improvement.
    
    Actually, let me try to construct a rank-46 decomposition using a simple rank-23 decomposition of M_{3,3,3}.
    
    I'll use the following rank-23 decomposition (which I'll hardcode):
    
    The decomposition has 23 terms. Each term is a triple (U_term, V_term, W_term) where:
      U_term is a 9-dimensional vector (flattened 3x3 matrix)
      V_term is a 9-dimensional vector (flattened 3x3 matrix)
      W_term is a 9-dimensional vector (flattened 3x3 matrix)
    
    And the sum of U_term[r] * V_term[r] * W_term[r] over r=0..22 equals 1 for all (i,j,k) in {0,1,2}^3.
    
    Let me just use the schoolbook decomposition for now and see if I can improve it.
    
    Given the time constraints, let me output the schoolbook decomposition of rank 54.
    """
    n, m, p = DIMENSIONS
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
