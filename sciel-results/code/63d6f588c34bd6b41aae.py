"""Matrix multiplication 2,4,7 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # safety margin before 150s wall-clock limit


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all indices."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy einsum."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True


def _construct_strassen_block(n, m, p):
    """
    Construct a Strassen-based decomposition for 2x4x7.
    
    The 2x4 matrix A is split into two 2x2 blocks: A = [A1 | A2]
    The 4x7 matrix B is split into four 2x? blocks.
    
    We use Strassen for the 2x2 multiplications where applicable.
    For 2x4 times 4x7:
    - A is 2x4: split into A_left (2x2) and A_right (2x2)
    - B is 4x7: split into B_top (2x7) and B_bottom (2x7)
    
    C = A_left * B_top + A_right * B_bottom
    
    Each 2x2 * 2x7 product can be done with Strassen-like approach.
    But 2x2 * 2x7 is not directly Strassen. Let me think differently.
    
    Actually, for 2x4 * 4x7, we can view it as:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    
    One approach: split B into B1 (2x7, rows 0-1) and B2 (2x7, rows 2-3)
    C = A1 * B1 + A2 * B2 where A1, A2 are 2x2.
    
    For each 2x2 * 2x7 product, we need 4 multiplications of 1x7 vectors.
    Total: 8 scalar-by-vector multiplications = 8 * 7 = 56 scalar mults.
    
    With Strassen on 2x2 * 2x2: 7 mults instead of 4.
    For 2x2 * 2x7: we can pad or use a different decomposition.
    
    Alternative: Use the known rank bound.
    The bilinear complexity of 2x4x7 is at most 28 (by flattening: min(n*p, m*p) approach doesn't directly apply).
    
    Let me try a direct construction with rank 28 or so.
    
    Actually, let me try the "middle" approach: 
    rank = max(n*m, m*p, p*n) isn't a valid upper bound for tensor rank of matmul.
    The tensor rank of matmul(n,m,p) is at most n*m*p (schoolbook).
    
    Known: rank of 2x2x2 is 7 (Strassen).
    For 2x4x7: we can decompose into blocks.
    
    Let me try: split the 4 in B into two groups of 2.
    C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    
    Each term is 2x2 @ 2x7.
    
    For 2x2 @ 2x7: 
    A = [[a, b], [c, d]], B = 2x7 matrix with rows b1, b2.
    Result row 1: a*b1 + b*b2 (each element)
    Result row 2: c*b1 + d*b2 (each element)
    
    This is 4 vector multiplications (a*b1, b*b2, c*b1, d*b2), each of length 7.
    As tensor rank: 4 * 7 = 28? No, each scalar*vector is rank 1 in the appropriate tensor.
    
    Actually, the tensor for 2x2*2x7 has rank at most 4*7 = 28? No.
    The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,...,6} with T=1 everywhere.
    This is the all-ones tensor of shape (2,2,7), which has tensor rank 2 (since it's separable: 
    T[i,j,k] = 1*1*1, so it's rank 1! Wait no, that's the outer product of three vectors of all ones).
    
    Oh wait, the matrix multiplication tensor is T[i,j,k] = 1 if the indices are coupled.
    For standard matrix multiplication, T[i,j,k] = 1 for ALL i,j,k (in the sense that every entry of C depends on 
    every product A[i,j]*B[j,k]). So the tensor IS the all-ones tensor.
    
    The all-ones tensor of shape (n,m,p) has tensor rank 1! Because:
    T[i,j,k] = 1 = 1 * 1 * 1 for all i,j,k.
    So U[i,j,r=0] = 1, V[j,k,r=0] = 1, W[k,i,r=0] = 1.
    
    But that gives C[i,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_j U[i,j,0]*V[j,0,k]*W[k,i,0]
    = sum_j 1*1*1 = m. Not 1!
    
    I need to be more careful. The matrix multiplication tensor is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In the tensor decomposition framework:
    We want C[i,k] = sum_r U[i,j,r]... wait, that's not right either.
    
    Let me re-read the convention:
    U: shape (n*m, rank) -> U[i*m + j, r]
    V: shape (m*p, rank) -> V[j*p + k, r]  
    W: shape (p*n, rank) -> W[k*n + i, r]
    
    And the constraint is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This means the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 for all i,j,k.
    
    So yes, T is the all-ones tensor of shape (n,m,p).
    
    The all-ones tensor (n,m,p) can be written as:
    T[i,j,k] = 1 = (1/sqrt(n)) * (1/sqrt(m)) * (1/sqrt(p)) * sqrt(n) * sqrt(m) * sqrt(p)
    
    Actually, it's just the outer product of three vectors of all ones:
    T[i,j,k] = 1_i * 1_j * 1_k where 1_i = 1 for all i.
    
    So it's rank 1! With U[i*m+j, 0] = 1/sqrt(n), V[j*p+k, 0] = 1/sqrt(m), W[k*n+i, 0] = 1/sqrt(p)?
    No: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = U[i,j,0]*V[j,k,0]*W[k,i,0] = 1*1*1 = 1.
    
    So rank 1 works! U = all 1s (column), V = all 1s (column), W = all 1s (column).
    
    But wait, that can't be right for matrix multiplication. Let me re-examine.
    
    The matrix multiplication tensor in the standard sense is:
    M_{i,j,k} = 1 if the indices are "coupled" in the multiplication, i.e., always.
    
    But the actual matrix multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    
    In the bilinear decomposition: C[i,k] = sum_r sum_j U[i,j,r] A[i,j] V[j,k,r] * (something)
    
    I think I'm confusing the tensor definition. Let me look at this more carefully.
    
    The standard tensor decomposition for matrix multiplication:
    We write C[i,k] = sum_j A[i,j] B[j,k]
    
    A bilinear decomposition of rank r writes:
    C[i,k] = sum_{r=1}^r sum_j u_{i,j,r} A[i,j] v_{j,k,r} w_{k,i,r}
    
    where the "w" part is just 1 typically... no.
    
    Actually, the proper formulation:
    The matrix multiplication tensor T is in U*V*W* space where:
    - U corresponds to A (size n*m)
    - V corresponds to B (size m*p)  
    - W corresponds to C (size n*p)
    
    T is defined by: T = sum_{i,j,k} e_{ij}^A ⊗ e_{jk}^B ⊗ e_{ik}^C
    
    The tensor rank is the minimum number of rank-1 tensors needed to sum to T.
    
    A rank-1 tensor is u^A ⊗ v^B ⊗ w^C where u is in A-space, v in B-space, w in C-space.
    
    So we need: sum_{r=1}^R (u_r^A ⊗ v_r^B ⊗ w_r^C) = T
    
    where T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ik}
    
    In coordinates: for each (i,j,k), the coefficient of e_{ij} ⊗ e_{jk} ⊗ e_{ik} in T is 1.
    
    For a rank-1 tensor u ⊗ v ⊗ w, the coefficient at (ij, jk, ik) is u_{ij} * v_{jk} * w_{ik}.
    
    So we need: sum_r u_{ij}^{(r)} * v_{jk}^{(r)} * w_{ik}^{(r)} = 1 for all i, j, k.
    
    This is exactly what the verification checks! And the all-ones solution works:
    u_{ij}^{(1)} = 1, v_{jk}^{(1)} = 1, w_{ik}^{(1)} = 1.
    
    So the tensor rank of the all-ones tensor (n,m,p) is 1.
    
    BUT WAIT. This seems too good. Let me verify with n=m=p=2.
    T[i,j,k] = 1 for all i,j,k in {0,1}.
    Rank 1: u_{ij}=1, v_{jk}=1, w_{ik}=1. sum = 1*1*1 = 1. ✓
    
    But the Strassen algorithm says 2x2x2 has rank 7, not 1!
    
    The issue: the matrix multiplication tensor is NOT the all-ones tensor!
    
    Let me reconsider. The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix:
    
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{ij}^{(A)} ⊗ E_{jk}^{(B)} ⊗ E_{ik}^{(C)}
    
    where E_{ij}^{(A)} is the matrix unit in the A-space (position (i,j) of the n×m matrix),
    E_{jk}^{(B)} is the matrix unit in the B-space (position (j,k) of the m×p matrix),
    E_{ik}^{(C)} is the matrix unit in the C-space (position (i,k) of the n×p matrix).
    
    Now, when we flatten:
    - A-space has dimension n*m, indexed by (i,j) -> index i*m+j
    - B-space has dimension m*p, indexed by (j,k) -> index j*p+k
    - C-space has dimension n*p, indexed by (i,k) -> index i*p+k
    
    The tensor T as a trilinear form: for inputs a in A-space, b in B-space, c in C-space,
    T(a,b,c) = sum_{i,j,k} T_{ij,jk,ik} a_{ij} b_{jk} c_{ik}
    
    But what is T_{ij,jk,ik}? It's 1 for all valid i,j,k (since each triple (i,j,k) appears exactly once in the sum defining C).
    
    So T(a,b,c) = sum_{i,j,k} a_{ij} b_{jk} c_{ik}
    
    For the decomposition: T(a,b,c) = sum_r (sum_{ij} u_{ij}^{(r)} a_{ij}) (sum_{jk} v_{jk}^{(r)} b_{jk}) (sum_{ik} w_{ik}^{(r)} c_{ik})
    
    We need: sum_r u_{ij}^{(r)} v_{jk}^{(r)} w_{ik}^{(r)} = 1 for all i,j,k.
    
    This IS the all-ones condition. And rank 1 works with all u=v=w=1.
    
    But this contradicts Strassen! The answer must be that I'm misunderstanding the tensor.
    
    AH, I see the issue. In the standard formulation, the tensor for matrix multiplication is:
    
    T = sum_{i,j,k} E_{ij} ⊗ E_{jk} ⊗ E_{ki}  (note: E_{ki} not E_{ik}!)
    
    OR the convention is different. Let me re-read the task:
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed by (k, i) with row index k*n+i. This means W[k*n+i, r] corresponds to the (k,i) position.
    
    In the standard matrix multiplication C = A*B where A is n×m and B is m×p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The C-space should be indexed by (i,k). But the task says W has shape (p*n, rank) with row k*n+i.
    So W's row index is k*n+i, meaning the first index is k (range p) and second is i (range n).
    So W is indexed by (k, i), not (i, k).
    
    This is a transposed convention for the output. The tensor element is T[i, j, k] where:
    - i indexes rows of A (and C)
    - j indexes columns of A (and rows of B)
    - k indexes columns of B (and C)
    
    But W is indexed by (k, i) instead of (i, k).
    
    So the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    This is still the all-ones condition on (i,j,k). The reindexing of W doesn't change the fact that 
    we need the sum to be 1 for every (i,j,k).
    
    So the answer should be rank 1? But that can't be the intent of the problem.
    
    Let me re-read: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 7 matrix."
    
    I think the issue is that the tensor is NOT all ones. Let me reconsider.
    
    In the standard tensor decomposition of matrix multiplication, the tensor is:
    
    M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j}^{(n,m)} ⊗ E_{j,k}^{(m,p)} ⊗ E_{i,k}^{(n,p)}
    
    When we view this as a 3-tensor with dimensions (n*m) × (m*p) × (n*p), the entry at position 
    (i*m+j, j*p+k, i*p+k) is 1, and all other entries are 0.
    
    So it's NOT the all-ones tensor! It's a sparse tensor where only specific positions are 1.
    
    Specifically, T[(i*m+j), (j*p+k), (i*p+k)] = 1 for valid i,j,k, and 0 otherwise.
    
    But in the verification formula: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Note that W is indexed by k*n+i, not i*p+k. With n=2, p=7: k*n+i ranges over k*2+i for k in 0..6, i in 0..1.
    This gives indices 0..13. But i*p+k would give i*7+k for i in 0..1, k in 0..6, also 0..13.
    
    The difference: k*n+i vs i*p+k. These are different orderings of the same pair (i,k) when n≠p.
    
    So the tensor in the (n*m)×(m*p)×(p*n) space has:
    T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k.
    
    This is still "all ones" in the sense that for every combination of (i,j,k), the corresponding 
    position in the 3D array (with the specific index mappings) is 1.
    
    And the verification checks exactly this: for each (i,j,k), the sum equals 1.
    
    So we're back to: the all-ones condition, which has rank 1.
    
    I'm clearly missing something. Let me think about what makes matrix multiplication non-trivial.
    
    OH WAIT. I think the issue is that the tensor is not just about the values being 1, but about the 
    LINEAR structure. The matrix multiplication is a bilinear map: (A, B) -> C where C = A*B.
    
    The tensor decomposition is decomposing the BILINEAR MAP, not the all-ones tensor.
    
    In the bilinear map formulation:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r bilinear decomposition is:
    C[i,k] = sum_{r=1}^r (sum_j u_{ij}^{(r)} A[i,j]) * (sum_j v_{jk}^{(r)} B[j,k])
    
    There's no "W" factor in the standard bilinear decomposition of matrix multiplication!
    The output C is computed as a sum of products of two linear forms.
    
    But the task has THREE factors U, V, W. This is a trilinear decomposition, which is different.
    
    In a trilinear decomposition:
    C[i,k] * (some linear form in C) = sum_r (linear form in A) * (linear form in B) * (linear form in C)
    
    Or more precisely, the tensor T satisfies:
    T(a, b, c) = sum_r a(u_r) * b(v_r) * c(w_r)
    
    where T(a,b,c) = sum_{i,j,k} a_{ij} b_{jk} c_{ik} (with the appropriate index conventions).
    
    And we need this to hold for ALL a, b, c.
    
    This means: sum_{i,j,k} a_{ij} b_{jk} c_{ik} = sum_r (sum_{ij} u_{ij}^{(r)} a_{ij}) (sum_{jk} v_{jk}^{(r)} b_{jk}) (sum_{ik} w_{ik}^{(r)} c_{ik})
    
    Expanding the RHS: sum_r sum_{i,j,j',j'',k,k'} u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)} a_{ij} b_{j'k'} c_{ik'}
    
    For this to equal the LHS for all a,b,c, we need the coefficient of a_{ij} b_{jk} c_{ik} to be 1 and all other coefficients to be 0.
    
    Coefficient of a_{ij} b_{j'k'} c_{ik'} on RHS:
    sum_r u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)} [where the indices must match: j=j', k=k']
    
    Wait, more carefully:
    LHS coefficient of a_{ij} b_{j'k'} c_{ik'}: this is 1 if j=j' and k=k', else 0. (Since LHS = sum_{i,j,k} a_{ij}b_{jk}c_{ik})
    
    Actually LHS = sum_{i,j,k} a_{ij} b_{jk} c_{ik}. The coefficient of a_{ij} b_{j'k'} c_{ik'} is 1 if (j=j' AND k=k'), else 0.
    
    RHS coefficient of a_{ij} b_{j'k'} c_{ik'}:
    sum_r u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)}
    
    Wait no. Let me redo this.
    
    RHS = sum_r (sum_{i1,j1} u_{i1,j1}^{(r)} a_{i1,j1}) * (sum_{j2,k2} v_{j2,k2}^{(r)} b_{j2,k2}) * (sum_{i3,k3} w_{i3,k3}^{(r)} c_{i3,k3})
    
    = sum_r sum_{i1,j1,j2,k2,i3,k3} u_{i1,j1}^{(r)} v_{j2,k2}^{(r)} w_{i3,k3}^{(r)} a_{i1,j1} b_{j2,k2} c_{i3,k3}
    
    The coefficient of a_{ij} b_{j'k'} c_{ik'} is:
    sum_r u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)}
    
    (Setting i1=i, j1=j, j2=j', k2=k', i3=i, k3=k'. Note i3 must equal i and k3 must equal k'.)
    
    For this to equal the LHS coefficient (which is 1 if j=j' and k=k', else 0), we need:
    sum_r u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)} = 1 if j=j' and k=k'
    sum_r u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)} = 0 if j≠j' or k≠k'
    
    This is a MUCH stronger condition than just sum_r u_{ij}^{(r)} v_{jk}^{(r)} w_{ik}^{(r)} = 1!
    
    The verification in the code only checks the case j=j', k=k' (i.e., the diagonal). It doesn't check the off-diagonal conditions!
    
    So the verification function _verify_fast is INCORRECT. It only checks:
    for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    But it should ALSO check:
    for each (i,j,k,j',k') with j≠j' or k≠k': sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i, r] == 0
    
    This is why rank 1 (all ones) passes the current verification but is NOT a valid matrix multiplication decomposition!
    
    With all ones: sum_r u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)} = 1*1*1 = 1 for ALL i,j,j',k',k'.
    But we need it to be 0 when j≠j' or k≠k'.
    
    So the correct condition is much more restrictive, and the tensor rank is NOT 1.
    
    Now I understand. The current verification is buggy - it only checks a subset of the conditions.
    But the task says "exactly decompose the matrix multiplication tensor", so the evaluator must be checking the full condition.
    
    Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    So the evaluator checks the full tensor decomposition, not just the diagonal.
    
    The schoolbook decomposition (rank 56) has:
    For each (i,j,k), one rank-1 term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1, and all other entries 0.
    
    This gives: sum_r u_{ij}^{(r)} v_{j'k'}^{(r)} w_{ik'}^{(r)} = 1 if there exists r with u_{ij}^{(r)}=v_{j'k'}^{(r)}=w_{ik'}^{(r)}=1, else 0.
    In schoolbook, for a given (i,j,k), the r index is uniquely determined, and u_{ij}^{(r)}=1 only when r corresponds to (i,j,k), similarly for v and w. So the sum is 1 iff j=j' AND k=k' AND i=i (always true), which gives 1 when j=j',k=k' and 0 otherwise. ✓
    
    So the full condition is:
    sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i, r] = δ_{j,j'} * δ_{k,k'}
    for all i, j, j', k, k'.
    
    This is equivalent to: the matrices U^T * V (with appropriate slicing) satisfy certain orthogonality conditions.
    
    Specifically, for fixed i:
    sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i, r] = δ_{j,j'} δ_{k,k'}
    
    Let's define for fixed i:
    U_i[j, r] = U[i*m + j, r] (m × rank matrix)
    V[i][j', k', r] = V[j'*p + k', r] but this doesn't depend on i...
    
    Hmm, V doesn't depend on i. Let me think differently.
    
    For fixed i, k':
    sum_r U[i*m+j, r] * W[k'*n+i, r] * V[j'*p+k', r] = δ_{j,j'} * δ_{k,k'}
    
    This is like: (U_i * diag(W_{k',i}) * V^T)[j, j', k'] = δ_{j,j'} δ_{k,k'}
    
    where U_i is m×rank, W_{k',i} is a row of W (length rank), and V^T is rank×(m*p).
    
    Actually, let me define:
    For fixed i: M_i = U_i ⊙ W_i (element-wise product of m×rank and 1×rank, broadcast to m×rank)
    Then: sum_r M_i[j, r] * V[j'*p+k', r] = δ_{j,j'} δ_{k,k'}
    
    This means: M_i * V^T (as a matrix product) should give a specific pattern.
    (M_i * V^T)[j, j'*p+k'] = sum_r M_i[j,r] * V[j'*p+k', r] = δ_{j,j'} δ_{k,k'}
    
    But the RHS depends on k, which is the column index of V. So:
    (M_i * V^T)[j, j'*p+k] = δ_{j,j'} δ_{k,k}
    
    This means M_i * V^T should be a block diagonal matrix where the (j,j') block is δ_{j,j'} * I_p.
    
    In other words: M_i * V^T = I_m ⊗ I_p (the Kronecker product)
    
    So: (U_i ⊙ W_i) * V^T = I_m ⊗ I_p for each i.
    
    Equivalently: V * (U_i ⊙ W_i)^T = I_m ⊗ I_p
    
    This means V^T has a left inverse that depends on i, which constrains the structure significantly.
    
    For this to work, V must have rank at least m*p (since V^T is rank×(m*p) and we need its left null space to be trivial in a specific way). Actually, V * (U_i ⊙ W_i)^T = I_{mp} means that V has full row rank mp, so rank >= mp = 28.
    
    So the lower bound is rank >= m*p = 28.
    
    Similarly, we can derive rank >= n*m = 8 and rank >= n*p = 14 from other index arrangements.
    
    The tightest lower bound is rank >= m*p = 28.
    
    Can we achieve rank 28? That would be optimal.
    
    Let me try to construct a rank-28 decomposition.
    
    We need V to be 28×28 (full rank), and for each i, (U_i ⊙ W_i) = (V^T)^{-1} (I_m ⊗ I_p) = (V^T)^{-1}.
    
    Wait, that's not quite right. Let me redo:
    
    M_i * V^T = I_{mp} where M_i = U_i ⊙ W_i (m×rank).
    
    If rank = mp = 28, then V^T is 28×28. If V is invertible, then M_i = (V^T)^{-1} = (V^{-1})^T.
    
    But M_i depends on i, while (V^{-1})^T doesn't. So M_i must be the same for all i!
    
    M_i[j, r] = U[i*m+j, r] * W[k*n+i, r]... wait, W_i depends on i through the row index k*n+i.
    
    Hmm, I think I made an error. Let me redo more carefully.
    
    The condition is: for all i, j, j', k, k':
    sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i, r] = δ_{j,j'} δ_{k,k'}
    
    Fix i and k'. Then for all j, j', k:
    sum_r U[i*m+j, r] * W[k'*n+i, r] * V[j'*p+k', r] = δ_{j,j'} δ_{k,k'}
    
    Define A_i = U reshaped as (n, m, rank), so A_i[j, r] = U[i*m+j, r].
    Define B = V reshaped as (m, p, rank), so B[j', k', r] = V[j'*p+k', r].
    Define C = W reshaped as (p, n, rank), so C[k', i, r] = W[k'*n+i, r].
    
    Condition: for all i, k': sum_r A_i[j, r] * C[k', i, r] * B[j', k', r] = δ_{j,j'} δ_{k,k'}
    
    For fixed (i, k'): let D_i,k'[j, r] = A_i[j, r] * C[k', i, r].
    Then: sum_r D_i,k'[j, r] * B[j', k', r] = δ_{j,j'} δ_{k,k'}
    
    This means: D_i,k' * B_{k'}^T = I_m ⊗ I_p (as a matrix product, where B_{k'} is the p-th "block" of B for fixed k'... )
    
    Actually B[j', k', r] for fixed k' is an m×rank matrix (indexing j' and r). Let's call it B_{k'} (m×rank).
    D_i,k' is m×rank.
    D_i,k' * B_{k'}^T is m×m... no, D_i,k' is m×rank, B_{k'}^T is rank×m, product is m×m.
    But we need it to be m×(m*p) = m×28.
    
    I think I'm getting confused with the matrix shapes. Let me just focus on the scalar condition.
    
    For fixed (i, k'), the condition over (j, j', k) is:
    sum_r D_i,k'[j, r] * B[j', k', r] = δ_{j,j'} δ_{k,k'}
    
    This is m * m * p = 2*2*7 = 28 equations in rank unknowns (per j).
    
    For the schoolbook with rank=56: each (i,j,k) has its own r, so the condition is trivially satisfied.
    
    For a lower rank, we need to find a non-trivial solution.
    
    Given the complexity of finding an explicit low-rank solution, and given that the current program scores 1.0 (schoolbook, rank 56), let me try a different approach:
    
    Use the known Strassen decomposition for 2×2×2 (rank 7) and extend it.
    
    For 2×4×7:
    Split A (2×4) into two 2×2 blocks: A = [A1 A2] where A1, A2 are 2×2.
    Split B (4×7) into two 2×7 blocks: B = [B1; B2] where B1, B2 are 2×7.
    
    C = A1*B1 + A2*B2
    
    Now, 2×2 * 2×7: this is a 2×2 matrix times a 2×7 matrix.
    Result is 2×7.
    
    For 2×2 * 2×7, we can use:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is just 2 multiplications per output element, 7 output elements, 2 rows = 28 scalar multiplications.
    As a tensor: the 2×2×7 all-ones tensor (with the correct index coupling).
    
    Hmm, this is getting complicated. Let me just try to use the corrected verification and search for a valid decomposition.
    
    Actually, given the time constraints and the complexity, let me just implement the CORRECT verification and use the schoolbook as baseline, then try a few structured approaches.
    
    The key insight from my analysis: the lower bound is rank >= m*p = 28, and schoolbook is rank 56.
    There might be a decomposition with rank between 28 and 56.
    
    Let me try a block-based approach:
    - For the 2×2×2 sub-problem, use Strassen (rank 7).
    - For the remaining dimensions, use schoolbook.
    
    2×4×7 can be viewed as:
    - 2 output rows (i=0,1)
    - 4 input columns from A (j=0,1,2,3)
    - 7 output columns (k=0,...,6)
    
    One approach: group the j indices into pairs: (0,1) and (2,3).
    Group the k indices: we can't group 7 into pairs nicely.
    
    Alternative: just use schoolbook for the k dimension (7 terms) and Strassen for the 2×2×2 part.
    
    For fixed k: C[:,k] = A * B[:,k] where A is 2×4, B[:,k] is 4×1.
    This is a 2×4 × 4×1 multiplication, which is just a matrix-vector product.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    For each k, this is 8 multiplications (2×4). Total: 7*8 = 56.
    
    With Strassen on the 2×2 blocks:
    A = [A11 A12; A21 A22] (2×2 blocks), B[:,k] = [B1k; B2k] (2×1 blocks).
    
    C1k = A11*B1k + A12*B2k
    C2k = A21*B1k + A22*B2k
    
    Each "multiplication" here is 1×1 * 1×1 = scalar. So 4 scalar mults per k, total 28.
    But that's just schoolbook again (no Strassen benefit for 2×2 * 2×1).
    
    Strassen helps when both matrices are at least 2×2.
    
    Let me try: group k into pairs (0,1), (2,3), (4,5), and handle k=6 separately.
    
    For (0,1): C[:,0:2] = A * B[0:2, 0:2] where B[0:2,0:2] is 4×2.
    This is 2×4 * 4×2 = 2×2. Using Strassen: 7 mults.
    Similarly for (2,3): 7 mults, (4,5): 7 mults.
    For k=6: C[:,6] = A * B[:,6], which is 2×4 * 4×1 = 8 mults (schoolbook).
    
    Total: 7+7+7+8 = 29 mults.
    
    As a tensor decomposition, this would have rank 29? Let me verify this is valid.
    
    The Strassen decomposition for 2×2×2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A11+A22)B21
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21+A22)(B11-B12)
    M7 = (A11-A21)(B12+B22)
    
    C11 = M1 - M4 + M5 - M7
    C12 = M3 + M7
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 2×4 * 4×2, with A = [A1 A2] (A1, A2 are 2×2) and B = 4×2:
    C = A1 * B_top + A2 * B_bottom where B_top, B_bottom are 2×2.
    
    Using Strassen on each 2×2*2×2: 7+7 = 14 mults.
    But we can share some terms? Probably not easily.
    
    Actually, for 2×4 * 4×2:
    A = [A11 A12 A21 A22] (each 2×2), B = [B1; B2] (each 2×2).
    C = A11*B1 + A22*B2... no, A is 2×4 so it's [A1 A2] where A1, A2 are 2×2.
    B is 4×2 = [B1; B2] where B1, B2 are 2×2.
    C = A1*B1 + A2*B2 (2×2 result).
    
    Strassen on A1*B1: 7 mults, Strassen on A2*B2: 7 mults. Total 14.
    
    For the full 2×4*4×7:
    - k in {0,1}: C[:,0:2] = A1*B[0:2,0:2] + A2*B[2:4,0:2]. Each 2×2*2×2, Strassen: 7+7=14.
    - k in {2,3}: similar, 14.
    - k in {4,5}: similar, 14.
    - k=6: C[:,6] = A[:,0:2]*B[0:2,6] + A[:,2:4]*B[2:4,6]. 2×2*2×1 each, schoolbook: 4+4=8.
    
    Total: 14+14+14+8 = 50.
    
    This matches the knowledge note: "Replaces schoolbook rank 56 with a block-structured Strassen decomposition of rank 50."
    
    Let me construct this rank-50 decomposition.
    
    For the Strassen part (2×2*2×2), the 7 multiplications are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A11+A22)B21
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21+A22)(B11-B12)
    M7 = (A11-A21)(B12+B22)
    
    In the tensor decomposition, each "multiplication" M corresponds to a rank-1 term.
    The A-part is a linear form in A, the B-part is a linear form in B, and the C-part distributes the result.
    
    For the full tensor, each rank-1 term has:
    - A coefficient: a 2×4 matrix (sparse, with specific pattern)
    - B coefficient: a 4×7 matrix (sparse)
    - C coefficient: a 2×7 matrix (sparse, with ±1 entries)
    
    Let me build this step by step.
    
    For the k-pair (k1, k2) (e.g., k1=0, k2=1):
    C[:, k1:k2+1] = A1 * B[0:2, k1:k2+1] + A2 * B[2:4, k1:k2+1]
    
    where A1 = A[:, 0:2] (2×2), A2 = A[:, 2:4] (2×2),
    B[0:2, k1:k2+1] is 2×2, B[2:4, k1:k2+1] is 2×2.
    
    Strassen on A1 * B_top (2×2*2×2):
    Let A1 = [[a, b], [c, d]] (rows 0,1 of A, cols 0,1)
    Let B_top = [[e, f], [g, h]] (rows 0,1 of B, cols k1,k2)
    
    M1 = (a+d)(e+h)
    M2 = (a+d)g
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c+d)(e-f)
    M7 = (a-c)(f+h)
    
    C1k1 = M1 - M4 + M5 - M7
    C1k2 = M3 + M7
    C2k1 = M2 + M4
    C2k2 = M1 - M2 + M3 + M6
    
    Similarly for A2 * B_bottom.
    
    Now, in the tensor decomposition, each M is a rank-1 term.
    For M1 from A1*B_top:
    A-part: a+d = A[0,0] + A[1,1] (linear form on A)
    B-part: e+h = B[0,k1] + B[1,k2] (linear form on B)
    C-part: contributes to C[0,k1] with +1, C[1,k2] with +1
    
    So U for this term: U[0*4+0, r]=1, U[1*4+1, r]=1, rest 0.
    V for this term: V[0*7+k1, r]=1, V[1*7+k2, r]=1, rest 0.
    W for this term: W[k1*2+0, r]=1, W[k2*2+1, r]=1, rest 0.
    
    Let me code this up.
    """
    n, m, p = 2, 4, 7
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    
    # Define Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h): A-part: a,d; B-part: e,h; C: C11+=1, C22+=1
    # M2 = (a+d)g:    A-part: a,d; B-part: g;    C: C21+=1
    # M3 = a(f-h):     A-part: a;   B-part: f,-h; C: C12+=1, C22+=1
    # M4 = d(g-e):     A-part: d;   B-part: g,-e; C: C21+=1, C22+=-1
    # M5 = (a+b)h:     A-part: a,b; B-part: h;    C: C11+=1
    # M6 = (c+d)(e-f): A-part: c,d; B-part: e,-f; C: C22+=1
    # M7 = (a-c)(f+h): A-part: a,-c; B-part: f,h; C: C11+=-1, C12+=1, C21+=-1
    # Wait let me recheck:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: 4 values for [a, b, c, d] (positions in 2x2)
    # B_coeffs: 4 values for [e, f, g, h] (positions in 2x2)
    # C_coeffs: 4 values for [C11, C12, C21, C22]
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a+d)g
        ([1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]),
        # M3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = d(g-e)
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, -1]),
        # M5 = (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
        # M6 = (c+d)(e-f)
        ([0, 0, 1, 1], [1, -1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a-c)(f+h)
        ([1, 0, -1, 0], [0, 1, 0, 1], [-1, 1, -1, 0]),
    ]
    
    # For each k-pair, apply Strassen to both A1*B_top and A2*B_bottom
    # k-pairs: (0,1), (2,3), (4,5)
    for k1, k2 in [(0, 1), (2, 3), (4, 5)]:
        # A1 * B_top: A1 = A[:, 0:2], B_top = B[0:2, k1:k2+1]
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # A-part: a_coeffs maps to A[:, 0:2]
            # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
            U[0 * m + 0, r] = a_coeffs[0]  # A[0,0]
            U[0 * m + 1, r] = a_coeffs[1]  # A[0,1]
            U[1 * m + 0, r] = a_coeffs[2]  # A[1,0]
            U[1 * m + 1, r] = a_coeffs[3]  # A[1,1]
            
            # B-part: b_coeffs maps to B[0:2, k1:k2+1]
            # e=B[0,k1], f=B[0,k2], g=B[1,k1], h=B[1,k2]
            V[0 * p + k1, r] = b_coeffs[0]  # B[0,k1]
            V[0 * p + k2, r] = b_coeffs[1]  # B[0,k2]
            V[1 * p + k1, r] = b_coeffs[2]  # B[1,k1]
            V[1 * p + k2, r] = b_coeffs[3]  # B[1,k2]
            
            # C-part: c_coeffs maps to C[:, k1:k2+1]
            # C11=C[0,k1], C12=C[0,k2], C21=C[1,k1], C22=C[1,k2]
            W[k1 * n + 0, r] = c_coeffs[0]  # C[0,k1]
            W[k2 * n + 0, r] = c_coeffs[1]  # C[0,k2]
            W[k1 * n + 1, r] = c_coeffs[2]  # C[1,k1]
            W[k2 * n + 1, r] = c_coeffs[3]  # C[1,k2]
            
            r += 1
        
        # A2 * B_bottom: A2 = A[:, 2:4], B_bottom = B[2:4, k1:k2+1]
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # A-part: a_coeffs maps to A[:, 2:4]
            U[0 * m + 2, r] = a_coeffs[0]  # A[0,2]
            U[0 * m + 3, r] = a_coeffs[1]  # A[0,3]
            U[1 * m + 2, r] = a_coeffs[2]  # A[1,2]
            U[1 * m + 3, r] = a_coeffs[3]  # A[1,3]
            
            # B-part: b_coeffs maps to B[2:4, k1:k2+1]
            V[2 * p + k1, r] = b_coeffs[0]  # B[2,k1]
            V[2 * p + k2, r] = b_coeffs[1]  # B[2,k2]
            V[3 * p + k1, r] = b_coeffs[2]  # B[3,k1]
            V[3 * p + k2, r] = b_coeffs[3]  # B[3,k2]
            
            # C-part: same as before
            W[k1 * n + 0, r] = c_coeffs[0]
            W[k2 * n + 0, r] = c_coeffs[1]
            W[k1 * n + 1, r] = c_coeffs[2]
            W[k2 * n + 1, r] = c_coeffs[3]
            
            r += 1
    
    # Handle k=6 separately with schoolbook
    # C[:, 6] = A[:, 0:2] * B[0:2, 6] + A[:, 2:4] * B[2:4, 6]
    # 8 scalar multiplications
    for j in range(4):
        for i in range(2):
            # A[i,j] * B[j,6] contributes to C[i,6]
            U[i * m + j, r] = 1.0
            V[j * p + 6, r] = 1.0
            W[6 * n + i, r] = 1.0
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    return U, V, W


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification of the tensor decomposition.
    
    Checks: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i, r] = delta_{j,j'} * delta_{k,k'}
    for all i, j, j', k, k'.
    """
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[jp * p + kp, r] * W[kp * n + i, r]
                        expected = 1.0 if (j == jp and k == kp) else 0.0
                        if abs(val - expected) > tol:
                            return False
    return True


def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, try the Strassen block decomposition (rank 50)
    U_str, V_str, W_str = _construct_strassen_block(n, m, p)
    
    # Verify with the full condition
    if _verify_full(U_str, V_str, W_str, n, m, p):
        best_U, best_V, best_W = U_str, V_str, W_str
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = _construct_schoolbook(n, m, p)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        pass
    
    # Convert to lists for the expected interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
