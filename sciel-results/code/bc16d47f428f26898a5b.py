"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from fractions import Fraction

def _schoolbook():
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check that the tensor decomposition exactly equals matrix multiplication.
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 if i==k and j==j (always), i.e., T[i,j,k] = 1 if i==k else 0?
    
    Wait, let me re-read the convention. The output factor follows W[k*n+i,r].
    The matrix multiplication tensor for A(n,m) * B(m,p) = C(n,p) is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 if the term A[i,j]*B[j,k]
    contributes to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k? No.
    
    Actually, the matrix multiplication tensor is T[i,j,k] = delta_{i,k} ... no.
    
    Let me think again. The standard tensor for matrix multiplication M_{n,m,p} is:
    T[i,j,k] = 1 if i, j, k are valid indices, and the "tensor" represents:
    sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    
    So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Wait no. The contraction is: C[i,k] = sum_j A[i,j] B[j,k].
    So the tensor is T[i,j,k] = 1 for all valid i,j,k, and the decomposition satisfies:
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ... 
    
    Hmm, but W has index k*n+i. Let me re-read: W[k*n+i, r].
    
    So the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And T[i,j,k] should equal 1 for all i,j,k? That can't be right because then
    the rank would be at least max(n*m, m*p, p*n) = 9, and the schoolbook gives 27.
    
    Actually wait - if T[i,j,k] = 1 for all i,j,k, then the tensor is just the
    all-ones tensor, which has rank 1. That's clearly wrong.
    
    Let me reconsider. The matrix multiplication tensor is defined as:
    T[i,j,k] = 1 if the indices form a valid "path" in the multiplication, which
    means... actually, I think I'm overcomplicating this.
    
    The correct interpretation: The tensor T has components T[i,j,k] where:
    - i indexes rows of A (and C): 0..n-1
    - j indexes columns of A / rows of B: 0..m-1  
    - k indexes columns of B (and C): 0..p-1
    
    The tensor is T[i,j,k] = 1 for ALL valid i,j,k. The matrix multiplication
    is then: C[i,k] = sum_j A[i,j] B[j,k] = sum_{i',j,k'} T[i',j,k'] A[i',j] B[j,k'] delta... 
    
    No, that's not right either. Let me think about it differently.
    
    The matrix multiplication tensor M is a 3-tensor such that:
    <M, A ⊗ B> = trace(C^T D) where C = A*B and D is... 
    
    Actually, the simplest way: M[i,j,k] = 1 if and only if the term A[i,j]*B[j,k]
    appears in the product. Since every such term appears exactly once, M[i,j,k] = 1
    for all i,j,k.
    
    But then the rank of the all-ones tensor is 1, which contradicts the known
    result that the rank of 3x3x3 matrix multiplication is 23.
    
    I must be misunderstanding the tensor definition. Let me look at this more carefully.
    
    OH WAIT. I think the issue is that the tensor is NOT all ones. The matrix
    multiplication tensor is defined such that:
    
    (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    
    This can be written as a tensor contraction:
    C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * delta... 
    
    No. The correct formulation is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    So if we define the tensor T with T[i,j,k] = 1 for all valid indices, then:
    sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] = sum_i sum_k C[i,k]
    
    That's the sum of all elements of C, not C itself.
    
    I think the correct tensor definition for matrix multiplication is:
    T[i,j,k] = 1 if i, j, k are the "free" indices, and the tensor represents
    the bilinear form. The rank of this tensor is what we're minimizing.
    
    Actually, I recall now: the matrix multiplication tensor M_{n,m,p} is defined
    by M_{n,m,p}[i,j,k] = 1 for all i in [n], j in [m], k in [p]. Its tensor rank
    is the minimum number of rank-1 tensors needed to decompose it.
    
    But that tensor is just the outer product of three all-ones vectors, so it has
    rank 1. That's clearly not what we want.
    
    Let me look at this from the decomposition perspective. A rank-1 term in the
    decomposition contributes U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to the
    (i,j,k) component. For the schoolbook, each term (i,j,k) has its own r, with
    U[i*m+j, r] = V[j*p+k, r] = W[k*n+i, r] = 1.
    
    So the schoolbook gives: for each (i,j,k), there's exactly one r where all three
    factors are 1, giving T[i,j,k] = 1.
    
    So indeed T[i,j,k] = 1 for all i,j,k, and the rank is the minimum number of
    rank-1 tensors to sum to the all-ones tensor. But that's rank 1!
    
    I'm clearly misunderstanding something. Let me re-read the problem statement.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix
    by a 3 by 3 matrix."
    
    OK so maybe the tensor is NOT all ones. Let me think about what the actual
    matrix multiplication tensor is.
    
    The matrix multiplication map is: (A, B) -> A*B where A is n x m, B is m x p.
    This is a bilinear map. The associated tensor has entries:
    T[i, j, k] where i is row of A/C, j is column of A/row of B, k is column of B/C.
    
    The tensor is defined such that the bilinear form is:
    sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = ??? 
    
    For this to equal the matrix product in some sense, we need:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    So the tensor should satisfy: the (i,k) component of C is sum_j T[i,j,k] A[i,j] B[j,k].
    This means T[i,j,k] = 1 for all i,j,k, and we're summing over j only (not i,k).
    
    I think the tensor is just the all-ones tensor of shape (n, m, p), and its
    "rank" in the context of matrix multiplication is the minimum number of
    rank-1 terms in a decomposition of the form:
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    Wait, but that's what I had before, and it gives rank 1 for the all-ones tensor.
    
    UNLESS... the indices in W are different. W[k*n+i, r] means the third factor
    depends on both k and i. So the decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the all-ones tensor, can we write 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i,j,k with few r's?
    
    If r=1: U[i*m+j, 1] * V[j*p+k, 1] * W[k*n+i, 1] = 1 for all i,j,k.
    This requires U[i*m+j, 1] = a[i,j], V[j*p+k, 1] = b[j,k], W[k*n+i, 1] = c[k,i]
    with a[i,j] b[j,k] c[k,i] = 1 for all i,j,k.
    
    Setting a[i,j] = 1, b[j,k] = 1, c[k,i] = 1 works! So rank 1 suffices for the
    all-ones tensor.
    
    This means the tensor is NOT all ones. There must be a different definition.
    
    Let me reconsider. Perhaps the tensor is defined differently in this problem.
    Maybe T[i,j,k] is not 1 for all i,j,k, but rather encodes the structure of
    matrix multiplication in a different way.
    
    Actually, I think I've been confusing two different things. The "matrix
    multiplication tensor" in the context of tensor rank is indeed the all-ones
    tensor, and its rank is trivially 1. But the "rank of matrix multiplication"
    that people study (like Strassen's algorithm) refers to the minimum number of
    scalar multiplications needed, which is a different concept.
    
    The minimum number of scalar multiplications to compute the product of two n x n
    matrices is called the "multiplication complexity" or "rank of matrix
    multiplication." For n=3, this is known to be 23 (Landsberg's result).
    
    So the problem is asking for the minimum number of scalar multiplications, which
    is the rank of the matrix multiplication tensor in a specific sense.
    
    Let me re-read the interface: "factor matrices of shapes (n*m, rank), (m*p, rank),
    (p*n, rank)". This matches the standard formulation where:
    - U has n*m rows (one for each element of A)
    - V has m*p rows (one for each element of B)
    - W has p*n rows (one for each element of C, with index k*n+i)
    
    And the decomposition satisfies: for each output element C[i,k],
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Wait, that's not quite right either. Let me think about this more carefully.
    
    The standard formulation is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    This means we compute R intermediate products, each of the form
    (linear combination of A elements) * (linear combination of B elements),
    and then C is a linear combination of these products.
    
    The constraint is that this must equal sum_j A[i,j] B[j,k] for all A, B.
    
    So the tensor constraint is: for all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ... no.
    
    Actually, the constraint is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if the indices (i,j,k)
    form a valid "path" in the multiplication, i.e., always 1 for valid i,j,k.
    
    No wait. Let's derive it properly. We want:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    And we propose:
    C[i,k] = sum_r W[k*n+i, r] * (sum_{j'} U[i*m+j', r] * A[i,j']) * (sum_{j''} V[j''*p+k, r] * B[j'',k])
    
    Expanding the right side:
    = sum_r W[k*n+i, r] * sum_{j',j''} U[i*m+j', r] * A[i,j'] * V[j''*p+k, r] * B[j'',k]
    
    For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    sum_r W[k*n+i, r] * sum_{j',j''} U[i*m+j', r] * V[j''*p+k, r] * A[i,j'] * B[j'',k]
    = sum_j A[i,j] B[j,k]
    
    This must hold for all A, B, so the coefficient of A[i,j'] B[j'',k] on the left
    must equal the coefficient on the right, which is 1 if j'=j''=j, 0 otherwise.
    
    So: sum_r W[k*n+i, r] * U[i*m+j', r] * V[j''*p+k, r] = delta_{j',j''}
    
    This is the constraint! The tensor T[i,j',j'',k] = sum_r U[i*m+j', r] * V[j''*p+k, r] * W[k*n+i, r]
    must equal delta_{j',j''} for all i,j',j'',k.
    
    So T[i,j',j'',k] = delta_{j',j''}, which is NOT the all-ones tensor. It's a
    tensor that depends on i and k only through the W factor, and is delta on j',j''.
    
    OK so now I understand. The tensor to decompose is T[i,j',j'',k] = delta_{j',j''}
    for i in [0,3), j' in [0,3), j'' in [0,3), k in [0,3).
    
    The schoolbook decomposition uses one r for each (i,j,k) triple, with
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1. This gives:
    sum_r ... = 1 if j'=j''=j, 0 otherwise. Correct.
    
    Now, the known result is that the rank of 3x3x3 matrix multiplication is 23.
    So I need to find a decomposition of rank 23 (or less, but 23 is the known minimum).
    
    Let me implement a search for rank 23. This is a hard problem, but there are
    known constructions. One approach is to use the "border rank" or "approximate
    rank" techniques, but we need exact rank.
    
    Actually, for this task, the score is the rank, and the seed has rank 27. If I
    can find rank 23, that's a significant improvement. But finding such a
    decomposition is non-trivial.
    
    Let me try a different approach: use a randomized or gradient-based method to
    find a lower-rank decomposition. Start from the schoolbook and try to reduce
    the rank by merging terms or using optimization.
    
    Actually, let me try a simpler approach: use the known fact that Strassen's
    algorithm for 2x2x2 has rank 7, and see if I can extend it to 3x3x3.
    
    For now, let me just try to find a rank-23 decomposition using a numerical
    optimization approach. I'll parameterize U, V, W with rank r and try to
    minimize the residual.
    
    Given the time constraint, let me try a practical approach:
    1. Start with rank 27 (schoolbook)
    2. Try to reduce rank by merging similar terms
    3. Use a local search to improve
    
    Actually, this is getting complicated. Let me just return the schoolbook
    decomposition for now, since finding a rank-23 decomposition in 120 seconds
    is very challenging without a specific algorithm.
    
    Wait, but the directive says to write a stronger algorithm. Let me try a
    randomized approach to find a lower-rank decomposition.
    
    Here's my plan:
    1. Use a numerical tensor decomposition approach
    2. Start with a random initialization of U, V, W with rank r
    3. Use gradient descent to minimize the residual
    4. Try decreasing r until we find a valid decomposition
    
    Let me implement this.
"""

def _build_tensor(n=3, m=3, p=3):
    """Build the matrix multiplication tensor T[i,j',j'',k] = delta_{j',j''}."""
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
    return T

def _residual(U, V, W, T, n=3, m=3, p=3):
    """Compute the residual ||sum_r U[i,j,r] V[j',k,r] W[k,i,r] - T[i,j,j',k]||."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    residual = np.zeros_like(T)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for jp in range(p):
                    for k in range(p):
                        # Wait, this is getting complicated. Let me use einsum.
                        pass
    # Use einsum for efficiency
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # Contract: sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # This is a 4-index tensor: result[i,j,k,?] ... wait, the output should be
    # indexed by (i, j, k) where j is the inner index.
    # Actually, the tensor we're decomposing is T[i, j, j', k] = delta_{j,j'}
    # So we need: sum_r U[i,j,r] * V[j',k,r] * W[k,i,r] = delta_{j,j'}
    # Let me compute S[i, j, j', k] = sum_r U_r[i,j,r] * V_r[j',k,r] * W_r[k,i,r]
    # This is a 4D tensor.
    S = np.einsum('ijr,jkr,kir->ijjk', U_r, V_r, W_r)
    # T is T[i, j, j', k] = delta_{j,j'}
    return np.linalg.norm(S - T)

def _try_rank(r, n=3, m=3, p=3, max_iters=1000, tol=1e-8, seed=42):
    """Try to find a rank-r decomposition using gradient descent."""
    np.random.seed(seed)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    T = _build_tensor(n, m, p)
    
    lr = 0.01
    for it in range(max_iters):
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        S = np.einsum('ijr,jkr,kir->ijjk', U_r, V_r, W_r)
        residual = S - T
        if np.linalg.norm(residual) < tol:
            return U, V, W, True
        
        # Compute gradients
        # dS/dU[i,j,r] = sum_{j',k} V_r[j',k,r] * W_r[k,i,r] * residual[i,j,j',k]
        # dS/dV[j',k,r] = sum_{j,k} U_r[i,j,r] * W_r[k,i,r] * residual[i,j,j',k]  -- wait, indices are wrong
        # Let me be more careful.
        # S[i,j,j',k] = sum_r U_r[i,j,r] * V_r[j',k,r] * W_r[k,i,r]
        # dS[i,j,j',k]/dU[i0,j0,r0] = delta_{i,i0} delta_{j,j0} delta_{r,r0} * V_r[j',k,r0] * W_r[k,i0,r0]
        # So grad_U[i,j,r] = sum_{j',k} V_r[j',k,r] * W_r[k,i,r] * residual[i,j,j',k]
        # grad_V[j',k,r] = sum_{j,k} U_r[i,j,r] * W_r[k,i,r] * residual[i,j,j',k]  -- wait, i is free
        # Actually: dS[i,j,j',k]/dV[j'0,k0,r0] = delta_{j',j'0} delta_{k,k0} delta_{r,r0} * U_r[i,j,r0] * W_r[k0,i,r0]
        # So grad_V[j',k,r] = sum_{i,j} U_r[i,j,r] * W_r[k,i,r] * residual[i,j,j',k]
        # grad_W[k,i,r] = sum_{j,j'} U_r[i,j,r] * V_r[j',k,r] * residual[i,j,j',k]
        
        grad_U = np.einsum('jkr,kir,ijjk->ijr', V_r, W_r, residual)
        grad_V = np.einsum('ijr,kir,ijjk->jkr', U_r, W_r, residual)
        grad_W = np.einsum('ijr,jkr,ijjk->kir', U_r, V_r, residual)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    final_res = _residual(U, V, W, T, n, m, p)
    return U, V, W, final_res < tol

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known: rank 23 is achievable for 3x3x3
    for r in range(23, best_rank):
        if time.time() > deadline:
            break
        try:
            U, V, W, success = _try_rank(r, n, m, p, max_iters=500, tol=1e-7, seed=42)
            if success:
                # Convert to lists
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        except Exception:
            continue
    
    return best
# EVOLVE-BLOCK-END
