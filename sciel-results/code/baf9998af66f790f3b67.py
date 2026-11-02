"""Matrix multiplication 3x3x3 rank-minimal tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]
n, m, p = DIMENSIONS
RANK_MAX = n * m * p  # 27 (schoolbook)

# Known best known rank for 3x3x3 is 23 (Strassen-like)
# We will attempt to find a valid rank-23 decomposition.

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 3x3 matrix multiplication.
    
    The decomposition satisfies:
        C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i,j,k, where the tensor is T[i,j,k] = delta(i,k) (i.e., C[i,k] = sum_j A[i,j]*B[j,k]).
    
    Actually the matrix multiplication tensor is:
        T[i,j,k] = 1 if i==k else 0  ... no wait.
    
    The matrix multiplication tensor M[i,j,k] is defined such that:
        (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    As a 3-tensor: M[i,j,k] = delta(i,k)? No.
    
    Let me be precise. The tensor for matrix multiplication has entries:
        M[i,j,k] = 1 if i == k, else 0? No that's not right either.
    
    Actually, the multiplication tensor is:
        M[i,j,k] = δ_{i,k}  ... no.
    
    The correct tensor: for multiplying A (n×m) by B (m×p) to get C (n×p):
        C[i,k] = Σ_j A[i,j] B[j,k]
    
    The associated tensor T has entries T[i,j,k] = δ_{i,k}? No.
    
    T[i,j,k] should be such that the bilinear form gives C[i,k].
    The tensor is: T[i,j,k] = 1 if we're "using" A[i,j] and B[j,k] to contribute to C[i,k].
    
    So T[i,j,k] = δ_{i,k} is wrong. The tensor is actually:
    T[i,j,k] = 1 for all valid (i,j,k) where the product A[i,j]*B[j,k] contributes to C[i,k].
    
    Wait, I think the standard form is: the tensor T has T[i,j,k] = 1 if the entry (i,j) of A and (j,k) of B multiply to give contribution to (i,k) of C. So T[i,j,k] = δ_{i,k}? No, that doesn't make sense dimensionally.
    
    Let me reconsider. The matrix multiplication tensor M_{ijk} is defined by:
        M_{ijk} = 1 if i=k (in the sense that output index i and output index k are the same? No...)
    
    OK let me just use the seed program's convention. The seed uses:
        U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    and for the schoolbook, each term has U=1, V=1, W=1.
    
    The tensor equation is:
        T[i,j,k] = Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T[i,j,k] = 1 if i==k? Let's check: in schoolbook, for each (i,j,k) triple, there's exactly one term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1, and all other terms have at least one zero. So T[i,j,k] = 1 for ALL i,j,k.
    
    Wait that can't be right for matrix multiplication. Let me re-examine.
    
    In matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
        C[i,k] = Σ_j A[i,j] * B[j,k]
    
    The tensor should encode: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1 if the indices match properly.
    
    Actually, the tensor T[i,j,k] represents: "A[i,j] times B[j,k] contributes to C[i,k] with coefficient T[i,j,k]". But C has indices (i,k), and the term A[i,j]*B[j,k] contributes to C[i,k]. So for a given output C[i,k], the sum is over j. The tensor entry T[i,j,k] = 1 means "the product A[i,j]*B[j,k] contributes to C[i,k]".
    
    But C[i,k] gets contributions from all j: C[i,k] = Σ_j (A[i,j]*B[j,k]). So T[i,j,k] = 1 for ALL i,j,k? That would make the tensor all-ones, which is rank 1, which is clearly wrong.
    
    I think I'm confusing myself. Let me look at it differently.
    
    The matrix multiplication tensor is a 3-tensor T ∈ R^{n×m × m×p × n×p} (or similar). The standard way:
    
    T is indexed by (i, j, k) where i∈{1..n}, j∈{1..m}, k∈{1..p}, and:
        T[i,j,k] = 1 if this term A[i,j]·B[j,k] contributes to C[i,k].
    
    But every term A[i,j]·B[j,k] DOES contribute to C[i,k]. So T[i,j,k] = 1 for all i,j,k.
    
    Hmm, but then the tensor is all-ones and has rank 1. That's clearly not the matrix multiplication tensor.
    
    I think the issue is that the tensor should be over the OUTPUT index as well. Let me re-read the problem.
    
    "U factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    "W[k*n+i,r]"
    
    So U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    
    The decomposition is:
        T[i,j,k] = Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And T[i,j,k] is the matrix multiplication tensor. For 3x3x3, T[i,j,k] = 1 if i==k? No...
    
    Actually, I believe the matrix multiplication tensor for n×m × m×p → n×p is:
        T[i,j,k] = δ_{i,k}  ... no.
    
    Let me just verify with the schoolbook. In schoolbook, rank = n*m*p = 27. Each basis element (i,j,k) has its own term. The tensor value at (i,j,k) is 1 for all i,j,k. So T is the all-ones tensor? That has rank 1, not 27.
    
    I must be misunderstanding the tensor definition. Let me think again.
    
    OH WAIT. I think the tensor is NOT all ones. The matrix multiplication tensor has:
        T[i,j,k] = 1 if i == k (output row matches... no)
    
    Actually, I think the correct definition is:
    
    The tensor M for matrix multiplication is in R^{n×p × n×m × m×p} or similar, where:
        M[c, a, b] = 1 if c = a's row and b's column, i.e., if we're computing C[c_row, c_col] = Σ A[a_row, a_col] * B[b_row, b_col] where a_row=c_row, b_col=c_col, and a_col=b_row.
    
    So in the indexing used here:
        U is indexed by (i, j) where i is row of A, j is column of A (also row of B)
        V is indexed by (j, k) where j is row of B, k is column of B (also column of C)
        W is indexed by (k, i) where k is column of C, i is row of C
    
    And the tensor T[i,j,k] = 1 if the triple (i,j,k) is a valid "path" through the multiplication, which is ALWAYS valid (any i,j,k forms a valid term A[i,j]*B[j,k] contributing to C[i,k]).
    
    So T[i,j,k] = 1 for ALL i,j,k. The all-ones tensor of size 3×3×3.
    
    But the all-ones tensor has rank 1! You can write it as U=ones(9,1), V=ones(9,1), W=ones(9,1).
    
    This is clearly wrong. So I must be misunderstanding the tensor.
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    And the schoolbook has rank 27 with each (i,j,k) being a separate term. If T were all-ones, rank would be 1.
    
    I think the tensor must be something else. Perhaps:
        T[i,j,k] = 1 if i == k, else 0?
    
    Let's check: if T[i,j,k] = δ_{i,k}, then T is a tensor where T[i,j,i] = 1 and T[i,j,k] = 0 for i≠k. This is like the "identity" tensor in some sense.
    
    For n=m=p=3, this tensor has 9 non-zero entries (one for each (i,j) with i=k... wait, i=k means k is determined by i, so for each i,j there's one k=i, giving 9 entries).
    
    Can we decompose this with rank < 9? The tensor T[i,j,k] = δ_{i,k} can be written as:
        T[i,j,k] = Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    With rank 9: for each (i,j), set r=(i,j), U[i*m+j, r]=1, V[j*p+i, r]=1 (since k=i), W[i*n+i, r]=1.
    
    But can we do better? This is essentially the "slice" problem.
    
    Hmm, but the known result is that 3x3x3 matrix multiplication has rank 23 (or is it 20? I recall it's between 19 and 23, with 23 being the best known upper bound).
    
    Wait, I think I had it backwards. The tensor IS the all-ones tensor in the sense that every (i,j,k) triple is a valid term, but the tensor values are NOT all 1. Let me reconsider.
    
    Actually, I think the confusion is that the "tensor" here is the structure tensor of the bilinear map, and for matrix multiplication, the structure tensor has entries that are 1 for each (i,j,k) because each product A[i,j]*B[j,k] appears exactly once in the sum for C[i,k].
    
    But if all entries are 1, the rank is 1. This contradiction means I'm wrong about the tensor entries.
    
    Let me look at this from a different angle. The bilinear map is:
        (A, B) ↦ C = A*B
    
    The structure tensor T is in R^{(n×m) × (m×p) × (n×p)} where:
        T[a, b, c] = coefficient of (e_a ⊗ e_b) in the expansion of the bilinear map evaluated at e_c.
    
    More precisely: T[a,b,c] = 1 if the product of the a-th basis element of the first space and the b-th basis element of the second space contributes to the c-th basis element of the output.
    
    For matrix multiplication:
        a = (i,j) where i=row, j=col of A
        b = (j',k) where j'=row, k=col of B
        c = (i',k') where i'=row, k'=col of C
        
        T[a,b,c] = 1 if i=i', j'=j, k=k'. I.e., T[(i,j),(j,k),(i,k)] = 1.
    
    So in the indexing of the problem:
        U is indexed by a = i*m + j
        V is indexed by b = j*p + k
        W is indexed by c = k*n + i  (note: "official trace convention W[k*n+i,r]")
        
        T[a,b,c] = 1 if the indices are consistent: the j in a equals the j in b, the i in a equals the i in c, and the k in b equals the k in c.
    
    So T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    
    In 3D array form with indices (i,j,k): T[i,j,k] = 1 (since for any i,j,k, the triple (i,j,k) corresponds to a valid term). Wait, that's back to all-ones.
    
    NO. The tensor is 3D with dimensions (n*m) × (m*p) × (n*p) = 9×9×9 for 3×3×3. It's NOT a 3×3×3 tensor. It's a 9×9×9 tensor with specific non-zero entries.
    
    T[a,b,c] where a∈{0..8}, b∈{0..8}, c∈{0..8}.
    T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k. Otherwise 0.
    
    So there are exactly 27 non-zero entries (one for each (i,j,k) triple), and each is 1.
    
    The rank of this 9×9×9 tensor is what we're minimizing. The schoolbook gives rank 27 (one term per non-zero entry). The question is whether we can do better.
    
    This makes much more sense! The tensor is sparse (27 non-zeros out of 729), and we're looking for a low-rank decomposition.
    
    Now, the known best rank for 3×3 matrix multiplication is 23 (I believe). Let me try to construct a rank-23 decomposition.
    
    Actually, I recall that the best known upper bound for 3×3 matrix multiplication rank is 23, achieved by various constructions. Let me try to find one.
    
    One approach: use the Strassen algorithm generalized to 3×3. Strassen does 2×2 with rank 7. For 3×3, there are known rank-23 decompositions.
    
    Let me try to use a known rank-23 decomposition. I'll look for one in the literature or derive one.
    
    Actually, let me just try to use numerical optimization to find a low-rank decomposition. I'll start from the schoolbook (rank 27) and try to reduce the rank by setting some entries to zero and re-solving.
    
    Given the time constraint, let me try a simpler approach: use the known fact that 3×3 matrix multiplication can be done with 23 multiplications, and try to hardcode a known decomposition.
    
    Let me try to find a rank-23 decomposition by starting from rank 27 and eliminating 4 terms.
    
    Actually, given the complexity, let me just return the schoolbook decomposition (rank 27) but with a small optimization attempt. If I can't find a better one quickly, I'll return rank 27.
    
    Wait, the seed already returns rank 27 with score 1.0. To improve, I need rank < 27.
    
    Let me try to use a numerical approach: start with a random rank-23 decomposition and use gradient descent to minimize the error. If I can get the error to zero (or very close), I have a valid rank-23 decomposition.
    
    But the entries need to be "integer/half-integer". So I'll need to round at the end and verify.
    
    Given the 150s time limit, let me try a few random restarts with gradient descent for rank 23, 24, 25, 26.
    
    Actually, this is getting complex. Let me just return the schoolbook for now, but with a small improvement: check if there's a simple rank reduction.
    
    Hmm, let me just implement a simple search: try rank 23 with random initialization and a few iterations of a simple optimization, and if it works, return it. Otherwise, fall back to rank 27.
    
    Given the time constraints and the difficulty of finding an exact decomposition numerically, let me just return the schoolbook. But that gives score 1.0, same as seed.
    
    Wait, the task says "score is the rank" and "lower is better". So rank 27 gives score 27? Or is the score normalized?
    
    Re-reading: "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement."
    
    So if seed has rank 27, and I find rank 23, the score would be 27/23 ≈ 1.174. That's an improvement.
    
    So I need to find a rank < 27 decomposition.
    
    Let me try to hardcode a known rank-23 decomposition for 3×3 matrix multiplication.
    
    I recall that there's a rank-23 decomposition based on the work of Strassen and others. Let me try to construct one.
    
    Actually, let me just try a numerical approach with a time limit. I'll use numpy and do a simple optimization.
    
    Given the complexity and time constraints, let me just return the schoolbook. The score will be 1.0, same as seed. But the directive says to improve.
    
    Let me try one more thing: use the known rank-23 decomposition from the literature. I'll try to find it online... but I can't. Let me just try to derive one.
    
    For 3×3 matrix multiplication, one approach is to use the fact that:
        [a b c]   [e f g]   [ae+bf+cg, af+bg+ch, ag+bh+ci]
        [d e f] * [h i j] = [de+ef+fg, df+eg+fh, dg+eh+fi]
        [g h i]   [k l m]   [ge+hf+ig, gf+ig+jh, gg+ih+ji]
    
    Wait, that's not right. Let me redo:
        A = [[a,b,c],[d,e,f],[g,h,i]]
        B = [[e,f,g],[h,i,j],[k,l,m]]  -- no, let me use different letters.
    
    A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    C = A*B where C[i,k] = Σ_j A[i,j]*B[j,k]
    
    C[0,0] = a11*b11 + a12*b21 + a13*b31
    C[0,1] = a11*b12 + a12*b22 + a13*b32
    C[0,2] = a11*b13 + a12*b23 + a13*b33
    C[1,0] = a21*b11 + a22*b21 + a23*b31
    C[1,1] = a21*b12 + a22*b22 + a23*b32
    C[1,2] = a21*b13 + a22*b23 + a23*b33
    C[2,0] = a31*b11 + a32*b21 + a33*b31
    C[2,1] = a31*b12 + a32*b22 + a33*b32
    C[2,2] = a31*b13 + a32*b23 + a33*b33
    
    That's 9 outputs, each requiring 3 products, totaling 27 products. Strassen's insight for 2×2 was to reduce this to 7 products by clever combinations.
    
    For 3×3, the best known is 23 products. Let me try to find a specific decomposition.
    
    Given the time constraints, let me just implement a simple numerical search and hope to find a valid decomposition. If not, I'll fall back to rank 27.
    
    Actually, let me just return the schoolbook. The score is 1.0, which is the same as the seed. The directive says to improve, but if I can't find a better decomposition quickly, I should at least return a valid one.
    
    Let me try a quick numerical approach: initialize a rank-23 decomposition randomly and do a few iterations of a simple optimization. If the residual is small enough, round to half-integers and verify.
    
    Given the 150s limit, I'll allocate most of the time to this search.
    
    Here's my plan:
    1. Try to find a rank-23 decomposition using numerical optimization.
    2. If successful (residual < 1e-10 after rounding), return it.
    3. Otherwise, fall back to rank 27.
    
    Let me implement this.
    
    Actually, numerical optimization for tensor decomposition is NP-hard in general and very challenging. The landscape is full of local minima. I'm not confident this will work within the time limit.
    
    Let me just return the schoolbook. Score 1.0.
    
    Wait, but the directive explicitly says to improve. Let me at least TRY the numerical approach with a time limit.
    
    OK here's what I'll do: implement a simple gradient descent for rank-23 decomposition with a few random restarts, each limited to a few seconds. If none succeed, return rank 27.
    
    Let me code this up.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 3]
n, m, p = DIMENSIONS

def _build_tensor():
    """Build the 9x9x9 matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(U, V, W, T, tol=1e-9):
    """Verify that U,V,W decompose T."""
    rank = U.shape[1]
    # Compute reconstructed tensor
    recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.max(np.abs(recon - T)) < tol

def _try_rank(R, T, time_limit=10.0, n_restarts=5, lr=0.01):
    """Try to find a rank-R decomposition using gradient descent."""
    start = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(n_restarts):
        if time.time() - start > time_limit:
            break
        # Random initialization
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        for it in range(500):
            if time.time() - start > time_limit:
                break
            # Compute residual
            recon = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            residual = recon - T
            err = np.sum(residual**2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-20:
                break
            
            # Compute gradients (simplified)
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(R):
                dU[:,r] = residual * np.einsum('bc,b->c', V[:,r], W[:,r])  # This is wrong, let me fix
                # Actually: d/dU[a,r] = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                dU[:,r] = np.einsum('abc,b,c->a', residual, V[:,r], W[:,r])
                dV[:,r] = np.einsum('abc,a,c->b', residual, U[:,r], W[:,r])
                dW[:,r] = np.einsum('abc,a,b->c', residual, U[:,r], V[:,r])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
    
    return best_err, best_U, best_V, best_W

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(2*x) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    T = _build_tensor()
    
    # Try ranks from 23 upward
    for R in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
        time_limit = min(20.0, deadline - time.time() - 5)
        if time_limit <= 2:
            break
        err, U, V, W = _try_rank(R, T, time_limit=time_limit, n_restarts=3)
        
        if U is not None and err < 1e-10:
            # Round to half-integers and verify
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            if _verify_decomp(U_r, V_r, W_r, T, tol=1e-9):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: schoolbook rank 27
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
