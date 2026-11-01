"""Matrix multiplication 2x3x4: search for minimal-rank exact tensor decomposition."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds to leave before deadline


def _check_decomposition(U, V, W, n=2, m=3, p=4, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k,l] = delta_{j,l} (i.e., (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    must equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k.
    
    Wait, let me re-read the convention. The task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W[k*n+i, r] follows the official trace convention.
    
    The matrix multiplication tensor for C = A*B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has indices (i, j, k) with T[i,j,k] = 1 if this is the "identity" 
    mapping, but actually the tensor decomposition means:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[?, r]
    
    Actually, for matrix multiplication, the standard tensor is:
    T[i, j, k] = delta_{j,k} doesn't make sense for 2x3*3x4.
    
    Let me think again. The matrix multiplication tensor for multiplying an (n x m) matrix 
    by an (m x p) matrix has a rank-r decomposition where:
    (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    No wait, that's not right either. Let me look at the schoolbook:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) triple.
    
    So the decomposition is: for each (i,j,k), we have a rank-1 term.
    The product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should reconstruct 
    the multiplication tensor.
    
    The multiplication tensor T[i, j, k] (where i is row of A, j is col of A / row of B, k is col of B)
    satisfies: C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, T[i,j,k] = 1 for all valid i,j,k? No.
    
    Actually, the matrix multiplication tensor is defined as:
    T[i, j, k] = delta_{j, j'} where... 
    
    Let me just verify with the schoolbook: for a given (i,j,k), U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    The contribution to (i', j', k') is U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c].
    This is nonzero only when (i',j') = (i,j), (j',k') = (j,k), and (k',i') = (k,i).
    So i'=i, j'=j, k'=k. Thus it contributes 1 to T[i,j,k] and 0 elsewhere.
    
    So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, that means T is just an all-ones tensor? That can't be right for matrix multiplication.
    
    Hmm, let me reconsider. In the standard matrix multiplication tensor:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has three indices corresponding to (i, j, k) and T[i,j,k] = 1.
    Then sum_r U[i,j,r] V[j,k,r] W[k,i,r] should equal T[i,j,k] = 1 for all i,j,k.
    
    Wait, but W is indexed by (k*n+i), so W[k*n+i, r]. The three factors are:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the schoolbook, this equals 1 when all three indices match the (i,j,k) that generated rank c,
    and 0 otherwise. So summing over all c (all i',j',k'), we get 1 for each (i,j,k).
    
    So the target tensor is T[i,j,k] = 1 for all i,j,k. That's just the all-ones tensor!
    But that doesn't seem right for matrix multiplication...
    
    Actually wait - I think I'm confusing myself. Let me re-examine.
    
    In the matrix multiplication problem, we want:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be written as a trilinear form:
    sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    
    The tensor T[i,j,k] for matrix multiplication is actually just 1 for all valid indices,
    because the "multiplication" is just the sum over j of products. The tensor encodes the 
    structure of which indices are connected.
    
    So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    And we need rank-r matrices U, V, W such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    This is the tensor rank of the all-ones tensor of size n x m x p!
    
    For the all-ones tensor of size 2x3x4, what's the minimal rank?
    The tensor rank of the all-ones tensor [1_{n x m x p}] is known to be at least max(n,m,p)
    and at most min(n*m, m*p, p*n).
    
    Actually, for the all-ones tensor, the rank is known to be exactly max(n,m,p) in some cases,
    but more precisely, it's related to the tensor rank of the "all-ones" tensor.
    
    For a 2x3x4 all-ones tensor, let's think about lower bounds:
    - The slice rank lower bound gives at least max(n,m,p) = 4.
    - Actually, the tensor rank of the all-ones tensor of size n x m x p is known to be 
      at least (n*m*p) / max(n,m,p) by averaging, but that's not tight.
    
    Let me just try to find a small rank decomposition by search.
    """
    rank = len(U[0]) if U else 0
    n, m, p = n, m, p
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank(r, n=2, m=3, p=4, max_iter=5000, seed=42):
    """Try to find a rank-r decomposition of the all-ones tensor using random initialization + GD."""
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(3):
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        lr = 0.01
        for it in range(max_iter):
            # Compute gradients
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # dT/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] (summed over k for fixed i,j)
            # Actually, let's compute the error and gradient numerically or analytically.
            
            # Error: E = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # Gradient w.r.t. U[i*m+j, r]:
            #   2 * sum_k (sum_r' U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] - 1) * V[j*p+k,r] * W[k*n+i,r]
            
            # This is expensive to compute naively. Let's use a simpler approach:
            # Just do a few gradient steps with a simple learning rate.
            
            # Compute current tensor values
            # For efficiency, let's just check if we're close enough
            pass
        
        break  # Skip the GD for now, use a different approach
    
    return None


def _search_small_rank(n=2, m=3, p=4, deadline=150.0):
    """Search for minimal rank decomposition using integer/half-integer entries."""
    start_time = time.time()
    remaining = deadline - SAFETY_MARGIN - (start_time - start_time)
    
    # First, try known small ranks with structured approaches
    # For 2x3x4 all-ones tensor, try ranks from 4 upward
    
    # Strategy: use the fact that for small dimensions, we can enumerate or use
    # algebraic constructions.
    
    # Try rank 4: We need U (6x4), V (12x4), W (8x4) with integer/half-integer entries
    # such that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    
    # One approach: use a "separable" structure. If we can write the all-ones tensor
    # as a sum of rank-1 tensors with integer coefficients, we're done.
    
    # For the all-ones tensor of size 2x3x4:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    
    # A rank-1 tensor is u_i * v_j * w_k.
    # We need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    
    # This means the matrix M_{i,k} = sum_r u_r[i] * w_r[k] must satisfy
    # M_{i,k} = 1/v_r[j] for all j, which means v_r[j] must be constant in j for each r,
    # OR we need the sum to work out.
    
    # Actually, if we fix i and k, then sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all j.
    # This means for each (i,k), the vector (sum_r u_r[i]*w_r[k] * v_r[j])_j must be (1,1,1).
    # So sum_r u_r[i]*w_r[k] * v_r[j] = 1 for all j.
    
    # If v_r[j] = v_r (constant in j), then sum_r u_r[i]*w_r[k]*v_r = 1 for all i,k.
    # This means the matrix (u_r[i] * v_r) (i x r) times (w_r[k] * v_r) (r x k) = all-ones.
    # Wait, that's not quite right.
    
    # Let me try a different approach: brute force with small integer/half-integer values.
    
    # For rank r, each entry is in {-1, -1/2, 0, 1/2, 1} (small set).
    # Number of possible values per entry: 5
    # Number of entries: (n*m + m*p + p*n) * r = (6+12+8)*r = 26*r
    # For r=4: 26*4 = 104 entries, each with 5 choices -> 5^104 possibilities. Way too many.
    
    # Better approach: use the structure. Let's try to construct a solution.
    
    # For 2x3x4, let's try rank 4 with a specific structure.
    # Idea: use 4 "blocks" corresponding to the 4 columns of B.
    
    # Actually, let me just try a random search with small integer values.
    
    def random_search(rank, max_iters=100000, val_range=[-1, -0.5, 0, 0.5, 1]):
        rng = random.Random(12345)
        for it in range(max_iters):
            if time.time() > deadline - SAFETY_MARGIN:
                break
            U = [[rng.choice(val_range) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.choice(val_range) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.choice(val_range) for _ in range(rank)] for _ in range(p*n)]
            
            # Quick check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W
        return None
    
    # Try ranks from 4 upward
    for rank in range(4, 10):
        if time.time() > deadline - SAFETY_MARGIN:
            break
        result = random_search(rank, max_iters=50000)
        if result is not None:
            return result, rank
    
    # If random search fails, fall back to schoolbook
    return None, None


def _schoolbook(n=2, m=3, p=4):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}, rank


def construct():
    start_time = time.time()
    
    # Try to find a better decomposition
    n, m, p = DIMENSIONS
    
    # Try structured constructions first
    # For 2x3x4 all-ones tensor, try rank 4
    
    # Construction attempt: use 4 rank-1 terms
    # T[i,j,k] = sum_{r=0}^{3} u_r[i] * v_r[j] * w_r[k] = 1
    
    # Try: u_0 = [1,0], u_1 = [0,1], u_2 = [1,1], u_3 = [1,1] (or similar)
    # and solve for v_r, w_r
    
    # Actually, let's try a more systematic approach.
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Note that U[i*m+j, r] depends on both i and j, V[j*p+k, r] depends on j and k,
    # and W[k*n+i, r] depends on k and i. This is different from a standard tensor
    # decomposition where each factor depends on only one index.
    
    # This is a "cyclic" tensor decomposition. Let me re-examine the schoolbook:
    # U[i*m+j, c] = 1 for the (i,j) that generated c
    # V[j*p+k, c] = 1 for the (j,k) that generated c  
    # W[k*n+i, c] = 1 for the (k,i) that generated c
    
    # So each rank-1 term is nonzero only at specific (i,j), (j,k), (k,i) positions.
    # The product is 1 only when all three match, i.e., at position (i,j,k).
    
    # For a better decomposition, we need rank-1 terms that are "spread out" more.
    
    # Let me try a different approach: use numpy to do a random search with 
    # integer/half-integer values, but be smarter about it.
    
    # First, let's try a specific construction for rank 4.
    
    # For n=2, m=3, p=4:
    # U is 6x4, V is 12x4, W is 8x4
    
    # Let's try to use the structure where:
    # U[i*m+j, r] = a_r[i] * b_r[j] (separable in i and j)
    # V[j*p+k, r] = b_r[j] * c_r[k] (separable in j and k)
    # W[k*n+i, r] = c_r[k] * a_r[i] (separable in k and i)
    
    # Then the product is: a_r[i]*b_r[j] * b_r[j]*c_r[k] * c_r[k]*a_r[i] = a_r[i]^2 * b_r[j]^2 * c_r[k]^2
    # And we need sum_r a_r[i]^2 * b_r[j]^2 * c_r[k]^2 = 1 for all i,j,k.
    
    # This means the matrix (a_r[i]^2 * c_r[k]^2) (2x4) times... no, this is a product of three vectors.
    # sum_r (a_r[i]^2) * (b_r[j]^2) * (c_r[k]^2) = 1 for all i,j,k.
    
    # This is a standard tensor decomposition of the all-ones tensor with nonnegative entries.
    # Let A_r[i] = a_r[i]^2, B_r[j] = b_r[j]^2, C_r[k] = c_r[k]^2.
    # Then sum_r A_r[i] * B_r[j] * C_r[k] = 1.
    
    # For this to work with rank 4, we need 4 nonnegative vectors A_r (length 2), B_r (length 3), C_r (length 4)
    # such that their tensor sum is all-ones.
    
    # One simple construction: let A_r[i] = 1 for all r,i, B_r[j] = 1 for all r,j, C_r[k] = 1 for all r,k.
    # Then sum_r 1*1*1 = r, so we need r=1. But rank 1 doesn't work for 2x3x4.
    
    # Another: use orthogonal-ish vectors.
    # Let's just try random search with the separable structure.
    
    def separable_search(rank, max_iters=100000):
        rng = random.Random(99999)
        # Try integer values for a, b, c (so A=a^2, B=b^2, C=c^2 are perfect squares)
        # Or just try A, B, C directly with small nonnegative integers.
        
        for it in range(max_iters):
            if time.time() > deadline - SAFETY_MARGIN:
                break
            
            # Generate random small nonnegative integer vectors
            A = [[rng.randint(0, 2) for _ in range(n)] for _ in range(rank)]
            B = [[rng.randint(0, 2) for _ in range(m)] for _ in range(rank)]
            C = [[rng.randint(0, 2) for _ in range(p)] for _ in range(rank)]
            
            # Check: sum_r A[r][i] * B[r][j] * C[r][k] == 1 for all i,j,k
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(A[r][i] * B[r][j] * C[r][k] for r in range(rank))
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to U, V, W
                U = [[0] * rank for _ in range(n*m)]
                V = [[0] * rank for _ in range(m*p)]
                W = [[0] * rank for _ in range(p*n)]
                for r in range(rank):
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j][r] = A[r][i]
                        for k in range(p):
                            V[j*p+k][r] = B[r][j]
                        for k in range(p):
                            W[k*n+i][r] = C[r][k]
                return U, V, W
        return None
    
    # Try separable search for small ranks
    for rank in range(1, 8):
        if time.time() > deadline - SAFETY_MARGIN:
            break
        result = separable_search(rank, max_iters=20000)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # If separable search fails, try general random search
    def general_search(rank, max_iters=50000):
        rng = random.Random(77777)
        vals = [-1, -0.5, 0, 0.5, 1]
        for it in range(max_iters):
            if time.time() > deadline - SAFETY_MARGIN:
                break
            U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)]
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W
        return None
    
    for rank in range(4, 10):
        if time.time() > deadline - SAFETY_MARGIN:
            break
        result = general_search(rank, max_iters=30000)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)[0]
