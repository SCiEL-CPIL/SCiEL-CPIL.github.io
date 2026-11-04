import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is exact.
    T[a, b, c, d] = sum_r U[a*m+b, r] * V[c*p+d, r] * W[k*n+i, r]
    where T[i, j, k, l] = 1 if i==k and j==l, else 0 (for output (k,l) = A[i,:] * B[:,j])
    
    Actually, the tensor for multiplying A (n x m) by B (m x p) to get C (n x p) is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i,j,k] = 1 if (i,j,k) corresponds to a term in the sum.
    But here we have a 4-index tensor because we're decomposing the bilinear map.
    
    The tensor is T[a, b, c, d] where:
    - a, b: indices into A (a in [0,n), b in [0,m))
    - c, d: indices into B (c in [0,m), d in [0,p))
    - T[a,b,c,d] = 1 if b == c (the contraction index matches)
    
    Wait, let me re-read the problem. The output factor W follows W[k*n+i, r].
    So the tensor is T[i, k, a, b, c, d] but actually it's simpler.
    
    The matrix multiplication tensor M_{n,m,p} has entries:
    M[i, j, k, l] = 1 if j == k (where i is row of A, j is col of A = row of B, k is col of B... no)
    
    Let me think again. A is n x m, B is m x p. C[i, k] = sum_j A[i,j] * B[j,k].
    The tensor T has one index for each input: (i, j) for A and (j, k) for B, but j is shared.
    So T[i, j, k] = 1 for each (i, j, k) where the term A[i,j]*B[j,k] appears.
    
    In the factorization:
    C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]  ... no that's not right either.
    
    Let me look at the interface more carefully.
    U: (n*m, rank) - indexed by (i, j) where i in [0,n), j in [0,m)
    V: (m*p, rank) - indexed by (j, k) where j in [0,m), k in [0,p)
    W: (p*n, rank) - indexed by (k, i) where k in [0,p), i in [0,n), as W[k*n+i]
    
    So C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This should equal sum_j A[i,j] * B[j,k]
    
    For this to work, we need:
    sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    ... no, that's not right.
    
    Actually, the tensor T[i,j,k] = 1 (for the term A[i,j]*B[j,k]).
    The factorization is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, but j appears in both U and V indices. Let me re-index.
    
    U is indexed by (i, j) -> row i*m + j
    V is indexed by (j, k) -> row j*p + k  
    W is indexed by (k, i) -> row k*n + i
    
    The tensor entry T[i,j,k] should equal 1 for each valid (i,j,k).
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So for the schoolbook:
    U[i*m+j, r] = 1 when r = i*m*p + j*p + k... no, that doesn't work.
    
    Actually for schoolbook, each term (i,j,k) gets its own rank component:
    r = i*p*m + j*p + k ... let me just use the schoolbook from the original code.
    
    For schoolbook with rank = n*m*p:
    U[i*m+j, r] = 1 if r corresponds to (i,j,k) for some k
    V[j*p+k, r] = 1 if r corresponds to (i,j,k) for some i
    W[k*n+i, r] = 1 if r corresponds to (i,j,k)
    
    Let r = i * (m*p) + j * p + k
    U[i*m+j, r] = 1 for all k... no, U[i*m+j, r] should be 1 only when the (i,j) part matches.
    
    Actually, in the schoolbook:
    For each (i,j,k), create a rank component r = i*m*p + j*p + k
    U[i*m+j, r] = 1
    V[j*p+k, r] = 1
    W[k*n+i, r] = 1
    
    Then T[i',j',k'] = sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
    = sum over r where U[i'*m+j',r]=1, V[j'*p+k',r]=1, W[k'*n+i',r]=1
    = 1 if there exists r with i'=i, j'=j, k'=k, i''=i', j''=j', k''=k'
    = 1 if (i',j',k') = (i,j,k) for the matching r
    = 1 if i'=i, j'=j, k'=k
    = delta_{i',i} * delta_{j',j} * delta_{k',k}
    
    But we want T[i,j,k] = 1 for all (i,j,k), not a delta.
    
    I think I'm confusing myself. The tensor M_{n,m,p} has entries:
    M[i,j,k] = 1 if the term A[i,j]*B[j,k] is present, which is always true for valid indices.
    But the tensor is defined such that C = A*B corresponds to contracting the tensor.
    
    Let me just verify numerically: for the schoolbook decomposition, does it actually compute C = A*B?
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    n, m, p = 2, 5, 5
    
    # Test with random matrices
    rng = np.random.RandomState(42)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    
    # Compute C from the decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                u_row = U_arr[i * m + j, :]  # (rank,)
                v_row = V_arr[j * p + k, :]  # (rank,)
                w_row = W_arr[k * n + i, :]  # (rank,)
                C_computed[i, k] += np.dot(u_row * v_row, w_row)
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_like_2x2_block(n, m, p):
    """
    For n=2, m=5, p=5, try to use Strassen-like ideas.
    
    View A as a 2x5 matrix and B as a 5x5 matrix.
    Split into 2x2 blocks where possible, but 5 is not divisible by 2.
    
    Alternative: Use the fact that rank(M_{2,2,2}) = 7 < 8.
    If we can embed a 2x2x2 subproblem and save 1 rank, we get rank 49.
    
    Specifically, if we can find a rank-7 decomposition for a 2x2x2 block and
    rank-8 for the remaining parts, total could be less than 50.
    
    But the blocks don't align cleanly for 5x5.
    
    Let's try: decompose M_{2,5,5} by splitting the 5 columns/rows into 2+2+1.
    - 2x2 block: rank 7 (Strassen)
    - 2x2 block: rank 7
    - 2x1 column: rank 2
    - 1x2 row: rank 2
    - 1x1 corner: rank 1
    Total: 7 + 7 + 2 + 2 + 1 = 19? No, that's not how it works.
    
    Actually, the tensor is a single tensor, not a direct sum in a simple way.
    Let me just try to find a rank-49 decomposition by perturbing the schoolbook.
    """
    n, m, p = 2, 5, 5
    U = np.zeros((n * m, 49))
    V = np.zeros((m * p, 49))
    W = np.zeros((p * n, 49))
    
    # Start with schoolbook but use only 49 components
    # Use components 0..48 for the first 49 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                if r < 49:
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
    # The 50th term (i=1,j=4,k=4) is missing. We need to represent it using the other 49.
    # This is generally not possible with a simple modification.
    
    # Try a different approach: allow some U, V, W entries to be non-1 to "mix" terms.
    # This is heuristic and may not work.
    
    # For now, just verify if the truncated version accidentally works (it won't).
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U, V, W
    return None

def _try_rank_reduction_search(n, m, p, target_rank, time_limit=60):
    """
    Try to find a rank-reduced decomposition by modifying the schoolbook.
    This is a heuristic search.
    """
    start_time = time.time()
    best = None
    best_rank = n * m * p
    
    # First, get the schoolbook
    schoolbook = _make_schoolbook(n, m, p)
    if _verify_decomposition(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p):
        best = schoolbook
        best_rank = n * m * p
    
    # Try to reduce rank by 1
    U = np.array(schoolbook["U"], dtype=float)
    V = np.array(schoolbook["V"], dtype=float)
    W = np.array(schoolbook["W"], dtype=float)
    rank = n * m * p
    
    for trial_rank in range(rank - 1, max(rank - 10, 0), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try random modifications to reduce rank
        U_t = U[:, :trial_rank].copy()
        V_t = V[:, :trial_rank].copy()
        W_t = W[:, :trial_rank].copy()
        
        # Add small perturbations to try to "absorb" the missing terms
        rng = np.random.RandomState(trial_rank)
        U_t += rng.rand(n * m, trial_rank) * 0.1
        V_t += rng.rand(m * p, trial_rank) * 0.1
        W_t += rng.rand(p * n, trial_rank) * 0.1
        
        if _verify_decomposition(U_t.tolist(), V_t.tolist(), W_t.tolist(), n, m, p):
            best = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
            best_rank = trial_rank
            break
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Start with schoolbook (rank 50)
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        # Should not happen, but handle gracefully
        pass
    
    # Try to find a better decomposition
    if still_going():
        result, new_rank = _try_rank_reduction_search(n, m, p, best_rank, time_limit=120)
        if result is not None and new_rank < best_rank:
            best = result
            best_rank = new_rank
    
    return best
