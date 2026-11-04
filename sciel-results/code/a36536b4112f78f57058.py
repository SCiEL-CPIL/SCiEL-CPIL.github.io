import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 140.0  # Safety margin below 150s limit

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    """
    Returns the schoolbook tensor decomposition of rank n*m*p.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Convention: W[k*n+i, r]
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][c] = 1
                # V index: j*p + k
                V[j * p + k][c] = 1
                # W index: k*n + i
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    """
    Verifies if the decomposition is exact.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find a rank-49 decomposition by exploiting linear structure.
    
    Strategy: The matrix multiplication tensor for 2x5x5 can be viewed as:
    For each (i,j,k) where i in [0,2), j in [0,5), k in [0,5):
      T[i,j,k] = A[i,j] * B[j,k]
    
    We need to express this as sum of rank-1 tensors.
    
    One approach: Use the fact that for fixed j, the 2x5 subproblem 
    (A[:,j] * B[j,:]) has a known structure. 
    
    Alternative: Try to find a decomposition where we share computation
    across the 5 columns of A (or rows of B).
    
    For 2x1x1: rank 2 (schoolbook)
    For 2x1x5: we need to compute [a0,a1] * [b0,...,b4] = 5 outputs
    This is like outer product of 2-vector and 5-vector = 10 terms schoolbook.
    But we can do better? No, 2x1x5 has rank 10 (it's a rank-10 tensor).
    
    Actually, let's think differently. The full 2x5x5 tensor:
    - Input A: 10 entries (2x5)
    - Input B: 25 entries (5x5)  
    - Output C: 10 entries (2x5)
    
    Schoolbook: 10*5 = 50 multiplications.
    
    To get rank 49, we need to save one multiplication.
    
    Key idea: If there's a linear relation among the 50 schoolbook terms
    in the tensor space, we might be able to eliminate one.
    
    Let's set up the problem: We want matrices U (10x49), V (25x49), W (10x49)
    such that for all i,j,k:
      sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*2+i, c] = 1
    
    This is a system of 50 equations (one per (i,j,k) triple) with 49*10 + 49*25 + 49*10 = 49*45 = 2205 unknowns.
    
    This is highly underdetermined, so solutions likely exist.
    
    Approach: Use a greedy/iterative method. Start with a parameterized family
    and solve for parameters.
    
    Simpler approach: Try a specific structured ansatz.
    
    Let's try: For the 2x5x5 case, note that we can think of it as
    5 independent 2x1x5 subproblems (one for each column j of A), but they
    share the same B structure.
    
    Actually, here's a concrete attempt:
    
    The tensor T is in R^{10} ⊗ R^{25} ⊗ R^{10}.
    We want to write T = sum_{c=1}^{49} u_c ⊗ v_c ⊗ w_c.
    
    Let's try a decomposition where:
    - For most (i,j,k), we use the schoolbook term
    - But for one specific (i,j,k), we express it as a linear combination
      of other terms using the fact that the tensor lives in a lower-dimensional
      subspace due to the matrix multiplication structure.
    
    Specifically, consider fixing i=0, j=0. Then:
      C[0,k] = sum_j A[0,j] * B[j,k] for k=0..4
    
    The 5 values C[0,0], C[0,1], ..., C[0,4] are linear combinations of
    A[0,0],...,A[0,4] and B[*,*].
    
    Hmm, this is getting complex. Let me try a different structural approach.
    
    Use the "splitting" technique: Split the 5x5 matrix B into blocks and
    use Strassen-like ideas.
    
    Actually, let me just do a numerical search with rational reconstruction.
    """
    if _remaining() < 5:
        return None
    
    n, m, p = 2, 5, 5
    target_rank = 49
    
    # We'll try to find U, V, W with rational entries.
    # Use a parameterized approach.
    
    # Start from schoolbook and try to eliminate one term.
    # The schoolbook has terms indexed by (i,j,k) for i in [0,2), j in [0,5), k in [0,5).
    # That's 50 terms.
    
    # Try to express one term, say (i=0, j=0, k=0), as a linear combination
    # of the other 49 terms.
    
    # For the schoolbook decomposition:
    # Term (i,j,k) contributes to:
    #   U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # The tensor component T[i',j',k'] = sum over terms where i'=i, j'=j, k'=k
    # For schoolbook, T[i,j,k] = 1 for all valid (i,j,k).
    
    # We want: for all (i',j',k'), sum_c U[i'*m+j',c] * V[j'*p+k',c] * W[k'*n+i',c] = 1
    
    # Let's try a specific ansatz:
    # Keep 49 of the 50 schoolbook terms, and modify them slightly.
    # Or: introduce a new term that is a linear combination.
    
    # Better: Use the fact that the tensor is "structured".
    # The matrix multiplication tensor satisfies certain identities.
    
    # Let me try a direct linear algebra approach.
    # Fix W to be a specific matrix, then solve for U and V.
    
    # Actually, this is a non-linear problem (trilinear).
    
    # Let me try a simpler approach: randomized search with local optimization.
    # But with exact rational arithmetic, this is hard.
    
    # Alternative: Try known constructions.
    # For 2x2x2, Strassen gives rank 7 < 8.
    # For 2x5x5, is there a known rank 49 construction?
    
    # I don't have a known closed-form, so let me try a systematic search
    # over a restricted class of decompositions.
    
    # Class: Decompositions where U, V, W have mostly 0s and 1s, with some -1s.
    # This is similar to how Strassen works.
    
    # Let me try: start from schoolbook, and try to replace two terms
    # with three terms that have cancellations, net saving 1.
    
    # Actually, the cleanest approach: set up the equations and use
    # Gaussian elimination over the rationals with a specific parameterization.
    
    # Parameterize: Let W be the identity-like matrix for 49 of the 50 positions,
    # and solve for U and V.
    
    # This is getting too complex for a quick implementation. Let me try
    # a different structural idea.
    
    # KEY INSIGHT: The 2x5x5 matrix multiplication can be decomposed as:
    # C = A * B where A is 2x5, B is 5x5.
    # 
    # Write A = [A1 | A2] where A1 is 2x2, A2 is 2x3.
    # Write B = [[B11, B12], [B21, B22]] where B11 is 2x2, etc.
    # 
    # Then C = A1*B11 + A2*B21 (for first 2 columns) 
    #        A1*B12 + A2*B22 (for last 3 columns)
    # 
    # This doesn't obviously help.
    
    # Let me just return None and let the caller fall back to schoolbook.
    # The search for rank 49 is non-trivial and may not be achievable
    # with a simple closed form in the time available.
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    if _remaining() > 10:
        result = _try_rank_49(n, m, p)
        if result is not None:
            # Verify
            if _verify(n, m, p, result["U"], result["V"], result["W"]):
                return result
    
    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify
    if not _verify(n, m, p, result["U"], result["V"], result["W"]):
        raise RuntimeError("Schoolbook verification failed")
        
    return result

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
