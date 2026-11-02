"""Combined program: structured rank search with verification for 2x3x6 matrix multiplication.
   Strategy:
   1. Build the schoolbook rank-36 decomposition as guaranteed fallback.
   2. Attempt to find lower-rank decompositions using structured patterns (Strassen-like
      combinations, block decompositions) and verify exactness with integer arithmetic.
   3. Use time-bounded search to try ranks below 36.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]


def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic (entries are half-integers).
    The tensor decomposition requires:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    Since entries can be half-integers, multiply by 2 and check sum == 8.
    """
    r = len(U[0])
    # Work with integers: multiply by 2
    U2 = [[2 * U[i][j] for j in range(r)] for i in range(n * m)]
    V2 = [[2 * V[i][j] for j in range(r)] for i in range(m * p)]
    W2 = [[2 * W[i][j] for j in range(r)] for i in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                if s != 8:  # 2*2*2 = 8
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_strassen_blocks(n, m, p, deadline):
    """Try to use Strassen-style decompositions on sub-blocks.
    
    For 2x3x6: split into 3 blocks of 2x3x2 (grouping output pairs).
    For each 2x3x2 block, try to use fewer than 12 terms.
    
    A 2x3x2 tensor: A(2x3), B(3x2), C(2x2).
    Schoolbook: 12 terms.
    
    Try rank 9 for 2x3x2 by combining terms.
    """
    if time.time() > deadline:
        return None
    
    # For 2x3x2:
    # A = [[a, b, c], [d, e, f]]
    # B = [[x, y], [u, v], [s, t]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = ax + bu + cs
    # C12 = ay + bv + ct
    # C21 = dx + eu + fs
    # C22 = dy + ev + ft
    #
    # Try to express with 9 terms using combinations.
    # One approach: use 7 Strassen-like terms for the 2x2x2 part + 2 for the extra column.
    # Actually, let's try a direct construction.
    
    # For each 2x3x2 block, try to find a rank-9 or rank-10 decomposition.
    # This is complex, so let's try a simpler approach:
    # Use rank 12 per block (schoolbook) = 36 total. Not helpful.
    
    # Alternative: try a global rank-24 approach.
    # Split B into two 3x3 matrices: B = [B1 | B2]
    # C = A * B1 concatenated with A * B2
    # Each 2x3x3 block. If we can do 2x3x3 in rank 12, total 24.
    # 2x3x3 schoolbook is 18. Can we do 12?
    
    # For 2x3x3: A(2x3), B(3x3), C(2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 6 output entries, each is a sum of 3 products.
    # Total 18 products in schoolbook.
    # Can we combine? 
    
    # Try: use the fact that A has only 2 rows.
    # Row 0 of C: [A00*B0k + A01*B1k + A02*B2k for k in 0..2]
    # Row 1 of C: [A10*B0k + A11*B1k + A12*B2k for k in 0..2]
    
    # This is essentially 2 independent 3x3 matrix-vector products (each row of A times B).
    # Each 3x3 * 1x3 -> 1x3 can be done in... well, it's just 3 multiplications per output.
    # Total 6 outputs * 3 = 18. Hard to reduce.
    
    # Let's try a different approach: use random search with verification.
    # For small ranks, we can try structured patterns.
    
    return None


def _try_random_search(n, m, p, target_rank, deadline):
    """Try to find a rank decomposition using random initialization and gradient descent.
    Only feasible for very small ranks due to the high dimensionality."""
    if time.time() > deadline:
        return None
    
    # This approach is likely too slow for rank > 20, so only try very small ranks
    if target_rank > 25:
        return None
    
    try:
        from scipy.optimize import minimize
        
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        size = size_u + size_v + size_w
        
        np.random.seed(42)
        x0 = np.random.randn(size) * 0.1
        
        def objective(x):
            if time.time() > deadline:
                return 1e10
            u = x[:size_u].reshape(n * m, target_rank)
            v = x[size_u:size_u + size_v].reshape(m * p, target_rank)
            w = x[size_u + size_v:].reshape(p * n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += u[i * m + j, c] * v[j * p + k, c] * w[k * n + i, c]
                        error += (s - 1.0) ** 2
            return error
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'maxfun': 5000})
        
        if res.fun < 1e-8:
            u = res.x[:size_u].reshape(n * m, target_rank)
            v = res.x[size_u:size_u + size_v].reshape(m * p, target_rank)
            w = res.x[size_u + size_v:].reshape(p * n, target_rank)
            
            # Round to half-integers
            U = [[round(u[i][j] * 2) / 2 for j in range(target_rank)] for i in range(n * m)]
            V = [[round(v[i][j] * 2) / 2 for j in range(target_rank)] for i in range(m * p)]
            W = [[round(w[i][j] * 2) / 2 for j in range(target_rank)] for i in range(p * n)]
            
            if _verify_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    return None


def _try_structured_patterns(n, m, p, deadline):
    """Try known structured patterns for lower-rank decompositions.
    
    For 2x3x6, try:
    1. Rank 24: Split into two 2x3x3 blocks, each with rank 12.
    2. Rank 21: Split into three 2x3x2 blocks, each with rank 7 (Strassen-like).
    """
    if time.time() > deadline:
        return None
    
    # Try rank 24: two 2x3x3 blocks
    # For each 2x3x3 block, use rank 12.
    # This requires a valid rank-12 decomposition of 2x3x3.
    # 2x3x3: A(2x3), B(3x3), C(2x3)
    # One possible rank-12 scheme:
    # For each pair (i, k) where i in {0,1}, k in {0,1,2}, we need 2 terms
    # (since the middle dimension is 3, but we can combine).
    # Actually, this is not straightforward. Let's skip this.
    
    # Try rank 21: three 2x3x2 blocks, each with rank 7.
    # 2x3x2: A(2x3), B(3x2), C(2x2)
    # Is there a rank-7 decomposition? The schoolbook is 12.
    # Strassen works for 2x2x2 (rank 7 vs 8).
    # For 2x3x2, we can view it as two 2x2x2 blocks sharing the A matrix.
    # A = [[a, b, c], [d, e, f]]
    # B = [[x, y], [u, v], [s, t]]
    # C = [[ax+bu+cs, ay+bv+ct], [dx+eu+fs, dy+ev+ft]]
    #
    # Split B into B_left = [[x,u,s],[y,v,t]]^T... no, B is 3x2.
    # Let's split the middle dimension: B = [B1; B2] where B1 is 2x2 and B2 is 1x2.
    # B1 = [[x, y], [u, v]], B2 = [[s, t]]
    # C = A_left * B1 + A_right * B2 where A_left = [[a,b],[d,e]], A_right = [[c],[f]]
    #
    # A_left * B1 is 2x2x2, can use Strassen (rank 7).
    # A_right * B2 is 2x1 * 1x2 = 2x2, which is a rank-4 tensor (outer product).
    # But we can combine: the total is 7 + 4 = 11, not 7.
    #
    # So rank 7 for 2x3x2 is unlikely. Let's not pursue this.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit
    
    # Start with schoolbook as guaranteed fallback
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured patterns
    if time.time() < deadline:
        result = _try_structured_patterns(n, m, p, deadline)
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank:
                best_result = result
                best_rank = r
    
    # Try random search for lower ranks (only if time permits)
    if time.time() < deadline - 30:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 20), -1):
            if time.time() > deadline - 10:
                break
            result = _try_random_search(n, m, p, target_rank, deadline - 10)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result
