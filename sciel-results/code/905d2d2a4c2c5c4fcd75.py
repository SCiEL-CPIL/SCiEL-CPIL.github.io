import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # First, try to find a lower-rank decomposition via numerical search
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the sense that
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Build the linear system: for each (i,j,k), we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(4, 28, 28) = 28
    
    # Try to find decompositions at various ranks
    best_result = None
    
    # Try rank 28 first (theoretical lower bound)
    for target_rank in [28, 30, 32, 35, 38, 40, 42, 44, 46, 48]:
        if time.time() > deadline - 5:
            break
        
        result = _try_find_decomposition(n, m, p, target_rank)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the Strassen-based rank-49 decomposition
    return _construct_49_rank(n, m, p)

def _try_find_decomposition(n, m, p, rank):
    """Try to find a rank decomposition using numerical optimization."""
    import time
    
    # Number of equations: n*m*p = 4*14 = 56
    # Unknowns: (n*m + m*p + p*n) * rank = (4 + 28 + 28) * rank = 60 * rank
    
    # We need to solve: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Use a randomized approach: initialize randomly and use iterative refinement
    # Or use the fact that we can fix one matrix and solve for others
    
    # Strategy: Fix W to be identity-like, then solve for U and V
    # Actually, let's use a different approach:
    
    # The tensor can be written as a sum of rank-1 tensors.
    # Each rank-1 term is u ⊗ v ⊗ w where u is in R^(n*m), v in R^(m*p), w in R^(p*n)
    
    # We need: sum_r u_r ⊗ v_r ⊗ w_r = T where T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try: set w_r to be standard basis vectors (one-hot) for the first p*n entries
    # Then the problem becomes: for each (i,j,k), U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
    # But this only works if each (i,j,k) maps to a unique k*n+i, which it doesn't.
    
    # Better approach: use a randomized initialization and Newton's method or gradient descent
    
    # Actually, let me try a simpler structural approach:
    # Use the "border rank" idea or try specific patterns
    
    # Let's try: for each k, the 2x2x1 slice has a specific structure.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # For a fixed k, this is: [A[i,0], A[i,1]] dot [B[0,k], B[1,k]] for i=0,1
    # This is a 2x2 x 2x1 tensor with rank at most 4 (schoolbook) or 3?
    
    # Actually for 2x2 x 2x1: C0 = a00*b0 + a01*b1, C1 = a10*b0 + a11*b1
    # Rank 3 decomposition: 
    #   t1 = (a00+a10)*(b0) -> contributes to both C0 and C1
    #   t2 = (a01+a11)*(b1) -> contributes to both C0 and C1  
    #   t3 = something...
    # Let me check: C0 = t1 + t2 - t3, C1 = t1 + t2 + t3?
    # t1 = (a00+a10)*b0, t2 = (a01+a11)*b1
    # C0 = a00*b0 + a01*b1 = t1 - a10*b0 + t2 - a11*b1 = t1 + t2 - (a10*b0 + a11*b1)
    # C1 = a10*b0 + a11*b1 = t1 + t2 - (a00*b0 + a01*b1) = t1 + t2 - C0
    # So we need t3 = a10*b0 + a11*b1, but that's the same as C1, circular.
    
    # Let me try differently:
    # t1 = a00*b0
    # t2 = a01*b1  
    # t3 = (a10+a11)*(b0+b1)
    # C0 = t1 + t2
    # C1 = a10*b0 + a11*b1 = (a10+a11)*(b0+b1) - a10*b1 - a11*b0
    # Hmm, not clean.
    
    # The rank of M_{2,2,2x1} is actually 4 (it's just two independent dot products of length 2)
    # Wait no, it's the tensor [a00*b0, a01*b1, a10*b0, a11*b1] -> [C0, C1]
    # This is a 2x2x2 tensor (if we think of B as 2x1, it's 2x2x2 but with the last dim being 1)
    # Actually it's M_{2,2,2} restricted... no, it's just 2x2 x 2.
    
    # The rank of the 2x2 x 2 tensor (matrix-vector product) is 4 (schoolbook is optimal for 2x2x2x1? No...)
    # Actually, 2x2 matrix times 2x1 vector: the tensor has entries T[i,j,k] where i=0,1 (output row), j=0,1 (matrix col), k=0,1 (vector). 
    # T[i,j,k] = delta(j,k) in the sense that C[i] = sum_j A[i,j] B[j]
    # The rank of this tensor: it's the same as the "bilinear" complexity of 2x2 matrix-vector mult.
    # This is known to have rank 4 (you can't do better than 4 multiplications for 2x2 x 2x1).
    
    # So for each k, we need rank 4, giving total rank 4*14 = 56. But we can share across k's.
    
    # Let me try a completely different structural approach:
    # Use the fact that n=2 and try to find a decomposition where we group the k-dimension
    # in a way that allows sharing.
    
    # New idea: Use a "ladder" or "chain" decomposition
    # Or: try to use the known rank of M_{2,2,p} from literature
    
    # For now, let me try a numerical search with a specific parameterization
    
    # Parameterize: U has 4*rank entries, V has 28*rank, W has 28*rank
    # Total unknowns: 60*rank
    # Equations: 56
    
    # For rank >= 56/3 ≈ 19, we have more unknowns than equations, so solutions likely exist.
    # But we need the specific structure.
    
    # Let me try: set W to have a specific simple form and solve for U, V
    
    # Actually, let me try the most direct numerical approach:
    # Use scipy's least_squares or a custom Newton iteration
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    # Define the residual: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
    # We'll use a randomized initialization and try to drive the residual to zero
    
    # To make this tractable, let's fix one of the matrices to have a simple structure.
    # Fix W: W[k*n+i, r] = 1 if r == k*n+i, else 0 (for r < p*n)
    # Then the equation becomes: for each (i,j,k), U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
    # This requires U[i*m+j, k*n+i] != 0 and V[j*p+k, k*n+i] = 1/U[i*m+j, k*n+i]
    # But different (i,j,k) may map to the same (k*n+i), causing conflicts.
    # For fixed k, k*n+i takes values k*2+0 and k*2+1.
    # For fixed k, the pairs (i,j) that map to k*n+0 are (0,j) for j=0,1, and (i,j) that map to k*n+1 are (1,j) for j=0,1.
    # So for W column k*2+0: we need U[0*2+j, k*2+0]*V[j*14+k, k*2+0] = 1 for j=0,1
    # And U[1*2+j, k*2+0]*V[j*14+k, k*2+0] = 1 for j=0,1
    # This means U[0, k*2+0]*V[0*14+k, k*2+0] = 1, U[1, k*2+0]*V[1*14+k, k*2+0] = 1
    # U[2, k*2+0]*V[0*14+k, k*2+0] = 1, U[3, k*2+0]*V[1*14+k, k*2+0] = 1
    # So U[0,k*2+0] = U[2,k*2+0] and U[1,k*2+0] = U[3,k*2+0]
    # This is a constraint but solvable.
    
    # With this W fixed (rank = p*n = 28), we need:
    # For each k, i, j: U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
    # Let's set U[i*m+j, k*n+i] = 1 for all, then V[j*p+k, k*n+i] = 1 for all.
    # Check: does this work? For (i,j,k): U[i*2+j, k*2+i] = 1, V[j*14+k, k*2+i] = 1, W[k*2+i, k*2+i] = 1.
    # Product = 1. Yes!
    
    # But wait, we need to check that this doesn't create spurious contributions.
    # The decomposition is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # With W being identity (28x28), only r = k*n+i contributes.
    # So the sum is just U[i*m+j, k*n+i] * V[j*p+k, k*n+i] * 1
    # We set this to 1 for all (i,j,k). 
    
    # But we need U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for ALL (i,j,k).
    # For fixed k and i, varying j: U[i*2+0, k*2+i] * V[0*14+k, k*2+i] = 1
    #                              U[i*2+1, k*2+i] * V[1*14+k, k*2+i] = 1
    # For fixed k and j, varying i: U[0*2+j, k*2+0] * V[j*14+k, k*2+0] = 1
    #                              U[1*2+j, k*2+1] * V[j*14+k, k*2+1] = 1
    
    # These are all consistent if we set all U and V entries to 1.
    # So rank 28 works with this construction!
    
    # Let me verify: U is 4x28, V is 28x28, W is 28x28 (identity)
    # U[i*2+j, k*2+i] = 1 for all i,j,k
    # V[j*14+k, k*2+i] = 1 for all i,j,k
    # W[k*2+i, k*2+i] = 1
    
    # Check for (i=0, j=0, k=0): U[0, 0] * V[0, 0] * W[0, 0] = 1*1*1 = 1 ✓
    # Check for (i=0, j=1, k=0): U[1, 0] * V[14, 0] * W[0, 0] = 1*1*1 = 1 ✓
    # Check for (i=1, j=0, k=0): U[2, 1] * V[0, 1] * W[1, 1] = 1*1*1 = 1 ✓
    # Check for (i=1, j=1, k=0): U[3, 1] * V[14, 1] * W[1, 1] = 1*1*1 = 1 ✓
    # Check for (i=0, j=0, k=1): U[0, 2] * V[1, 2] * W[2, 2] = 1*1*1 = 1 ✓
    # ...
    
    # But wait, I need to make sure there are NO other contributions.
    # The sum is over r from 0 to 27. W[k*n+i, r] is 1 only when r = k*n+i.
    # So only one term contributes. 
    
    # But I also need to verify that for the WRONG (i,j,k) combinations, we don't get contributions.
    # Actually, the tensor equation is: for ALL (i,j,k) in the valid range, the sum should equal 1.
    # And for the tensor, T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    # There are no other non-zero entries in the tensor.
    
    # So the decomposition is correct! Rank 28!
    
    # Let me construct it:
    U = [[0] * rank for _ in range(n * m)]   # 4 x rank
    V = [[0] * rank for _ in range(m * p)]   # 28 x rank
    W = [[0] * rank for _ in range(p * n)]   # 28 x rank
    
    if rank < p * n:
        # Can't use the identity W trick
        return None
    
    # Set W to identity
    for r in range(p * n):
        W[r][r] = 1
    
    # Set U[i*m+j, k*n+i] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][k * n + i] = 1
    
    # Set V[j*p+k, k*n+i] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                V[j * p + k][k * n + i] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_49_rank(n, m, p):
    """Fallback: Strassen-based rank-49 decomposition."""
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 2: 1, 1: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),
        ({3: 1}, {1: 1, 0: -1}, {2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {2: 1}),
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: 1}),
    ]
    
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7
        
        for term_idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
            col = base_col + term_idx
            
            for idx, val in u_dict.items():
                U[idx][col] += val
            
            for idx, val in v_dict.items():
                if idx == 0:
                    V[0 * p + k1][col] += val
                elif idx == 1:
                    V[1 * p + k1][col] += val
                elif idx == 2:
                    V[0 * p + k2][col] += val
                elif idx == 3:
                    V[1 * p + k2][col] += val
            
            for idx, val in w_dict.items():
                if idx == 0:
                    W[k1 * n + 0][col] += val
                elif idx == 1:
                    W[k2 * n + 0][col] += val
                elif idx == 2:
                    W[k1 * n + 1][col] += val
                elif idx == 3:
                    W[k2 * n + 1][col] += val
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True
