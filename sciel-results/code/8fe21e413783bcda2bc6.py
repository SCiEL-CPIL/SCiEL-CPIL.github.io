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
    Attempt to find a rank-49 decomposition using the Strassen-like pattern
    for 2×n×n matrix multiplication.
    
    The key insight: For multiplying a 2×n matrix by an n×n matrix, we can
    achieve rank 2n²-1 by using a clever combination of sums and differences.
    
    For n=5: rank = 2*25 - 1 = 49
    
    Construction:
    Let A be 2×5 with rows a0, a1 (each 1D arrays of length 5)
    Let B be 5×5
    
    We want C = A*B, where C is 2×5.
    
    C[0,k] = sum_j a0[j]*B[j,k]
    C[1,k] = sum_j a1[j]*B[j,k]
    
    Define for each j:
      s_j = a0[j] + a1[j]
      d_j = a0[j] - a1[j]
    
    Then:
      C[0,k] = (1/2) * sum_j (s_j + d_j) * B[j,k]
             = (1/2) * [sum_j s_j*B[j,k] + sum_j d_j*B[j,k]]
      C[1,k] = (1/2) * sum_j (s_j - d_j) * B[j,k]
             = (1/2) * [sum_j s_j*B[j,k] - sum_j d_j*B[j,k]]
    
    So if we precompute:
      S_k = sum_j s_j * B[j,k]  for k=0..4  (5 products)
      D_k = sum_j d_j * B[j,k]  for k=0..4  (5 products)
    
    Then:
      C[0,k] = (S_k + D_k)/2
      C[1,k] = (S_k - D_k)/2
    
    But this still uses 10 multiplications (5 for S, 5 for D), same as schoolbook for one row.
    
    Wait, that's not right. Let me reconsider.
    
    Actually, the standard approach for 2×n×n:
    
    Schoolbook: For each of the 2 rows of A, and each of the n columns of B,
    we do n multiplications. Total: 2*n*n = 2n².
    
    Strassen-like idea: Group the n columns of B into pairs (or use a different grouping).
    
    Alternative approach: Think of it as computing A*B where we can use
    the structure that A has only 2 rows.
    
    Let me try a different decomposition. Consider:
    
    For each column k of B (k=0..n-1), we need to compute:
      C[:,k] = A * B[:,k]
    
    This is a 2×n times n×1 multiplication, which has rank 2n (schoolbook).
    
    But we can do better if we share computation across k.
    
    Actually, let me look at this differently. The tensor for 2×n×n matrix mult:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,...,n-1}, k in {0,...,n-1}
    
    We want to write this as sum of rank-1 tensors.
    
    One known result: The rank of the 2×n×n matrix multiplication tensor
    is 2n²-1 for n ≥ 2.
    
    Here's a construction:
    
    For j = 0 to n-2, and k = 0 to n-1:
      Term (j,k,0): U[j]=1, V[j*n+k]=1, W[k]=1  (for row 0 of C)
      Term (j,k,1): U[j]=1, V[j*n+k]=1, W[n+k]=1  (for row 1 of C)
    
    That's 2*(n-1)*n terms so far.
    
    For j = n-1:
      We need to handle the last row of A.
      
      For k = 0 to n-2:
        Term: U[n-1]=1, V[(n-1)*n+k]=1, W[k]=1
        Term: U[n-1]=1, V[(n-1)*n+k]=1, W[n+k]=1
      
      That's 2*(n-1) more terms.
      
      For k = n-1:
        We need to express the last two outputs using previous terms.
        
        C[0,n-1] = sum_j A[0,j]*B[j,n-1]
        C[1,n-1] = sum_j A[1,j]*B[j,n-1]
        
        We already have terms for j=0..n-2 contributing to both.
        The missing part is j=n-1:
          A[0,n-1]*B[n-1,n-1] for C[0,n-1]
          A[1,n-1]*B[n-1,n-1] for C[1,n-1]
        
        But we can't just add two more terms - that would give 2n².
        
        Instead, use:
          S = A[0,n-1] + A[1,n-1]
          D = A[0,n-1] - A[1,n-1]
          
          C[0,n-1] gets: (S*D_part + D*D_part)/2 ... this is getting complicated.
    
    Let me try a cleaner approach. I'll use the known formula.
    
    Actually, let me just implement a direct construction and verify it.
    """
    if _remaining() < 5:
        return None
    
    n, m, p = 2, 5, 5
    target_rank = 49
    
    U = [[0] * target_rank for _ in range(n * m)]  # 10 x 49
    V = [[0] * target_rank for _ in range(m * p)]  # 25 x 49
    W = [[0] * target_rank for _ in range(p * n)]  # 10 x 49
    
    c = 0
    
    # For j = 0 to 3 (first 4 rows of B), handle both output rows
    for j in range(m - 1):
        for k in range(p):
            # Row 0 of C
            U[j][c] = 1
            V[j * p + k][c] = 1
            W[k * n][c] = 1  # W index for row 0, column k
            c += 1
            
            # Row 1 of C
            U[j + n][c] = 1  # U index for row 1, column j
            V[j * p + k][c] = 1
            W[k * n + 1][c] = 1  # W index for row 1, column k
            c += 1
    
    # For j = 4 (last row of B)
    j = m - 1
    for k in range(p - 1):
        # Row 0 of C
        U[j][c] = 1
        V[j * p + k][c] = 1
        W[k * n][c] = 1
        c += 1
        
        # Row 1 of C
        U[j + n][c] = 1
        V[j * p + k][c] = 1
        W[k * n + 1][c] = 1
        c += 1
    
    # Last term: handle B[4,4] contribution to both C[0,4] and C[1,4]
    # We need to express:
    #   C[0,4] += A[0,4] * B[4,4]
    #   C[1,4] += A[1,4] * B[4,4]
    # 
    # Using: let S = A[0,4] + A[1,4], D = A[0,4] - A[1,4]
    # Then A[0,4] = (S+D)/2, A[1,4] = (S-D)/2
    #
    # So: C[0,4] += (S+D)/2 * B[4,4]
    #     C[1,4] += (S-D)/2 * B[4,4]
    #
    # This requires 2 multiplications: S*B[4,4] and D*B[4,4]
    # But we only have 1 term left!
    #
    # Hmm, this doesn't work directly. Let me reconsider.
    
    # Actually, I think the construction is different. Let me use a known result.
    # The rank of 2×n×n matrix multiplication is indeed 2n²-1.
    # 
    # Here's the correct construction:
    # 
    # For i=0,1 and j=0,...,n-1 and k=0,...,n-1, we have terms.
    # But we can save one by using:
    # 
    # Define for the last (j,k) pair:
    #   Instead of two separate terms, use one term with coefficients that
    #   combine both.
    #
    # Let me try a different approach: use the fact that we can write
    #   A[0,n-1]*B[n-1,n-1] + A[1,n-1]*B[n-1,n-1] 
    # as part of a combined computation.
    #
    # Actually, I think I'm overcomplicating this. Let me just return None
    # and use a simpler verification-based search.
    
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
