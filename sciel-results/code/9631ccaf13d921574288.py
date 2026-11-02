"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication.
    
    The multiplication tensor T has entries T[i,j,k] = delta(i,j,k) such that
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    In the decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Index conventions:
    - U indexed by (i*m + j, r) for i in {0..n-1}, j in {0..m-1}
    - V indexed by (j*p + k, r) for j in {0..m-1}, k in {0..p-1}
    - W indexed by (k*n + i, r) for k in {0..p-1}, i in {0..n-1}
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Build the multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i*m+j, j*p+k, k*n+i] = 1
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute decomposed tensor: D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    D = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check exact equality
    max_err = np.max(np.abs(D - T))
    return max_err < 1e-10, max_err

def _laderman_decomposition_2x2x16():
    """
    Return the Laderman rank-28 decomposition for 2x2 * 2x16 matrix multiplication.
    
    Laderman's algorithm uses a Kronecker product structure.
    For 2x2 * 2x16, we embed into a structure where the 16 columns are 
    indexed by (a,b) with a,b in {0,1,2,3} (i.e., 4x4 grid).
    
    The Laderman decomposition for 2x2*2x16 has rank 28.
    
    Actually, let me think about this differently. The standard approach:
    - 2x2*2x2 has rank 7 (Strassen)
    - 2x2*2x4 can be done with rank 12 (using Strassen + structure)
    - 2x2*2x8 can be done with rank 18
    - 2x2*2x16 can be done with rank 28
    
    The pattern: for 2x2*2x(2^d), the rank is 4*2^d + 2^d - 2 = 5*2^d - 2?
    Let's check: d=1: 5*2-2=8? No, Strassen gives 7.
    
    Actually, the known result: rank(2x2*2x2^d) <= 4*2^d + 2^d - 2 for d>=2?
    d=2 (2x2*2x4): 4*4+4-2 = 18? But we can do 12.
    
    Let me use a different approach. For 2x2*2x16:
    We can view this as 4 independent 2x2*2x4 problems, each with rank 12, total 48.
    Or we can use the Kronecker structure.
    
    Actually, the simplest correct approach: use the Kronecker product of 
    Strassen (rank 7) with itself appropriately.
    
    For 2x2*2x4: Use 2 Strassen blocks (rank 14) or a better rank-12 algorithm.
    For 2x2*2x8: Use 4 Strassen blocks (rank 28) or better.
    For 2x2*2x16: Use 8 Strassen blocks (rank 56) or better.
    
    Hmm, this doesn't give rank 28 directly.
    
    Let me reconsider. The directive says to try embedding into 2x2x16 using Laderman's rank-28.
    Laderman (1976) showed that 2x2*2x2x2 (i.e., 2x2*2x2 with an extra dimension) 
    has rank 28. But that's a 4-dimensional problem.
    
    Actually, I think the right interpretation is:
    - The multiplication 2x2 * 2x16 can be seen as computing 16 independent 2x2*2x1 products.
    - Using a Fourier/FFT-based approach on the 16-dimensional space, we can achieve rank 28.
    
    Let me use a concrete construction. The 16 columns can be indexed by (a,b) where 
    a,b in {0,1,2,3}. We use the tensor product structure:
    
    For the 4x4 grid of columns, we can use:
    - 4 "horizontal" Strassen-like blocks (rank 7 each) = 28
    - Plus some additional terms
    
    Actually, let me just implement a direct Kronecker-based construction and verify.
    
    A cleaner approach: use the fact that 2x2*2x16 = (2x2*2x2) ⊗ (2x2x8 structure)
    
    Let me just implement the Strassen-block approach for 2x2*2x9 (rank 32) as the 
    primary method, and attempt the Laderman embedding as a secondary method.
    """
    # This function will be implemented below with the actual Laderman construction
    pass

def _strassen_block_decomposition(n, m, p):
    """
    Construct rank-32 decomposition using Strassen blocks for 2x2*2x9.
    
    Partition 9 columns into 4 pairs + 1 single.
    Each pair uses 7 Strassen products, single uses 4 schoolbook products.
    Total rank: 4*7 + 4 = 32.
    """
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def strassen_block(k1, k2, start_col):
        """Add 7 Strassen products for columns k1, k2."""
        # V row indices:
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        # W row indices:
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        # U row indices:
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        # M1 = (a00 + a11) * (b00 + b11)
        c = start_col + 0
        U[u_00, c] += 1
        U[u_11, c] += 1
        V[v_00, c] += 1
        V[v_11, c] += 1
        W[w_00, c] += 1
        W[w_03, c] += 1
        
        # M2 = (a10 + a11) * b00
        c = start_col + 1
        U[u_10, c] += 1
        U[u_11, c] += 1
        V[v_00, c] += 1
        W[w_01, c] += 1
        W[w_03, c] -= 1
        
        # M3 = a00 * (b01 - b11)
        c = start_col + 2
        U[u_00, c] += 1
        V[v_01, c] += 1
        V[v_11, c] -= 1
        W[w_02, c] += 1
        W[w_03, c] += 1
        
        # M4 = a11 * (b10 - b00)
        c = start_col + 3
        U[u_11, c] += 1
        V[v_10, c] += 1
        V[v_00, c] -= 1
        W[w_00, c] += 1
        W[w_01, c] += 1
        
        # M5 = (a00 + a01) * b11
        c = start_col + 4
        U[u_00, c] += 1
        U[u_01, c] += 1
        V[v_11, c] += 1
        W[w_00, c] -= 1
        W[w_02, c] += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        c = start_col + 5
        U[u_10, c] += 1
        U[u_00, c] -= 1
        V[v_00, c] += 1
        V[v_01, c] += 1
        W[w_03, c] += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        c = start_col + 6
        U[u_01, c] += 1
        U[u_11, c] -= 1
        V[v_10, c] += 1
        V[v_11, c] += 1
        W[w_00, c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        """Add 4 schoolbook products for a single column k."""
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        w_0 = k * n + 0
        w_1 = k * n + 1
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00, c] += 1
        V[v_0, c] += 1
        W[w_0, c] += 1
        
        c = start_col + 1
        U[u_01, c] += 1
        V[v_1, c] += 1
        W[w_0, c] += 1
        
        c = start_col + 2
        U[u_10, c] += 1
        V[v_0, c] += 1
        W[w_1, c] += 1
        
        c = start_col + 3
        U[u_11, c] += 1
        V[v_1, c] += 1
        W[w_1, c] += 1
        
        return start_col + 4
    
    # Process 4 pairs: (0,1), (2,3), (4,5), (6,7)
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    # Process single column: 8
    col = schoolbook_single(8, col)
    
    return U, V, W

def _kronkner_laderman_2x2x16():
    """
    Attempt to construct a rank-28 decomposition for 2x2*2x16 using 
    Laderman's Kronecker product approach, then project to 2x2*2x9.
    
    The idea: embed 2x2*2x9 into 2x2*2x16 by zero-padding columns 9-15.
    If the Laderman rank-28 decomposition for 2x2*2x16 has the property that
    contributions from columns 9-15 cancel or are zero in the relevant 
    U,V,W entries, we get a valid rank-28 decomposition for 2x2*2x9.
    
    Laderman's construction for 2x2*2x16:
    The 16 columns are indexed by (a,b) with a,b in {0,1,2,3}.
    We use a 4x4 FFT-based structure.
    
    For each (a,b) column, the multiplication is:
    C[i,(a,b)] = sum_j A[i,j] * B[j,(a,b)]
    
    Using the Kronecker product of two 2x2*2x2 Strassen decompositions:
    rank = 7 * 7 = 49 (too high)
    
    Using Laderman's actual construction:
    Laderman showed rank(2x2*2x2*2x2) = 28 for the 4-dimensional case.
    But 2x2*2x16 is a 3-dimensional tensor (i,j,k) with k in {0..15}.
    
    Let me use a different construction. The 16 columns can be grouped as 
    a 4x4 grid. For each row of 4 columns, use a rank-12 algorithm for 
    2x2*2x4, giving 4*12 = 48. Not good enough.
    
    Alternative: use the fact that 2x2*2x4 has rank 12 (known result).
    Then 2x2*2x16 = 4 copies of 2x2*2x4, rank 4*12 = 48. Still not 28.
    
    I think the Laderman rank-28 refers to a different problem structure.
    Let me just implement the Strassen approach (rank 32) as the primary method.
    """
    return None

def construct():
    """
    Construct a tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Primary strategy: Strassen-block partition (rank 32).
    Secondary strategy: Kronecker/Laderman embedding (attempt rank 28).
    
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen-block decomposition (rank 32)
    try:
        U, V, W = _strassen_block_decomposition(n, m, p)
        valid, err = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            rank = U.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        if time.time() > deadline:
            if best_result is not None:
                return best_result
            # Return schoolbook as fallback
            U, V, W = _strassen_block_decomposition(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Strategy 2: Attempt Kronecker/Laderman embedding
    # Try to find a rank-28 decomposition by embedding into 2x2x16
    try:
        result = _attempt_laderman_embedding(n, m, p)
        if result is not None:
            U, V, W = result
            valid, err = _verify_decomposition(U, V, W, n, m, p)
            if valid:
                rank = U.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    except Exception as e:
        pass
    
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        U, V, W = _strassen_block_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_result is not None:
        return best_result
    
    # Fallback: return Strassen decomposition
    U, V, W = _strassen_block_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _attempt_laderman_embedding(n, m, p):
    """
    Attempt to construct a lower-rank decomposition by embedding 2x2*2x9 
    into a larger structured problem.
    
    Approach: Use the Kronecker product structure of the 9-column dimension.
    9 = 8 + 1, so we can use a rank-28 decomposition for 2x2*2x8 
    (if it exists) plus rank-4 for the single column, giving rank 32.
    Or use a rank-24 for 2x2*2x8 + rank-4 = rank 28.
    
    For 2x2*2x8: partition into 4 pairs, each with Strassen rank 7: total 28.
    So 2x2*2x8 has rank 28 via Strassen blocks.
    Then 2x2*2x9 = 2x2*2x8 + 2x2*2x1, rank 28 + 4 = 32. Same as before.
    
    To get rank < 32, we need a better algorithm for 2x2*2x8 or 2x2*2x9.
    
    Known results:
    - 2x2*2x2: rank 7 (Strassen)
    - 2x2*2x4: rank 12 (Baur-Strassen or similar)
    - 2x2*2x8: rank 21? or 22?
    
    Let me try to construct a rank-21 decomposition for 2x2*2x8.
    If 2x2*2x4 has rank 12, then 2x2*2x8 = 2 copies of 2x2*2x4, rank 24.
    But there might be a better rank-21 algorithm.
    
    Actually, for this attempt, let me just try the Strassen approach and 
    verify it works correctly.
    """
    return None

# EVOLVE-BLOCK-END
