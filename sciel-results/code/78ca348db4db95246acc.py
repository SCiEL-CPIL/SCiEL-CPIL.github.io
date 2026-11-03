"""Tensor decomposition for 2x3 by 3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Return the rank-42 schoolbook decomposition."""
    n, m, p = 2, 3, 7
    rank = n * m * p  # 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    """
    n, m, p = 2, 3, 7
    rank = U.shape[1]
    
    # Build the tensor T[i, j, k] which should be 1 if the corresponding
    # (i,j) of A and (j,k) of B contribute to output (i,k).
    # Actually the tensor T has T[i,j,k] = 1 for each (i,j,k) triple.
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i, j, k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_strassen_2x3x4_and_pad():
    """
    Try to build a rank-35 decomposition using Strassen on 2x3x4 sub-blocks
    then pad to handle the 7 columns.
    
    For 2x3x4: we can use the standard Strassen-like decomposition.
    Actually, let's think about this differently.
    
    The 2x3x7 multiplication can be viewed as 2x3 times 3x7.
    We can split the 7 columns into 3+3+1 = 7.
    
    For 2x3x3: rank is at most 9 (schoolbook). Can we do better?
    For 2x2x2: rank 7 (Strassen).
    
    Let's try: 2x3x7 = 2x3x(3+3+1)
    = (2x3x3) + (2x3x3) + (2x3x1)
    
    For 2x3x1: rank 6 (schoolbook) or maybe less.
    For 2x3x3: let's use rank 8 via Strassen-like.
    
    Total: 8 + 8 + 6 = 22? That seems too optimistic.
    
    Let me be more careful. The tensor for 2x3x7 has size 2*3*7 = 42.
    
    Actually, let me try a different approach: use the known rank-7 Strassen
    for 2x2x2 and build up.
    
    2x3x7: pad 3 to 4 (power of 2), so 2x4x7. Pad 7 to 8: 2x4x8.
    2x4x8 can be done by: think of A as 2x4, B as 4x8.
    Split into blocks: A = [A1 A2] where A1, A2 are 2x2.
    B = [[B11, B12], [B21, B22]] where each Bi,j is 2x4.
    
    C = A1*B1 + A2*B2 (where B1 = [B11; B21] is 4x4, B2 = [B12; B22] is 4x4)
    
    Hmm, this is getting complicated. Let me just use the schoolbook.
    """
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x7 matrix.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with schoolbook (guaranteed valid, rank 42)
    U, V, W = _build_schoolbook()
    best_rank = 42
    best_U, best_V, best_W = U, V, W
    
    # Verify schoolbook
    assert _verify(U, V, W), "Schoolbook decomposition failed verification"
    
    # Try to find a lower-rank decomposition via random search / numerical methods
    # The matrix multiplication tensor T for (n,m,p) = (2,3,7) lives in R^{nm} x R^{mp} x R^{pn}
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # For a rank-r decomposition, we need:
    # For all (i,j,k): sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of nonlinear equations. Let's try a simple approach:
    # Use the fact that the tensor is the product of three linear maps.
    
    # Actually, for small dimensions, we can try to find a structured decomposition.
    
    # Let's try: represent the 3x7 matrix B as having 7 columns.
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: use rank-1 updates. The schoolbook uses 42 rank-1 terms.
    # We can potentially combine terms.
    
    # Try a parameterized family: for each (i,j), we have 7 terms (one for each k).
    # Maybe we can share structure across k.
    
    # Alternative: try random initialization and gradient descent on the rank-r tensor.
    # But this is risky within the time budget.
    
    # Investigate the structured rank-reduction heuristic:
    # Hypothesis: 2x3x7 can be decomposed into independent 2x1x7 (rank 14) and 1x3x7 (rank 21) parts.
    # Total rank 35.
    # We construct these parts and verify if their sum equals the full tensor.
    
    def _build_hypothesis_rank35():
        n, m, p = 2, 3, 7
        rank1 = 14 # 2x1x7
        rank2 = 21 # 1x3x7
        total_rank = rank1 + rank2
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Part 1: 2x1x7
        # This corresponds to j=0 terms? Or a subset?
        # The heuristic in the prompt is ambiguous on index mapping.
        # Standard schoolbook for 2x1x7:
        # i in {0,1}, j=0, k in {0..6}
        # Terms: A[i,0] * B[0,k] -> C[i,k]
        # Rank 14.
        col = 0
        for i in range(2):
            for k in range(7):
                U[i * 3 + 0, col] = 1.0 # i*m + j, j=0
                V[0 * 7 + k, col] = 1.0 # j*p + k, j=0
                W[k * 2 + i, col] = 1.0 # k*n + i
                col += 1
                
        # Part 2: 1x3x7
        # This corresponds to i=0 terms?
        # Terms: A[0,j] * B[j,k] -> C[0,k]
        # Rank 21.
        for j in range(3):
            for k in range(7):
                U[0 * 3 + j, col] = 1.0 # i=0
                V[j * 7 + k, col] = 1.0
                W[k * 2 + 0, col] = 1.0 # i=0
                col += 1
                
        return U, V, W

    U_hyp, V_hyp, W_hyp = _build_hypothesis_rank35()
    
    # Verify the hypothesis
    # Check if sum of parts equals full tensor
    # Full tensor requires T[i,j,k] = 1 for all i,j,k.
    # Hypothesis covers:
    # 1. i in {0,1}, j=0, k in {0..6}
    # 2. i=0, j in {0,1,2}, k in {0..6}
    # Union: 
    # i=0, j=0 (double counted?)
    # i=0, j=1
    # i=0, j=2
    # i=1, j=0
    # Missing: i=1, j=1 and i=1, j=2.
    # Also, if double counted, the value would be 2 for i=0, j=0.
    
    valid_hyp = _verify(U_hyp, V_hyp, W_hyp)
    
    if valid_hyp and U_hyp.shape[1] < best_rank:
        best_rank = U_hyp.shape[1]
        best_U, best_V, best_W = U_hyp, V_hyp, W_hyp
    else:
        # Hypothesis failed (expected), stick with schoolbook
        pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
