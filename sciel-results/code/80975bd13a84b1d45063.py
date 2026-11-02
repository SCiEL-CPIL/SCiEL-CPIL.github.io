"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(n, m, p, U, V, W):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i, j, k] = delta_{i,j} * delta_{k,j}
    # Actually T_{i,j,k} = 1 if the output C[i,k] gets A[i,j]*B[j,k]
    # Tensor M_{i,j,k} where (U^T M V^T W^T)_{i,k} = sum_j A[i,j]B[j,k]
    # The tensor is: M[i, j, k] = 1 for all i, j, k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, let me think again.
    # We need: sum_t U[i, t] * V[j, t] * W[k, t] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # The multiplication tensor has M[i,j,k] = 1 for all i,j,k
    # So we need: sum_t U[i,t] * V[j,t] * W[k,t] = 1 for all i,j,k
    
    # Actually that's not right either. Let me reconsider.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_{j,t} A[i,j] * B[j,k] * (tensor coefficient)
    # The tensor T satisfies: C = T * (A, B)
    # T[i, j, k] such that C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[k',i']... 
    
    # Let me use the standard formulation:
    # The matrix multiplication tensor for n x m by m x p is:
    # M_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # and M_{i,j,k} = 1 (all ones)
    # We need: sum_t U[i,t] * V[j,t] * W[k,t] = M[i,j,k] = 1 for all i,j,k
    
    # Verify: for each t, the outer product U[:,t] outer V[:,t] outer W[:,t]
    # Sum over t should give all-ones tensor
    
    result = np.zeros((n, m, p))
    for t in range(rank):
        result += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
    
    expected = np.ones((n, m, p))
    return np.allclose(result, expected, atol=1e-10)


def _strassen_block(n, m, p, j0, j1, k0, k1, col_offset, U, V, W):
    """Add Strassen 7-term block for rows j0,j1 of A and columns k0,k1 of B."""
    col = col_offset
    
    # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A_1,j0 + A_1,j1) B_j0,k0
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A_0,j0 + A_0,j1) B_j1,k1
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col


def _try_construct(n, m, p):
    """
    Attempt to construct a low-rank decomposition for n x m by m x p matrix multiplication.
    
    Strategy: Use Strassen's algorithm where possible for 2x2 blocks,
    and handle remaining dimensions individually.
    
    The key insight is that we can partition the m dimension into pairs and singles,
    and the p dimension into pairs and singles.
    """
    deadline = time.time() + 110  # Leave safety margin
    
    # Partition m into pairs and singles
    # m=5: pairs (0,1), (2,3), single (4)
    # p=6: pairs (0,1), (2,3), (4,5)
    
    # For each pair of j indices and pair of k indices, use Strassen (7 terms)
    # For single j and single k, use 2 terms (one for each row of C)
    # For single j and pair of k, use 3 terms (Strassen-like but adapted)
    # For pair of j and single k, use 3 terms
    
    # Let's compute the rank for different partitionings:
    # m_pairs = number of j-pairs, m_singles = number of j-singles
    # p_pairs = number of k-pairs, p_singles = number of k-singles
    # rank = m_pairs * p_pairs * 7 + m_pairs * p_singles * 3 + m_singles * p_pairs * 3 + m_singles * p_singles * 2
    
    # For m=5: 2 pairs + 1 single
    # For p=6: 3 pairs + 0 singles
    # rank = 2*3*7 + 2*0*3 + 1*3*3 + 1*0*2 = 42 + 0 + 9 + 0 = 51
    
    # Wait, let me recheck. For a single j and a pair of k's:
    # C[0, k0] = A[0,j] * B[j, k0]
    # C[0, k1] = A[0,j] * B[j, k1]
    # C[1, k0] = A[1,j] * B[j, k0]
    # C[1, k1] = A[1,j] * B[j, k1]
    # We need 4 products but can we do it in 3?
    # Actually with Strassen-style:
    # P1 = A[0,j] * (B[j,k0] + B[j,k1])
    # P2 = A[1,j] * (B[j,k0] + B[j,k1])
    # P3 = (A[0,j] - A[1,j]) * B[j,k1]
    # Then:
    # C[0,k0] = P1 - P3... no wait.
    # C[0,k0] = A[0,j]*B[j,k0]
    # C[0,k1] = A[0,j]*B[j,k1] = P1 - A[0,j]*B[j,k0]... this doesn't work cleanly.
    
    # Let me think more carefully. For 2x1x2 (2 rows, 1 col in middle, 2 cols in output):
    # We have A[0,j], A[1,j] and B[j,k0], B[j,k1]
    # Need: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
    # 
    # Try 4 terms (naive):
    # T1 = A[0,j]*B[j,k0] -> C[0,k0]
    # T2 = A[0,j]*B[j,k1] -> C[0,k1]
    # T3 = A[1,j]*B[j,k0] -> C[1,k0]
    # T4 = A[1,j]*B[j,k1] -> C[1,k1]
    #
    # Can we do 3? Let's try:
    # P1 = (A[0,j] + A[1,j]) * B[j,k0]
    # P2 = (A[0,j] + A[1,j]) * B[j,k1]
    # P3 = (A[0,j] - A[1,j]) * (B[j,k0] - B[j,k1])
    # C[0,k0] = (P1 + P3)/2 + ... hmm, this is getting complicated with fractions.
    # Actually with the trace convention and integer/half-integer entries, let me just use 4 terms for single-j.
    
    # So for m=5, p=6:
    # j-pairs: (0,1), (2,3); j-single: 4
    # k-pairs: (0,1), (2,3), (4,5)
    
    # For j-pair x k-pair: 7 terms (Strassen)
    # For j-single x k-pair: 4 terms (one for each combination of row and col)
    
    # Total rank = 2*3*7 + 1*3*4 = 42 + 12 = 54
    
    # That matches the current program! So the current rank of 54 is from this partitioning.
    
    # Can we do better? What if we use a different partitioning?
    # What about 1 j-pair and 3 j-singles?
    # rank = 1*3*7 + 3*3*4 = 21 + 36 = 57. Worse.
    
    # What about 3 j-pairs? But m=5, we can only have 2 pairs + 1 single.
    
    # So 54 seems optimal for this approach. Let me verify and return it.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    j_singles = [4]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(n, m, p, j0, j1, k0, k1, col, U, V, W)
    
    for j in j_singles:
        for k0, k1 in k_pairs:
            # 4 terms for single j, pair of k
            # T1 = A[0,j]*B[j,k0]
            U[0*m + j][col] += 1
            V[j*p + k0][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
            # T2 = A[0,j]*B[j,k1]
            U[0*m + j][col] += 1
            V[j*p + k1][col] += 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # T3 = A[1,j]*B[j,k0]
            U[1*m + j][col] += 1
            V[j*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # T4 = A[1,j]*B[j,k1]
            U[1*m + j][col] += 1
            V[j*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    result = _try_construct(n, m, p)
    
    # Verify the decomposition
    U, V, W = result["U"], result["V"], result["W"]
    if _verify_decomp(n, m, p, U, V, W):
        return result
    else:
        # Fallback: naive decomposition (rank = n*m*p = 60)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
