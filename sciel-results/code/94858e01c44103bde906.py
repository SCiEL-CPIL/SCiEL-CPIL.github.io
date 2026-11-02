"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as 2D lists for schoolbook decomposition."""
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
    return U, V, W


def _verify_arrays(U, V, W, n, m, p, tol=1e-9):
    """Check tensor decomposition validity using numpy arrays."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_rank48(n, m, p):
    """Construct a rank-48 decomposition using a structured approach.
    
    For 3x3 * 3x8, partition B's columns into 4 pairs.
    For each pair, use 12 terms with shared U structure.
    """
    rank = 48
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    
    for g, (k1, k2) in enumerate(k_pairs):
        base = g * 12
        for i in range(n):
            for j in range(m):
                c = base + i * m + j
                U[i * m + j, c] = 1.0
                V[j * p + k1, c] = 1.0
                W[k1 * n + i, c] = 1.0
                # Add second term for k2 in same column? No, need separate column
                # Use next column for k2
                c2 = base + 9 + i * m + j
                U[i * m + j, c2] = 1.0
                V[j * p + k2, c2] = 1.0
                W[k2 * n + i, c2] = 1.0
    
    # Verify - this is actually rank 72 (24 terms per pair * 4 pairs = 96, but we only have 48)
    # The above uses 24 columns per pair, which is wrong. Let me check.
    # base goes from 0 to 36 in steps of 12, so 4*12=48 columns total.
    # For each pair, we use columns base..base+8 for k1 and base+9..base+17 for k2.
    # But base+17 > base+11 when base=0 (17>11). Bug!
    
    # Fix: use 6 columns per k-value, 12 per pair
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # This construction is flawed. Return None.
    return None


def _construct_rank48(n, m, p):
    """Construct a rank-48 decomposition using block structure.
    
    Partition B's 8 columns into 4 pairs: (0,1), (2,3), (4,5), (6,7).
    For each pair, use 12 rank-1 terms.
    
    For a pair (k1, k2), the 12 terms should satisfy:
      For j in {0,1,2}, i in {0,1,2}:
        sum_t U[i,j,t] * V[j,k1,t] * W[k1,i,t] = 1
        sum_t U[i,j,t] * V[j,k2,t] * W[k2,i,t] = 1
    """
    rank = 48
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    
    for g, (k1, k2) in enumerate(k_pairs):
        base = g * 12
        # For this pair, we need 12 terms
        # Use a structured pattern:
        # Terms 0-2: handle i=0 for both k1,k2
        # Terms 3-5: handle i=1 for both k1,k2  
        # Terms 6-8: handle i=2 for both k1,k2
        # Terms 9-11: correction terms
        
        for i in range(n):
            for j in range(m):
                c = base + i * m + j
                U[i * m + j, c] = 1.0
                V[j * p + k1, c] = 1.0
                W[k1 * n + i, c] = 1.0
                # Also set for k2? No, that would double-count
                # We need separate terms for k1 and k2, but share U
    
    # The above doesn't work because we need separate V,W for k1 and k2
    # Let me use a different approach: 6 terms per k-value, 12 per pair
    # But share U across the pair
    
    # Reset
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for g, (k1, k2) in enumerate(k_pairs):
        base = g * 12
        for i in range(n):
            for j in range(m):
                # Term for k1
                c1 = base + i * m + j
                U[i * m + j, c1] = 1.0
                V[j * p + k1, c1] = 1.0
                W[k1 * n + i, c1] = 1.0
                # Term for k2 (shares U column? No, different column)
                c2 = base + 9 + i * m + j
                U[i * m + j, c2] = 1.0
                V[j * p + k2, c2] = 1.0
                W[k2 * n + i, c2] = 1.0
    
    # This is just schoolbook with rank 72, not 48. The above uses 24 terms per pair = 96 total.
    # I made an error. Let me just return None and use a simpler fallback.
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s safety margin
    
    # Get schoolbook decomposition (rank 72)
    U, V, W = _schoolbook(n, m, p)
    
    # Verify it's correct
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if not _verify_arrays(Ua, Va, Wa, n, m, p):
        # Should not happen, but fallback
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
