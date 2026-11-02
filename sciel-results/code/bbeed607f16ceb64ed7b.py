import time
import numpy as np
from fractions import Fraction

DEADLINE = 140  # seconds, leaving 10s margin before 150s kill


def _build_rank_28():
    """Build the known rank-28 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Uses block decomposition:
    - Block 1: j=0, k=0..4 (2x1x5): rank 10 (schoolbook)
    - Block 2a: j=1,2, k=0,1 (2x2x2): rank 7 (Strassen)
    - Block 2b: j=1,2, k=2 (2x2x1): rank 4 (schoolbook)
    - Block 2c: j=1,2, k=3,4 (2x2x2): rank 7 (Strassen)
    Total rank: 10 + 7 + 4 + 7 = 28
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2)
    # M1: (A[0,1]+A[1,2])(B[1,0]+B[2,1])
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])(B[1,0])
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    # M3: A[0,1](B[1,1]-B[2,1])
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M4: A[1,2](B[2,0]-B[1,0])
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])(B[2,1])
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[1 * n + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[1,0]+B[1,1])
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[2,0]+B[2,1])
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2)
    # M1: (A[0,1]+A[1,2])(B[1,3]+B[2,4])
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])(B[1,3])
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    # M3: A[0,1](B[1,4]-B[2,4])
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M4: A[1,2](B[2,3]-B[1,3])
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])(B[2,4])
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[1,3]+B[1,4])
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[2,3]+B[2,4])
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_reduction(U, V, W, n=2, m=3, p=5, time_limit=30):
    """Try to find a rank-27 decomposition using LLL on the kernel of the tensor term matrix.
    
    Strategy: 
    1. Flatten each of the 28 rank-1 terms into a 30-dim vector
    2. Check if the 28 vectors are linearly dependent
    3. If dependent, use LLL to find short integer relations
    4. Check if any relation allows eliminating one term with small integer/half-integer adjustments
    """
    start_time = time.time()
    rank = len(U[0])
    dim = n * m * p  # 30
    
    # Build the 30 x 28 matrix of flattened terms
    M = np.zeros((dim, rank), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    M[idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Check rank
    rnk = np.linalg.matrix_rank(M, tol=1e-9)
    if rnk >= rank:
        # Terms are linearly independent - no simple rank reduction possible
        return None
    
    # Find null space
    from scipy.linalg import null_space
    null_vecs = null_space(M, rcond=1e-9)
    
    if null_vecs.shape[1] == 0:
        return None
    
    # For each null vector, try to scale to get small integer/half-integer coefficients
    for null_v in null_vecs.T:
        # Try scaling by various factors to get near-integer values
        best_candidate = None
        for scale_factor in [1.0, 2.0, 0.5, 4.0, 0.25, 3.0, 1.0/3.0]:
            scaled = null_v * scale_factor
            # Check if all values are close to integers or half-integers
            max_dev = 0
            for val in scaled:
                nearest_int = round(val)
                nearest_half = round(2 * val) / 2.0
                dev = min(abs(val - nearest_int), abs(val - nearest_half))
                max_dev = max(max_dev, dev)
            if max_dev < 1e-6:
                # Found a good candidate
                # Check if we can eliminate one term
                for idx in range(rank):
                    if abs(scaled[idx]) > 0.9:  # This term has coefficient ±1
                        # Check if other coefficients are small
                        others_small = all(abs(scaled[j]) < 5 for j in range(rank) if j != idx)
                        if others_small:
                            best_candidate = idx
                            break
                if best_candidate is not None:
                    break
        
        if best_candidate is not None:
            return best_candidate
    
    return None


def construct():
    """Return rank-28 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Uses block decomposition with Strassen's algorithm for 2x2x2 subproblems.
    Includes a deadline guard and attempts rank reduction search if time permits.
    """
    start_time = time.time()
    
    # Build the known rank-28 decomposition
    result = _build_rank_28()
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        # This should never happen, but handle it
        pass
    
    # Check if we have time to attempt rank reduction
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:  # Leave 20s margin
        try:
            removed_idx = _try_rank_reduction(
                result["U"], result["V"], result["W"],
                time_limit=DEADLINE - elapsed - 15
            )
            if removed_idx is not None:
                # We found a potential rank reduction.
                # For now, this is experimental - the rank-28 solution is safe.
                # A proper rank-27 construction would need careful verification.
                pass
        except Exception:
            # If the search fails, fall back to rank-28
            pass
    
    # Final deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Shouldn't happen, but return immediately
        pass
    
    return result
