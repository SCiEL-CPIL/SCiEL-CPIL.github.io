"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2x2_block(a, b, c, d):
    """
    For a 2x2x2 tensor with U = diag(a,b,c,d) (4x4),
    V = [[a,b],[c,d]] (2x2), W = [[a,b],[c,d]] (2x2),
    return the 7 Strassen terms as (u, v, w) triples.
    """
    terms = []
    # M1 = (a+c)*(b+d)
    terms.append((a + c, b + d, a * b + b * d + c * b + c * d))
    # M2 = (a+d)*(b+c)
    terms.append((a + d, b + c, a * b + a * c + d * b + d * c))
    # M3 = (c+d)*(b-a)
    terms.append((c + d, b - a, c * b - c * a + d * b - d * a))
    # M4 = (b-c)*(d-a)
    terms.append((b - c, d - a, b * d - b * a - c * d + c * a))
    # M5 = a*(d-c)
    terms.append((a, d - c, a * d - a * c))
    # M6 = b*(a-c)
    terms.append((b, a - c, a * b - b * c))
    # M7 = d*(c-b)
    terms.append((d, c - b, d * c - d * b))
    return terms


def _block_3x2x7(n, m, p):
    """
    Decompose 3x2x7 matrix multiplication into rank-39 blocks.
    Splits the 2x2 part into four 2x2x2 Strassen blocks,
    plus additional terms to handle the 7-column structure.
    """
    # This is a placeholder for the actual construction.
    # The exact construction is complex, so we return None if we can't verify.
    # For now, we'll attempt a known construction if available.
    # Since the exact coefficients are not trivially derivable here without
    # extensive search, we return None to indicate this path is not used
    # unless explicitly validated.
    return None


def _combine_blocks(n, m, p):
    """
    Attempt to combine 3x2x7 and 3x1x7 blocks into a full 3x5x7 decomposition.
    """
    block1 = _block_3x2x7(n, 2, p)
    if block1 is None:
        return None
    # block2 for 3x1x7 would be rank 21 (schoolbook for that part)
    # Total rank = 39 + 21 = 60? That seems too low.
    # Actually, the known best is 99. Let's not guess.
    return None


def _validate(U, V, W, n, m, p, tol=1e-10):
    """
    Validate that U, V, W correctly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = 1 if we're multiplying (i,j) by (j,k).
    The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}
    Wait, the trace convention is W[k*n+i, r].
    The matrix product C = A * B has C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor T_{i,j,k} = 1 for all i,j,k.
    The decomposition must satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != rank or len(V[r]) != rank or len(W[r]) != rank:
            return False
    
    # Verify the tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_reduction(n, m, p, target_rank, time_limit=10):
    """
    Attempt to find a lower-rank decomposition via local search.
    This is a placeholder for a more sophisticated search.
    """
    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with the schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Validate the schoolbook decomposition
    if not _validate(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed validation")
    
    # Attempt to find a lower-rank decomposition
    # Try the known 99-rank construction if available
    try:
        combined = _combine_blocks(n, m, p)
        if combined is not None:
            U2, V2, W2 = combined
            if _validate(U2, V2, W2, n, m, p):
                rank2 = len(U2[0])
                if rank2 < best_rank:
                    best_rank = rank2
                    best_result = {"U": U2, "V": V2, "W": W2}
    except Exception:
        pass
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > 140:
        return best_result
    
    # Attempt local search for further reduction
    try:
        reduced = _try_rank_reduction(n, m, p, best_rank - 1, time_limit=5)
        if reduced is not None:
            U3, V3, W3 = reduced
            if _validate(U3, V3, W3, n, m, p):
                rank3 = len(U3[0])
                if rank3 < best_rank:
                    best_rank = rank3
                    best_result = {"U": U3, "V": V3, "W": W3}
    except Exception:
        pass
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > 145:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
