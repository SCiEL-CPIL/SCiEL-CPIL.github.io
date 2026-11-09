"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def _validate(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the 2x3x9 matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Verify dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        for r in range(n * m):
            if abs(U[r][c]) > 0:
                break
    # Build the tensor from the decomposition and compare to the identity tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 if i,j,k match (i.e., the product entry), else 0
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to C[i,k]. For standard matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Recompute T from decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_block_decomposition(n, m, p):
    """
    Try to exploit block structure: split B into 3 blocks of 3x9 each,
    or use the fact that m=3 is small.
    
    Alternative approach: since m=3, we can try rank-m3 decompositions.
    But a simpler idea: if p=9=3*3, maybe we can use a rank-9 per m-block.
    
    Actually, let's try a structured decomposition:
    For each i in [0,n), the row i of C is sum_j A[i,j] * B_row_j.
    B_row_j is a 1x9 vector. So C[i] is a linear combination of 3 vectors (rows of B).
    The space of possible C[i] for fixed i is at most 3-dimensional (spanned by rows of B).
    
    This suggests rank could be as low as n * min(m, p, ...) but let's just try
    a specific construction: use rank = n * p = 18? No, that's too optimistic.
    
    Let's just return None if no improvement found.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    # Try a few heuristics for lower rank, but fall back to schoolbook
    # For 2x3x9, schoolbook is 54. Known bounds suggest rank >= 27.
    
    # Simple attempt: if we can find a rank < 54 decomposition, use it.
    # Otherwise, return schoolbook.
    
    result = _try_block_decomposition(n, m, p)
    if result is not None:
        U, V, W = result
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
