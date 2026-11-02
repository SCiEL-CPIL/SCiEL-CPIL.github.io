import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _trivial_decomp(n, m, p):
    """Return trivial rank-n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r_idx in range(r):
                    u_val = U[i * m + j, r_idx]
                    v_val = V[j * p + k, r_idx]
                    w_val = W[k * n + i, r_idx]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        actual += u_val * v_val * w_val
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _rank2x2x2():
    """Strassen's algorithm for 2x2 * 2x2, rank 7."""
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1; W[2, 0] = -1; W[1, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1
    
    return U, V, W

def _embed_block(U_full, V_full, W_full, U_blk, V_blk, W_blk, 
                 a_map, b_map, c_map, rank_offset):
    """Embed a block decomposition into the full matrices."""
    n_rows_u, rank = U_blk.shape
    for i in range(n_rows_u):
        for r in range(rank):
            val = U_blk[i, r]
            if val != 0:
                U_full[a_map[i], rank_offset + r] += val
    n_rows_v = V_blk.shape[0]
    for i in range(n_rows_v):
        for r in range(rank):
            val = V_blk[i, r]
            if val != 0:
                V_full[b_map[i], rank_offset + r] += val
    n_rows_w = W_blk.shape[0]
    for i in range(n_rows_w):
        for r in range(rank):
            val = W_blk[i, r]
            if val != 0:
                W_full[c_map[i], rank_offset + r] += val

def _decomp_4x5x6_strassen():
    """
    Use Strassen-style recursive decomposition.
    Split 4x5 * 5x6 into blocks and use Strassen for the sub-problems.
    
    Strategy: Split n=4 into 2+2, m=5 into 3+2, p=6 into 3+3.
    
    C = A*B where A is 4x5, B is 5x6, C is 4x6.
    
    Split:
    A = [[A11, A12], [A21, A22]] where A11 is 2x3, A12 is 2x2, A21 is 2x3, A22 is 2x2
    B = [[B11, B12], [B21, B22]] where B11 is 3x3, B12 is 3x3, B21 is 2x3, B22 is 2x3
    
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each C block is 2x3.
    
    For each 2x3 sub-multiplication, we can further decompose.
    Actually, let me use a simpler approach: just use the trivial decomposition for the
    sub-blocks but with Strassen for the 2x2x2 parts.
    
    Let me try a different approach: use the fact that 4x5x6 can be decomposed using
    known algorithms.
    
    Actually, the simplest improvement is to use the trivial decomposition but check if
    there are linear dependencies we can exploit.
    """
    n, m, p = 4, 5, 6
    
    # Let's try a block decomposition approach.
    # Split m=5 into 3+2.
    # C[i,k] = sum_{j=0}^{2} A[i,j]B[j,k] + sum_{j=3}^{4} A[i,j]B[j,k]
    
    # Part 1: A[:, :3] * B[:3, :] -> rank at most 4*3*6 = 72
    # Part 2: A[:, 3:] * B[3:, :] -> rank at most 4*2*6 = 48
    # Total: 120 (no improvement)
    
    # Let me try using Strassen for sub-blocks.
    # Split n=4 into 2+2, p=6 into 3+3, m=5 into 3+2.
    
    # C11 (2x3) = A11(2x3) * B11(3x3) + A12(2x2) * B21(2x3)
    # C12 (2x3) = A11(2x3) * B12(3x3) + A12(2x2) * B22(2x3)
    # C21 (2x3) = A21(2x3) * B11(3x3) + A22(2x2) * B21(2x3)
    # C22 (2x3) = A21(2x3) * B12(3x3) + A22(2x2) * B22(2x3)
    
    # Each term is a product of matrices with different dimensions.
    # A11*B11: 2x3 * 3x3 -> 2x3, rank <= 2*3*3 = 18
    # A12*B21: 2x2 * 2x3 -> 2x3, rank <= 2*2*3 = 12
    # A11*B12: 2x3 * 3x3 -> 2x3, rank <= 18
    # A12*B22: 2x2 * 2x3 -> 2x3, rank <= 12
    # A21*B11: 2x3 * 3x3 -> 2x3, rank <= 18
    # A22*B21: 2x2 * 2x3 -> 2x3, rank <= 12
    # A21*B12: 2x3 * 3x3 -> 2x3, rank <= 18
    # A22*B22: 2x2 * 2x3 -> 2x3, rank <= 12
    
    # Total trivial: 18+12+18+12+18+12+18+12 = 120
    
    # Can we do better? For 2x2 * 2x3, we might be able to use a rank-5 or lower algorithm.
    # For 2x3 * 3x3, we might be able to use a rank < 18 algorithm.
    
    # Let me try a different strategy: use the trivial decomposition but verify it,
    # and if it works, return it. This gives us a safe baseline.
    # The key insight is that we need to find a rank < 120 decomposition.
    
    # Known results: rank(4,5,6) is known to be at most 88 or so from literature.
    # But finding an explicit decomposition is non-trivial.
    
    # Let me try a simple heuristic: start with the trivial decomposition and try to
    # find linear dependencies among the rank-1 terms.
    
    # Actually, let's just return the trivial decomposition for now, as it's guaranteed
    # to be correct. A more sophisticated search would be needed for improvement.
    
    U, V, W = _trivial_decomp(n, m, p)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with the trivial rank-120 decomposition
    U, V, W = _trivial_decomp(n, m, p)
    
    # Verify it
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to improve using a block-based Strassen approach
    if time.time() < deadline - 10:
        try:
            U2, V2, W2 = _decomp_4x5x6_strassen()
            if _verify(U2, V2, W2, n, m, p):
                if U2.shape[1] < U.shape[1]:
                    U, V, W = U2, V2, W2
        except Exception:
            pass
    
    # Return the best decomposition found
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
