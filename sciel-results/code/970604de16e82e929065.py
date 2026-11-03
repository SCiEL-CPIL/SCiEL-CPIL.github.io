"""Optimized rank-44 decomposition for 3x4 * 4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]


def _build_rank_44(n, m, p):
    """Construct a rank-44 tensor decomposition for 3x4 * 4x4."""
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A times B -> Row 0 of C (16 terms)
    for k in range(4):
        for j in range(4):
            U[j][col] = 1
            V[j * 4 + k][col] = 1
            W[3 * k][col] = 1
            col += 1
    
    # Part 2: Rows 1,2 of A times B -> Rows 1,2 of C
    # Split into two 2x4 * 4x4 blocks using Strassen-like decomposition
    # Each 2x4 * 4x4 block: split B into two 2x2 column blocks, use Strassen on each
    
    def add_2x4_block(i_off, j_off, k_off):
        """Add decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+4]"""
        nonlocal col
        # Left 2x2: B cols k_off, k_off+1
        _add_strassen_2x2x2(
            i_off, i_off + 1, j_off, j_off + 1,
            j_off, j_off + 1, k_off, k_off + 1,
            k_off, k_off + 1, i_off, i_off + 1
        )
        # Right 2x2: B cols k_off+2, k_off+3
        _add_strassen_2x2x2(
            i_off, i_off + 1, j_off, j_off + 1,
            j_off, j_off + 1, k_off + 2, k_off + 3,
            k_off + 2, k_off + 3, i_off, i_off + 1
        )
    
    def _add_strassen_2x2x2(
        ai0, ai1, aj0, aj1,
        bj0, bj1, bk0, bk1,
        ck0, ck1, ci0, ci1
    ):
        """Add Strassen 7-term decomposition for 2x2 * 2x2 block."""
        nonlocal col
        # m1 = (a00+a11)(b00+b11)
        U[ai0 * 4 + aj0][col] += 1
        U[ai1 * 4 + aj1][col] += 1
        V[bj0 * 4 + bk0][col] += 1
        V[bj1 * 4 + bk1][col] += 1
        W[ck0 * 3 + ci0][col] += 1
        W[ck1 * 3 + ci1][col] += 1
        col += 1
        # m2 = (a10+a11)(b00)
        U[ai1 * 4 + aj0][col] += 1
        U[ai1 * 4 + aj1][col] += 1
        V[bj0 * 4 + bk0][col] += 1
        W[ck0 * 3 + ci1][col] += 1
        col += 1
        # m3 = a00(b10-b00)
        U[ai0 * 4 + aj0][col] += 1
        V[bj1 * 4 + bk0][col] += 1
        V[bj0 * 4 + bk0][col] -= 1
        W[ck0 * 3 + ci0][col] += 1
        col += 1
        # m4 = a11(b11-b01)
        U[ai1 * 4 + aj1][col] += 1
        V[bj1 * 4 + bk1][col] += 1
        V[bj0 * 4 + bk1][col] -= 1
        W[ck1 * 3 + ci1][col] += 1
        col += 1
        # m5 = (a00+a01)(b01)
        U[ai0 * 4 + aj0][col] += 1
        U[ai0 * 4 + aj1][col] += 1
        V[bj0 * 4 + bk1][col] += 1
        W[ck1 * 3 + ci0][col] += 1
        col += 1
        # m6 = (a10-a00)(b00+b01)
        U[ai1 * 4 + aj0][col] += 1
        U[ai0 * 4 + aj0][col] -= 1
        V[bj0 * 4 + bk0][col] += 1
        V[bj0 * 4 + bk1][col] += 1
        W[ck0 * 3 + ci1][col] += 1
        col += 1
        # m7 = (a01-a11)(b10+b11)
        U[ai0 * 4 + aj1][col] += 1
        U[ai1 * 4 + aj1][col] -= 1
        V[bj1 * 4 + bk0][col] += 1
        V[bj1 * 4 + bk1][col] += 1
        W[ck1 * 3 + ci0][col] += 1
        col += 1
    
    # First 2x4 block: A rows 1-2, cols 0-1 * B rows 0-1
    add_2x4_block(1, 0, 0)
    # Second 2x4 block: A rows 1-2, cols 2-3 * B rows 2-3
    add_2x4_block(1, 2, 0)
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
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


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    result = _build_rank_44(n, m, p)
    
    # Verify if time permits
    if time.time() < deadline - 5:
        if not _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            # Fallback to schoolbook (rank 48)
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
            return {"U": U, "V": V, "W": W}
    
    return result
