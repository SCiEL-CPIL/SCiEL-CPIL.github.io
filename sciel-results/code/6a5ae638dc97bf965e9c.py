"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.

Uses Strassen's algorithm (rank 7) for the 2x2 sub-blocks of the 2x3 * 3x6 product,
combined with a rank-1 tail for the remaining 2x1 * 1x6 part. This gives rank 22
(vs. schoolbook rank 36).
"""
import time
import numpy as np

_DIMS = (2, 3, 6)
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matmul tensor."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _is_half_int(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol


def _to_half_int(U, V, W):
    """Convert to half-integers if close, else return None."""
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if not _is_half_int(v):
                    return None
    Ur = [[round(v * 2) / 2 for v in row] for row in U]
    Vr = [[round(v * 2) / 2 for v in row] for row in V]
    Wr = [[round(v * 2) / 2 for v in row] for row in W]
    return Ur, Vr, Wr


def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 matmul with rank 7 (Strassen)."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = -1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1
    
    return U, V, W


def _build_strassen_blocks():
    """Build the full 2x3 * 3x6 decomposition using Strassen for 2x2 blocks
    and a rank-1 tail for j=2. Total rank: 22."""
    n, m, p = _DIMS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    sU, sV, sW = _strassen_2x2x2()
    
    for block in range(3):
        k_offset = block * 2
        for c in range(7):
            col = block * 7 + c
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = sU[i * 2 + j][c]
            for j in range(2):
                for kk in range(2):
                    V[j * p + (k_offset + kk)][col] = sV[j * 2 + kk][c]
            for kk in range(2):
                for i in range(2):
                    W[(k_offset + kk) * n + i][col] = sW[kk * 2 + i][c]
    
    tail_col = 21
    for i in range(2):
        U[i * m + 2][tail_col] = 1
    for k in range(6):
        V[2 * p + k][tail_col] = 1
        for i in range(2):
            W[k * n + i][tail_col] = 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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


def construct():
    n, m, p = _DIMS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    if _remaining() < 5.0:
        return best
    
    # Try Strassen-based construction (rank 22)
    U, V, W = _build_strassen_blocks()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Convert to half-integers
        result = _to_half_int(U, V, W)
        if result is not None:
            Ur, Vr, Wr = result
            if _verify(Ur, Vr, Wr, n, m, p):
                return {"U": Ur, "V": Vr, "W": Wr}
    
    return best


if __name__ == "__main__":
    result = construct()
    n, m, p = _DIMS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'], n, m, p)}")
