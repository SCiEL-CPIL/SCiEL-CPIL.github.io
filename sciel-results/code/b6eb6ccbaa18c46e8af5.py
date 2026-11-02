"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, rank, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_rank46():
    """
    Rank-46 decomposition of 3x3 * 3x6 using Laderman's rank-23 for 3x3*3x3.
    """
    n, m, p = 3, 3, 6
    rank = 46

    # Laderman's 23-term decomposition for 3x3 * 3x3
    # Each term: (u, v, w) where each is a 9-vector
    # u: A[0,0],A[0,1],A[0,2],A[1,0],A[1,1],A[1,2],A[2,0],A[2,1],A[2,2]
    # v: B[0,0],B[0,1],B[0,2],B[1,0],B[1,1],B[1,2],B[2,0],B[2,1],B[2,2]
    # w: C[0,0],C[0,1],C[0,2],C[1,0],C[1,1],C[1,2],C[2,0],C[2,1],C[2,2]
    # C[i,k] = sum_t w[i,k,t] * (sum_{a,b} u[a,b,t]*A[a,b]) * (sum_{c,d} v[c,d,t]*B[c,d])

    laderman_terms = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,1,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,1,0]),
        ([0,1,0, 1,0,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,1,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([1,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,1, 0,0,0], [0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 1,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
    ]

    # Verify the Laderman decomposition works for 3x3*3x3
    np.random.seed(42)
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_expected = A_test @ B_test

    C_computed = np.zeros((3, 3))
    for t in range(23):
        u, v, w = laderman_terms[t]
        u = np.array(u).reshape(3, 3)
        v = np.array(v).reshape(3, 3)
        w = np.array(w).reshape(3, 3)
        x_t = np.sum(u * A_test)
        y_t = np.sum(v * B_test)
        C_computed += w * (x_t * y_t)

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return None

    # Build rank-46 decomposition for 3x3 * 3x6
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 46

    # Fill U and V and W for left half (k=0,1,2) and right half (k=3,4,5)
    for t in range(23):
        u, v, w = laderman_terms[t]
        u = np.array(u).reshape(3, 3)
        v = np.array(v).reshape(3, 3)
        w = np.array(w).reshape(3, 3)

        # Left half: columns 0,1,2 of B
        for a in range(3):
            for b in range(3):
                U[a * m + b][t] = u[a, b]
        for j in range(3):
            for k in range(3):
                V[j * p + k][t] = v[j, k]
        for k in range(3):
            for i in range(3):
                W[k * n + i][t] = w[i, k]

        # Right half: columns 3,4,5 of B
        for a in range(3):
            for b in range(3):
                U[a * m + b][t + 23] = u[a, b]
        for j in range(3):
            for k in range(3):
                V[j * p + (k + 3)][t + 23] = v[j, k]
        for k in range(3):
            for i in range(3):
                W[(k + 3) * n + i][t + 23] = w[i, k]

    # Verify the full decomposition
    if not _verify(U, V, W, rank, n, m, p):
        return None

    return {"U": U, "V": V, "W": W}


def _build_rank42():
    """
    Attempt rank-42 decomposition by sharing some terms between left and right halves.
    This is a heuristic attempt - may not work.
    """
    n, m, p = 3, 3, 6
    rank = 42

    # We need to find a way to share 4 terms between the two 3x3*3x3 subproblems
    # This is non-trivial. Let's try a different approach: use the fact that
    # the two halves are independent and see if we can find shared structure.
    
    # For now, return None to indicate this doesn't work
    return None


def _build_rank40():
    """
    Attempt rank-40 decomposition.
    """
    n, m, p = 3, 3, 6
    rank = 40
    return None


def _build_rank38():
    """
    Attempt rank-38 decomposition.
    """
    n, m, p = 3, 3, 6
    rank = 38
    return None


def construct():
    t0 = time.time()
    DEADLINE = t0 + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 3, 6

    # Try rank-46 Laderman construction first
    laderman_result = _build_rank46()
    if laderman_result is not None:
        return laderman_result

    # Try lower ranks (these currently return None)
    for rank in [42, 40, 38]:
        if time.time() > DEADLINE:
            break
        result = _build_rank42() if rank == 42 else _build_rank40() if rank == 40 else _build_rank38()
        if result is not None:
            return result

    # Fallback: Schoolbook decomposition: rank = n*m*p = 54
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify schoolbook (should always pass)
    if not _verify(U, V, W, rank_sb, n, m, p):
        pass  # Should not happen

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
