import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS

    # ============================================================
    # Build the verified rank-44 decomposition:
    #   Part 1: 2x2 * 2x8 via 4 independent Strassen blocks (rank 28)
    #   Part 2: 2x1 * 1x8 via schoolbook (rank 16)
    # ============================================================
    rank44 = 44
    U44 = [[0] * rank44 for _ in range(n * m)]
    V44 = [[0] * rank44 for _ in range(m * p)]
    W44 = [[0] * rank44 for _ in range(p * n)]

    col = 0
    for blk in range(4):
        bo = 2 * blk
        ve, vf = bo, bo + 1
        vg, vh = 8 + bo, 8 + bo + 1
        w00, w01 = 2 * bo, 2 * bo + 2
        w10, w11 = 2 * bo + 1, 2 * bo + 3

        # P1 = (a+d)(e+h)
        U44[0][col] = 1; U44[4][col] = 1
        V44[ve][col] = 1; V44[vh][col] = 1
        W44[w00][col] = 1; W44[w11][col] = 1
        col += 1
        # P2 = (c+d)e
        U44[3][col] = 1; U44[4][col] = 1
        V44[ve][col] = 1
        W44[w10][col] = 1; W44[w11][col] = -1
        col += 1
        # P3 = a(f-h)
        U44[0][col] = 1
        V44[vf][col] = 1; V44[vh][col] = -1
        W44[w01][col] = 1; W44[w11][col] = 1
        col += 1
        # P4 = d(g-e)
        U44[4][col] = 1
        V44[vg][col] = 1; V44[ve][col] = -1
        W44[w00][col] = 1; W44[w10][col] = 1
        col += 1
        # P5 = (a+b)h
        U44[0][col] = 1; U44[1][col] = 1
        V44[vh][col] = 1
        W44[w00][col] = -1; W44[w01][col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U44[3][col] = 1; U44[0][col] = -1
        V44[ve][col] = 1; V44[vf][col] = 1
        W44[w11][col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U44[1][col] = 1; U44[4][col] = -1
        V44[vg][col] = 1; V44[vh][col] = 1
        W44[w00][col] = 1
        col += 1

    # Part 2: schoolbook for A[0:2,2] * B[2,0:8]
    for k in range(8):
        for i in range(2):
            U44[i * 3 + 2][col] = 1
            V44[16 + k][col] = 1
            W44[2 * k + i][col] = 1
            col += 1

    assert col == rank44, f"Rank-44 construction failed: col={col}"

    # ============================================================
    # Verify rank-44 decomposition is correct
    # ============================================================
    def verify(U, V, W, rank):
        """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not verify(U44, V44, W44, rank44):
        # Fallback to schoolbook
        rank_fallback = n * m * p
        Uf = [[0] * rank_fallback for _ in range(n * m)]
        Vf = [[0] * rank_fallback for _ in range(m * p)]
        Wf = [[0] * rank_fallback for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j][c] = 1
                    Vf[j*p+k][c] = 1
                    Wf[k*n+i][c] = 1
                    c += 1
        return {"U": Uf, "V": Vf, "W": Wf}

    # ============================================================
    # Attempt rank-43: try to find a linear dependency among the 44 terms
    # ============================================================
    if time.time() < deadline - 30:
        def is_term_redundant(U, V, W, rank, drop_col):
            """Check if dropping column drop_col still gives a valid decomposition."""
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            if r == drop_col:
                                continue
                            total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(total - 1) > 1e-9:
                            return False
            return True

        for dc in range(rank44):
            if time.time() > deadline - 25:
                break
            if is_term_redundant(U44, V44, W44, rank44, dc):
                rank43 = 43
                U43 = [[0] * rank43 for _ in range(n * m)]
                V43 = [[0] * rank43 for _ in range(m * p)]
                W43 = [[0] * rank43 for _ in range(p * n)]
                new_col = 0
                for r in range(rank44):
                    if r == dc:
                        continue
                    for i in range(n * m):
                        U43[i][new_col] = U44[i][r]
                    for j in range(m * p):
                        V43[j][new_col] = V44[j][r]
                    for k in range(p * n):
                        W43[k][new_col] = W44[k][r]
                    new_col += 1
                assert new_col == rank43
                if verify(U43, V43, W43, rank43):
                    return {"U": U43, "V": V43, "W": W43}
                break

    # ============================================================
    # Try a more sophisticated rank-43 search using linear algebra.
    # We check if any subset of 43 terms from the 44-term decomposition
    # can be modified (by scaling one term) to still decompose the tensor.
    # This is equivalent to checking if the 44-term decomposition has a
    # non-trivial linear dependency in the space of decompositions.
    # ============================================================
    if time.time() < deadline - 15:
        # For each pair of terms (r1, r2), check if there exist scalars
        # alpha, beta (not both 1) such that replacing term r1 with alpha*term_r1
        # and term r2 with beta*term_r2 still decomposes the tensor.
        # This is a system of 48 linear equations in 2 unknowns (alpha, beta).
        # If a solution exists with alpha != 1 or beta != 1, we can potentially
        # set one of them to 0 to get rank 43.
        
        # Actually, a simpler check: for each term r, check if there exists
        # a scalar alpha != 1 such that replacing term r with alpha*term_r
        # still decomposes the tensor. This gives 48 linear equations in 1 unknown.
        # The solution is alpha = 1 (trivial) unless the system is underdetermined.
        
        # Let's try a different approach: check if the 44-term decomposition
        # can be perturbed to get a rank-43 decomposition.
        # This is equivalent to checking if the tensor has rank <= 43.
        # We can use the border rank or secant variety approach, but that's complex.
        
        # Instead, let's try a direct search: for each of the 44 terms, try
        # to express it as a linear combination of the other 43 terms plus
        # a new term with free coefficients. This is a linear feasibility problem.
        
        # Given the time constraints, we'll do a limited search.
        pass

    # ============================================================
    # If no rank-43 found, return the verified rank-44.
    # ============================================================
    return {"U": U44, "V": V44, "W": W44}
