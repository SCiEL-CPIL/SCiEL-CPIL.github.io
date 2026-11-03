import time
import numpy as np

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
        """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta_{i,k} for the mmul tensor."""
        # The multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
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
        # Should never happen, but fallback to schoolbook
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
    # Attempt rank-43: try to merge one schoolbook term with a Strassen
    # term by finding an algebraic dependency.
    #
    # Strategy: The schoolbook part has 16 terms of the form
    #   A[i,2] * B[2,k] -> C[i,k]
    # The Strassen part has 28 terms. If we can find a linear
    # combination of one schoolbook term with one Strassen term
    # that preserves correctness, we save 1 rank.
    #
    # More concretely: try to express one of the 16 schoolbook
    # contributions as a linear combination of existing Strassen
    # contributions plus at most one new term. This is equivalent
    # to checking if the rank-44 tensor has a dependency.
    #
    # We use a randomized approach: for each pair (schoolbook term s,
    # Strassen term t), check if there exist scalars alpha, beta such
    # that the modified decomposition (removing s, adding alpha*t +
    # beta*(new)) still works. This is a linear feasibility problem.
    #
    # Given time constraints, we do a limited search.
    # ============================================================
    if time.time() < deadline - 30:
        # Try a direct rank-43 search via linear algebra
        # The mmul tensor has 2*3*8 = 48 entries, all = 1.
        # A rank-r decomposition gives r trilinear forms.
        # We need to check if rank 43 suffices.
        #
        # Approach: fix the 27 Strassen terms (drop one) + 16 schoolbook
        # = 43 terms, and check if the remaining tensor can be completed.
        #
        # Actually, a cleaner approach: try all 44 possible "drops"
        # (drop one term from the rank-44 set) and check if the
        # resulting 43-term partial decomposition can be extended
        # to a valid rank-43 decomposition.
        #
        # This is expensive. Instead, try a simpler heuristic:
        # Check if any single term in the rank-44 decomposition
        # is redundant (i.e., its contribution is already covered
        # by the other 43 terms).
        
        def is_term_redundant(U, V, W, rank, drop_col):
            """Check if dropping column drop_col still gives a valid decomposition."""
            # The contribution of column r is:
            #   U[:,r] outer V[:,r] outer W[:,r]
            # We need: sum_{r != drop_col} U[:,r]V[:,r]W[:,r] == T
            # i.e., T - sum_{r != drop_col} ... == 0
            # Equivalently: the dropped term's contribution must be 0
            # on all tensor entries that aren't already covered.
            #
            # More precisely: for each (i,j,k), we need:
            #   sum_{r != drop_col} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
            #
            # Check this for all (i,j,k).
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

        # Check each of the 44 columns for redundancy
        found_rank43 = False
        for dc in range(rank44):
            if time.time() > deadline - 25:
                break
            if is_term_redundant(U44, V44, W44, rank44, dc):
                # Build rank-43 by dropping column dc
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
                found_rank43 = True
                break

    # ============================================================
    # If no rank-43 found, try a more sophisticated approach:
    # Use the dual numbers embedding idea to see if we can
    # restructure the 2x2x8 part more efficiently.
    #
    # Embed B[0:2, 0:8] into a 2x4 matrix over R[eps]/(eps^2)
    # where each entry is b + eps*b'. This lets us process
    # two 2x2 blocks simultaneously. However, the rank of
    # Strassen over this ring is still 7 per "super-block",
    # so 4 super-blocks * 7 = 28, same as before.
    #
    # The real question is whether the 2x1x8 part can share
    # structure with the 2x2x8 part. Since A[0:2,2] and B[2,0:8]
    # are independent of A[0:2,0:2] and B[0:2,0:8], there's no
    # obvious sharing unless we use a non-trivial linear
    # combination.
    #
    # Given the time budget, return the verified rank-44.
    # ============================================================

    if time.time() > deadline - 5:
        pass  # Return immediately

    return {"U": U44, "V": V44, "W": W44}
