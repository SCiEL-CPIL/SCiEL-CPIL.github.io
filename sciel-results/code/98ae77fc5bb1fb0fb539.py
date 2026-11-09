"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition."""
import time
import itertools

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    best = None  # (rank, U, V, W)

    # Build the multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    #   sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]  for all a,b,c_idx
    # where a = i*m+j, b = j*p+k, c_idx = k*n+i.

    # Symmetries:
    #   - Column permutation: fix nothing yet (we'll handle in search)
    #   - Scaling: U_c * V_c * W_c = 1. Fix W_c = 1, then U_c * V_c = 1.
    #   - So W is all 1s, and V_c = 1/U_c (but we want integers, so U_c = ±1, V_c = ±1)
    #     OR we allow U_c and V_c to be arbitrary integers with U_c*V_c = 1,
    #     which forces U_c = V_c = ±1.
    #
    # Actually, that's too restrictive. Let me reconsider.
    # The constraint is sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx].
    # With W_c = 1: sum_c U[a,c] V[b,c] = T[a,b,c_idx]  ... no, W depends on c_idx too.
    #
    # Let me just do a direct search over small integer entries.

    # For rank r, we have r columns. Each column c has:
    #   U[:,c] in Z^12, V[:,c] in Z^20, W[:,c] in Z^15
    # Constraint: for all (a,b,c_idx): sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
    #
    # This is a system of 60 equations in 12r + 20r + 15r = 47r unknowns.
    # For r < 60, the system is underdetermined, so solutions exist generically,
    # but we need integer solutions.
    #
    # Strategy: try ranks from 60 downward. For each rank, do a DFS search
    # over integer entries of U, V, W with small magnitude, checking constraints.

    # Precompute tensor entries
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[(a, b, c_idx)] = 1

    # All (a, b, c_idx) triples
    triples = list(T.keys())

    def check(U, V, W, r):
        """Check if U, V, W of rank r satisfy the tensor equations."""
        for (a, b, c_idx) in triples:
            s = 0
            for c in range(r):
                s += U[a][c] * V[b][c] * W[c_idx][c]
            if s != T[(a, b, c_idx)]:
                return False
        return True

    # Schoolbook solution (rank 60)
    U0 = [[0]*60 for _ in range(n*m)]
    V0 = [[0]*60 for _ in range(m*p)]
    W0 = [[0]*60 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    best = (60, U0, V0, W0)

    # For ranks below 60, use a randomized + local search approach
    # Try to find integer decompositions of lower rank.
    #
    # Approach: Start from the schoolbook solution and try to merge columns.
    # Two columns c1, c2 can be merged if their "tensor contributions" sum
    # to something that can be represented by a single column.
    #
    # Alternative: Use the fact that for matrix multiplication, there are
    # known lower bounds and constructions. For 3x4x5, let's try a
    # systematic search with pruning.

    # Let's try a direct DFS for small ranks, with time limit.
    # For rank r, we search over U entries first (12*r variables),
    # then for each U, solve for V and W.
    #
    # Given U (12 x r), the constraint is:
    #   sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
    #
    # This is bilinear in (V, W) for fixed U. Hard to solve directly.
    #
    # Let me try a different approach: use the structure of matrix multiplication.
    # A 3x4 times 4x5 matrix product can be written as:
    #   (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor is T[i,j,k] = A[i,j]*B[j,k].
    #
    # Known results: the rank of matrix multiplication m x n x p is at least
    # max(m,n,p)^2 and at most m*n*p. For 3x4x5, lower bound is 25, upper is 60.
    #
    # Let me try a randomized search: pick random integer U, V, W with small
    # entries and check. This is unlikely to work for rank < 60.
    #
    # Better: use the schoolbook solution and try to reduce rank by
    # identifying linear dependencies.
    #
    # Actually, let me just try a few specific constructions.
    #
    # One known trick: if we can find a rank-1 decomposition of the "difference"
    # tensor, we can reduce the rank.
    #
    # For now, let's try ranks 59, 58, ... with a time-limited search.
    # If we can't find a better solution, return the schoolbook solution.

    for r in range(59, 24, -1):
        if time.time() > deadline - 10:
            break
        # Try to find a rank-r decomposition
        # Use a simple randomized approach with restarts
        found = False
        start = time.time()
        while not found and time.time() - start < 10:
            # Random initialization with small integers
            import random
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            if check(U, V, W, r):
                found = True
                best = (r, U, V, W)
                break
        if found:
            break

    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
