import time
import numpy as np
from itertools import combinations, product

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i * m + j, j * p + k, k * n + i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i * m + j][r] += v
        for (j, k), v in Bc.items():
            V[j * p + k][r] += v
        for (i, k), v in Cc.items():
            W[k * n + i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_np[a], V_np[b] * W_np[c])
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _build_base(n, m, p):
    """Build a known low-rank decomposition using Strassen-style blocks."""
    cols = []

    def strassen(ai, aj, bj, bk, ci, ck):
        def A(i, j):
            return (ai + i, aj + j)

        def B(j, k):
            return (bj + j, bk + k)

        def C(i, k):
            return (ci + i, ck + k)

        M = [
            ([(A(0, 0), 1), (A(1, 1), 1)], [(B(0, 0), 1), (B(1, 1), 1)], [(C(0, 0), 1), (C(1, 1), 1)]),
            ([(A(1, 0), 1), (A(1, 1), 1)], [(B(0, 0), 1)], [(C(1, 0), 1), (C(1, 1), -1)]),
            ([(A(0, 0), 1)], [(B(1, 0), 1), (B(1, 1), -1)], [(C(0, 1), 1), (C(1, 1), 1)]),
            ([(A(1, 1), 1)], [(B(0, 1), 1), (B(1, 0), -1)], [(C(0, 0), 1), (C(1, 0), 1)]),
            ([(A(0, 0), 1), (A(0, 1), 1)], [(B(1, 0), 1)], [(C(0, 0), -1), (C(0, 1), 1)]),
            ([(A(0, 1), 1), (A(0, 0), -1)], [(B(0, 0), 1), (B(1, 0), 1)], [(C(1, 1), 1)]),
            ([(A(1, 0), 1), (A(1, 1), -1)], [(B(1, 1), 1), (B(0, 1), 1)], [(C(0, 0), 1), (C(0, 1), -1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))

    # 2x2 Strassen for top-left block (rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B)
    strassen(0, 0, 0, 0, 0, 0)
    # 2x2 Strassen for top-right block (rows 0-1, cols 0-1 of A; rows 0-1, cols 2-3 of B)
    strassen(0, 0, 0, 2, 0, 2)
    # Cross terms: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for k in range(2):
        cols.append((
            {(2, 0): 1, (2, 1): 1},
            {(0, k): 1, (1, k): 1},
            {(2, k): 1}
        ))
    # Cross terms: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    for k in range(2):
        cols.append((
            {(2, 0): 1, (2, 1): 1},
            {(0, 2 + k): 1, (1, 2 + k): 1},
            {(2, 2 + k): 1}
        ))
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    cols.append((
        {(0, 2): 1, (1, 2): 1},
        {(2, 0): 1, (2, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    ))
    # A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4]
    cols.append((
        {(0, 2): 1, (1, 2): 1},
        {(2, 2): 1, (2, 3): 1},
        {(0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1}
    ))
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    cols.append((
        {(2, 2): 1},
        {(2, 0): 1, (2, 1): 1},
        {(2, 0): 1, (2, 1): 1}
    ))
    # A[2, 2] * B[2, 2:4] -> C[2, 2:4]
    cols.append((
        {(2, 2): 1},
        {(2, 2): 1, (2, 3): 1},
        {(2, 2): 1, (2, 3): 1}
    ))

    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def _try_merge_columns(U, V, W, n, m, p, t0, deadline):
    """
    Try to reduce rank by merging columns that have compatible structure.
    Two columns r1, r2 can be merged if there exists a scalar alpha such that
    U[r1] = alpha * U[r2], V[r1] = alpha * V[r2], W[r1] = alpha * W[r2]
    (up to the constraint that the tensor equation still holds).
    
    More generally, we try to find a linear dependency among columns.
    """
    rank = len(U[0])
    if time.time() - t0 > deadline:
        return None

    # Build the tensor constraint matrix
    # For each column r, we have U[:,r], V[:,r], W[:,r]
    # The tensor equation: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    n_a = n * m
    n_b = m * p
    n_c = p * n
    
    # Try dropping pairs of columns and checking if the remaining system is consistent
    # For each pair (r1, r2), check if column r2 can be expressed as a combination
    # that preserves the tensor equation
    
    for r1 in range(rank):
        if time.time() - t0 > deadline:
            return None
        for r2 in range(r1 + 1, rank):
            # Check if columns r1 and r2 are "compatible" for merging
            # i.e., there exists alpha such that removing r2 and scaling r1 works
            # This requires: U[:,r2] = alpha * U[:,r1], etc.
            # But more flexibly: try to solve for the merged column
            
            # Simple check: are the support patterns compatible?
            U1 = U_np[:, r1]
            U2 = U_np[:, r2]
            V1 = V_np[:, r1]
            V2 = V_np[:, r2]
            W1 = W_np[:, r1]
            W2 = W_np[:, r2]
            
            # Check if U2 is proportional to U1
            if np.allclose(U2, 0):
                continue
            # Find alpha from first nonzero entry
            idx = np.nonzero(U2)[0]
            if len(idx) == 0:
                continue
            i0 = idx[0]
            if U1[i0] == 0:
                continue
            alpha = U2[i0] / U1[i0]
            
            if np.allclose(U2, alpha * U1) and np.allclose(V2, alpha * V1) and np.allclose(W2, alpha * W1):
                # These columns are redundant - drop r2
                U_new = [row[:r2] + row[r2+1:] for row in U]
                V_new = [row[:r2] + row[r2+1:] for row in V]
                W_new = [row[:r2] + row[r2+1:] for row in W]
                if _verify(U_new, V_new, W_new, n, m, p, rank - 1):
                    return (U_new, V_new, W_new)
    
    return None

def _linear_feasibility_search(n, m, p, target_rank, t0, deadline):
    """
    Search for a rank-r decomposition by fixing U to unit vectors
    and solving for V, W via linear systems.
    
    For each rank-1 term with U being a unit vector e_a (selecting row a of A),
    the contribution to the tensor is: U[a,r]=1, and we need V[:,r] * W[:,r]
    to match the remaining pattern.
    
    We enumerate possible U patterns (sparse) and check linear feasibility.
    """
    if time.time() - t0 > deadline:
        return None
    
    U_np = np.zeros((n * m, target_rank))
    V_np = np.zeros((m * p, target_rank))
    W_np = np.zeros((p * n, target_rank))
    
    # Strategy: each column of U is a single 1 (unit vector)
    # Enumerate assignments of which row each column picks
    # This is a combinatorial search, so we use a heuristic approach
    
    # For a unit vector U[:,r] = e_{a_r}, the constraint becomes:
    # For each (a_r, b, c): V[b,r] * W[c,r] = T[a_r, b, c]
    # For each (a != a_r, b, c): no constraint from this term
    # But the sum over all r must equal T[a,b,c]
    
    # This is still nonlinear in V, W. Let's use a different approach:
    # Fix U, then the equations are bilinear in V, W.
    # We can fix V and solve for W, or use alternating minimization.
    
    # Simpler: try small structured U patterns
    # Pattern 1: each column picks a distinct row
    # Pattern 2: columns pick rows in a balanced way
    
    # For 3x3 x 3x4, n*m = 9, m*p = 12, p*n = 12
    # Try target_rank from 18 to 21
    
    # Use a greedy approach: start with schoolbook, try to merge
    # Actually, let's try a direct construction for specific ranks
    
    # Known: Strassen gives 7 for 2x2. For 3x3, best known is 19 (Larsen).
    # For 3x3 x 3x4, we expect around 18-22.
    
    # Let's try to construct a 19-rank or 20-rank decomposition
    # by combining Strassen blocks more cleverly.
    
    # Approach: Use 2x2 Strassen for the 2x2 sub-blocks and handle
    # the remaining rows/cols with outer products, but try to share terms.
    
    cols = []
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, 7 terms)
    def strassen_block(ai, aj, bi, bk, ci, ck):
        def A(i, j): return (ai + i, aj + j)
        def B(j, k): return (bi + j, bk + k)
        def C(i, k): return (ci + i, ck + k)
        M = [
            ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
            ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
            ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
            ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
            ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))
    
    # Use 2x2 Strassen for four 2x2 sub-blocks
    # But we only have 2 rows in A that can be paired, and 3 cols
    # Let's use a different partition: A as 2+1, B as 2+1
    
    # Clear and rebuild with a better strategy
    cols = []
    
    # Partition: A rows {0,1} and {2}, A cols {0,1} and {2}
    # B rows {0,1} and {2}, B cols {0,1} and {2,3}
    
    # Top-left: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : 7 terms
    strassen_block(0, 0, 0, 0, 0, 0)
    
    # Top-right: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] : 7 terms
    strassen_block(0, 0, 0, 2, 0, 2)
    
    # Bottom-left: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] : 2 terms (outer product)
    for k in range(2):
        cols.append((
            {(2, 0): 1, (2, 1): 1},
            {(0, k): 1, (1, k): 1},
            {(2, k): 1}
        ))
    
    # Bottom-right: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4] : 2 terms
    for k in range(2):
        cols.append((
            {(2, 0): 1, (2, 1): 1},
            {(0, 2+k): 1, (1, 2+k): 1},
            {(2, 2+k): 1}
        ))
    
    # Cross top: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] : 1 term
    cols.append((
        {(0, 2): 1, (1, 2): 1},
        {(2, 0): 1, (2, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    ))
    
    # Cross right: A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4] : 1 term
    cols.append((
        {(0, 2): 1, (1, 2): 1},
        {(2, 2): 1, (2, 3): 1},
        {(0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1}
    ))
    
    # Bottom cross: A[2, 2] * B[2, 0:2] -> C[2, 0:2] : 1 term
    cols.append((
        {(2, 2): 1},
        {(2, 0): 1, (2, 1): 1},
        {(2, 0): 1, (2, 1): 1}
    ))
    
    # Bottom cross: A[2, 2] * B[2, 2:4] -> C[2, 2:4] : 1 term
    cols.append((
        {(2, 2): 1},
        {(2, 2): 1, (2, 3): 1},
        {(2, 2): 1, (2, 3): 1}
    ))
    
    rank = len(cols)
    U, V, W = _from_columns(cols, n, m, p, rank)
    
    # Try to reduce by merging
    if time.time() - t0 < deadline:
        result = _try_merge_columns(U, V, W, n, m, p, t0, deadline)
        if result is not None:
            return result
    
    return (U, V, W)

def _try_reduced_ranks(n, m, p, t0, deadline):
    """Try to find decompositions with rank < 22."""
    best = None
    best_rank = 22
    
    # Try various constructions
    # The base construction gives 22. Try to find 21, 20, 19.
    
    # One approach: use a 3x3 Strassen-like decomposition for the 3x3 part
    # and handle the extra column separately.
    
    # For 3x3 * 3x3, best known rank is 19 (Larsen 2010).
    # For 3x3 * 3x4, we can potentially do better than 22.
    
    # Let's try: 3x3 * 3x3 part with rank 19, plus 3x3 * 3x1 with rank 3
    # Total: 22. Not an improvement.
    
    # Alternative: interleave the computations to share terms.
    # This is complex, so we'll rely on the merge search.
    
    U, V, W = _build_base(n, m, p)
    current_rank = len(U[0])
    
    # Iteratively try to merge columns
    for iteration in range(10):
        if time.time() - t0 > deadline:
            break
        result = _try_merge_columns(U, V, W, n, m, p, t0, deadline)
        if result is None:
            break
        U, V, W = result
        current_rank = len(U[0])
        if current_rank < best_rank:
            best_rank = current_rank
            best = (U, V, W)
    
    if best is not None:
        return best
    return (U, V, W)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    # Get base decomposition
    U, V, W = _build_base(n, m, p)
    current_rank = len(U[0])
    
    best_result = (U, V, W)
    best_rank = current_rank
    
    # Try to reduce rank
    if time.time() - t0 < DEADLINE:
        result = _try_reduced_ranks(n, m, p, t0, DEADLINE)
        if result is not None:
            r = len(result[0][0])
            if r < best_rank:
                best_result = result
                best_rank = r
    
    # Verify
    if _verify(best_result[0], best_result[1], best_result[2], n, m, p, best_rank):
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
