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

def _try_merge_columns_advanced(U, V, W, n, m, p, t0, deadline):
    """
    Advanced column merging using linear algebra to find dependencies.
    For a set of columns S, we check if there exist coefficients alpha_r (not all zero)
    such that for all (a,b,c): sum_{r in S} alpha_r * U[a,r]*V[b,r]*W[c,r] = 0
    This is a multilinear constraint. We approximate by checking pairwise and triple
    combinations more carefully.
    """
    rank = len(U[0])
    if time.time() - t0 > deadline:
        return None

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    n_a = n * m
    n_b = m * p
    n_c = p * n
    
    # First pass: exact proportional merge (fast)
    for r1 in range(rank):
        if time.time() - t0 > deadline:
            return None
        for r2 in range(r1 + 1, rank):
            U1 = U_np[:, r1]
            U2 = U_np[:, r2]
            V1 = V_np[:, r1]
            V2 = V_np[:, r2]
            W1 = W_np[:, r1]
            W2 = W_np[:, r2]
            
            if np.allclose(U2, 0) or np.allclose(U1, 0):
                continue
            idx = np.nonzero(U2)[0]
            if len(idx) == 0:
                continue
            i0 = idx[0]
            if U1[i0] == 0:
                continue
            alpha = U2[i0] / U1[i0]
            
            if np.allclose(U2, alpha * U1, atol=1e-10) and np.allclose(V2, alpha * V1, atol=1e-10) and np.allclose(W2, alpha * W1, atol=1e-10):
                U_new = [row[:r2] + row[r2+1:] for row in U]
                V_new = [row[:r2] + row[r2+1:] for row in V]
                W_new = [row[:r2] + row[r2+1:] for row in W]
                if _verify(U_new, V_new, W_new, n, m, p, rank - 1):
                    return (U_new, V_new, W_new)
    
    # Second pass: try to find a column that can be expressed as a linear combination
    # of others in a way that preserves the tensor structure
    # For each column r, check if removing it and adjusting others works
    for r_drop in range(rank):
        if time.time() - t0 > deadline:
            return None
        
        # Try dropping column r_drop and see if we can fix the remaining columns
        # by solving a linear system
        # The constraint is: for each (a,b,c), sum_{r != r_drop} U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # This is nonlinear, so we use a different approach:
        # Check if the column r_drop is "redundant" in the sense that its contribution
        # can be absorbed by other columns
        
        # Build the residual tensor if we drop r_drop
        U_rem = np.delete(U_np, r_drop, axis=1)
        V_rem = np.delete(V_np, r_drop, axis=1)
        W_rem = np.delete(W_np, r_drop, axis=1)
        
        # Compute the current tensor with r_drop removed
        current = np.zeros((n_a, n_b, n_c))
        for a in range(n_a):
            for b in range(n_b):
                for c in range(n_c):
                    current[a, b, c] = np.dot(U_rem[a], V_rem[b] * W_rem[c])
        
        # The target tensor
        target = _target_tensor(n, m, p)
        target_arr = np.zeros((n_a, n_b, n_c))
        for (a, b, c), val in target.items():
            target_arr[a, b, c] = val
        
        # Residual = target - current
        residual = target_arr - current
        
        # If residual is zero, we can drop r_drop
        if np.allclose(residual, 0, atol=1e-10):
            U_new = [row[:r_drop] + row[r_drop+1:] for row in U]
            V_new = [row[:r_drop] + row[r_drop+1:] for row in V]
            W_new = [row[:r_drop] + row[r_drop+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p, rank - 1):
                return (U_new, V_new, W_new)
        
        # Try to see if residual can be represented as a rank-1 tensor
        # i.e., residual[a,b,c] = U_new[a] * V_new[b] * W_new[c] for some new column
        # This is equivalent to checking if residual has rank 1 in the tensor sense
        # We can check by seeing if residual[a,b,c] * residual[a',b',c'] == residual[a,b',c'] * residual[a',b,c]
        # for all indices (this is the condition for a rank-1 tensor)
        
        # But we need to find specific U, V, W vectors
        # Let's try: pick a nonzero entry of residual, say residual[a0,b0,c0]
        nonzero = np.nonzero(residual)
        if len(nonzero[0]) == 0:
            continue
        
        a0, b0, c0 = nonzero[0][0], nonzero[1][0], nonzero[2][0]
        val0 = residual[a0, b0, c0]
        
        # Try to construct: U_new[a] = residual[a, b0, c0] / val0
        #                V_new[b] = residual[a0, b, c0] / val0
        #                W_new[c] = residual[a0, b0, c] / val0
        # Then check if U_new[a] * V_new[b] * W_new[c] == residual[a,b,c]
        
        U_new_vec = residual[:, b0, c0] / val0
        V_new_vec = residual[a0, :, c0] / val0
        W_new_vec = residual[a0, b0, :] / val0
        
        # Verify
        reconstructed = np.einsum('a,b,c->abc', U_new_vec, V_new_vec, W_new_vec)
        if np.allclose(reconstructed, residual, atol=1e-10):
            # We can add this new column and drop r_drop
            U_new = [row[:r_drop] + [U_new_vec[a]] for row in U]
            V_new = [row[:r_drop] + [V_new_vec[b]] for row in V]
            W_new = [row[:r_drop] + [W_new_vec[c]] for row in W]
            # But this keeps the same rank! We need to drop r_drop and NOT add
            # Actually, we want to check if the residual can be absorbed
            # Let me reconsider: if residual is rank-1, it means we need an extra column
            # So this doesn't help reduce rank directly
            pass
    
    return None

def _linear_feasibility_search(n, m, p, target_rank, t0, deadline):
    """
    Search for a rank-r decomposition by fixing U to unit vectors
    and solving for V, W via linear systems.
    """
    if time.time() - t0 > deadline:
        return None
    
    U_np = np.zeros((n * m, target_rank))
    V_np = np.zeros((m * p, target_rank))
    W_np = np.zeros((p * n, target_rank))
    
    cols = []
    
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
    
    cols = []
    
    strassen_block(0, 0, 0, 0, 0, 0)
    strassen_block(0, 0, 0, 2, 0, 2)
    
    for k in range(2):
        cols.append((
            {(2, 0): 1, (2, 1): 1},
            {(0, k): 1, (1, k): 1},
            {(2, k): 1}
        ))
    
    for k in range(2):
        cols.append((
            {(2, 0): 1, (2, 1): 1},
            {(0, 2+k): 1, (1, 2+k): 1},
            {(2, 2+k): 1}
        ))
    
    cols.append((
        {(0, 2): 1, (1, 2): 1},
        {(2, 0): 1, (2, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    ))
    
    cols.append((
        {(0, 2): 1, (1, 2): 1},
        {(2, 2): 1, (2, 3): 1},
        {(0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1}
    ))
    
    cols.append((
        {(2, 2): 1},
        {(2, 0): 1, (2, 1): 1},
        {(2, 0): 1, (2, 1): 1}
    ))
    
    cols.append((
        {(2, 2): 1},
        {(2, 2): 1, (2, 3): 1},
        {(2, 2): 1, (2, 3): 1}
    ))
    
    rank = len(cols)
    U, V, W = _from_columns(cols, n, m, p, rank)
    
    if time.time() - t0 < deadline:
        result = _try_merge_columns_advanced(U, V, W, n, m, p, t0, deadline)
        if result is not None:
            return result
    
    return (U, V, W)

def _try_reduced_ranks(n, m, p, t0, deadline):
    """Try to find decompositions with rank < 22."""
    best = None
    best_rank = 22
    
    U, V, W = _build_base(n, m, p)
    current_rank = len(U[0])
    
    for iteration in range(10):
        if time.time() - t0 > deadline:
            break
        result = _try_merge_columns_advanced(U, V, W, n, m, p, t0, deadline)
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
    DEADLINE = 145.0

    U, V, W = _build_base(n, m, p)
    current_rank = len(U[0])
    
    best_result = (U, V, W)
    best_rank = current_rank
    
    if time.time() - t0 < DEADLINE:
        result = _try_reduced_ranks(n, m, p, t0, DEADLINE)
        if result is not None:
            r = len(result[0][0])
            if r < best_rank:
                best_result = result
                best_rank = r
    
    if _verify(best_result[0], best_result[1], best_result[2], n, m, p, best_rank):
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
