"""
Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition.

Strategy:
Partition j-indices into groups of 2 (plus one singleton) and k-indices similarly.
Use Strassen's 7-factor algorithm for 2x2x2 blocks, and schoolbook for the rest.

j_groups = [[0,1], [2,3], [4]]
k_groups = [[0,1], [2,3], [4]]

Blocks:
(0,0): 2x2x2 -> Strassen rank 7
(0,1): 2x2x2 -> Strassen rank 7
(0,2): 2x2x1 -> schoolbook rank 4
(1,0): 2x2x2 -> Strassen rank 7
(1,1): 2x2x2 -> Strassen rank 7
(1,2): 2x2x1 -> schoolbook rank 4
(2,0): 2x1x2 -> schoolbook rank 4
(2,1): 2x1x2 -> schoolbook rank 4
(2,2): 2x1x1 -> schoolbook rank 2

Total rank = 7+7+4+7+7+4+4+4+2 = 46
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS


def _verify(U, V, W, n, m, p, num_tests=10):
    """Verify decomposition using multiple random test matrices."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    for test in range(num_tests):
        rng = np.random.RandomState(42 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()
        
        # W uses k*n+i indexing: C_flat[l] where l = k*n + i corresponds to C[i,k]
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        
        C_comp = np.zeros(p * n)
        for c in range(rank):
            vA = float(U_np[:, c] @ A)
            vB = float(V_np[:, c] @ B)
            if vA != 0.0 and vB != 0.0:
                C_comp += W_np[:, c] * vA * vB
        
        if not np.allclose(C_comp, C_true_wn, atol=1e-9):
            return False
    
    return True


def _build_rank46():
    """
    Build a rank-46 decomposition for 2x5x5.
    
    Partition j into [[0,1], [2,3], [4]] and k into [[0,1], [2,3], [4]].
    Use Strassen for 2x2x2 blocks, schoolbook otherwise.
    """
    n, m, p = N, M, P
    
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]
    
    U_entries = []  # (factor_id, u_row, value)
    V_entries = []  # (factor_id, v_row, value)
    W_entries = []  # (factor_id, w_row, value)
    
    fid = 0
    
    # Strassen's 7 factors for 2x2x2
    # Each factor is (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, coeff)
    # v_entries: list of (j, k, coeff)
    # w_entries: list of (k, i, coeff)
    strassen_7 = [
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)]),
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)]),
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)]),
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]
    
    for a in range(len(j_groups)):
        for b in range(len(k_groups)):
            jg = j_groups[a]
            kg = k_groups[b]
            sj = len(jg)
            sk = len(kg)
            
            if sj == 2 and sk == 2:
                # Use Strassen
                for (u_ent, v_ent, w_ent) in strassen_7:
                    for (i, jl, c) in u_ent:
                        gj = jg[jl]
                        U_entries.append((fid, i * m + gj, c))
                    for (jl, kl, c) in v_ent:
                        gj = jg[jl]
                        gk = kg[kl]
                        V_entries.append((fid, gj * p + gk, c))
                    for (kl, i, c) in w_ent:
                        gk = kg[kl]
                        W_entries.append((fid, gk * n + i, c))
                    fid += 1
            else:
                # Schoolbook
                for i in range(n):
                    for jl in range(sj):
                        for kl in range(sk):
                            gj = jg[jl]
                            gk = kg[kl]
                            U_entries.append((fid, i * m + gj, 1))
                            V_entries.append((fid, gj * p + gk, 1))
                            W_entries.append((fid, gk * n + i, 1))
                            fid += 1
    
    rank = fid
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (f, r, v) in U_entries:
        U[r][f] += v
    for (f, r, v) in V_entries:
        V[r][f] += v
    for (f, r, v) in W_entries:
        W[r][f] += v
    
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Return the schoolbook rank-50 decomposition."""
    n, m, p = N, M, P
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


def _try_reduce_rank(base_result, start_time, deadline):
    """
    Try to reduce the rank of the base decomposition by checking if any
    factor can be removed or if factors can be merged.
    
    This is a heuristic search that tries to find a lower rank decomposition
    by exploiting linear dependencies in the factor matrices.
    """
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    
    n, m, p = N, M, P
    rank = U.shape[1]
    
    # Check if any column of U, V, or W is zero
    # If a column is zero in any of the three, that factor contributes nothing
    for r in range(rank):
        if time.time() > deadline - 10:
            break
        if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
            # Remove this factor
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            rank -= 1
            print(f"Removed zero factor at position {r}, new rank: {rank}")
    
    # Try to find linear dependencies among the factors
    # If factor r1 and r2 are linearly dependent in all three matrices,
    # we might be able to merge them (but this is complex)
    
    # For now, just check if we can reduce by checking rank of the factor matrices
    # If rank(U) < rank, some factors are redundant
    U_rank = np.linalg.matrix_rank(U)
    V_rank = np.linalg.matrix_rank(V)
    W_rank = np.linalg.matrix_rank(W)
    
    min_rank = min(U_rank, V_rank, W_rank)
    if min_rank < rank:
        # There are linear dependencies, but we need to be careful
        # A factor is redundant only if it's in the null space of all three
        # This is more complex, so we'll just report the potential
        print(f"Potential rank reduction: U_rank={U_rank}, V_rank={V_rank}, W_rank={W_rank}, current={rank}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
        "rank": U.shape[1]
    }


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix mult.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = N, M, P
    
    # Build the rank-46 decomposition
    result = _build_rank46()
    
    # Verify it's correct
    if not _verify(result["U"], result["V"], result["W"], n, m, p, num_tests=10):
        # Should not happen, but fallback to schoolbook
        return _build_schoolbook()
    
    # Try to reduce rank if we have time
    if time.time() < deadline - 20:
        reduced = _try_reduce_rank(result, start_time, deadline)
        if reduced["rank"] < len(result["U"][0]):
            # Verify the reduced version
            if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p, num_tests=10):
                return reduced
    
    return result
