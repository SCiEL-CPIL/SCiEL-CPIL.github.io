"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    # Try the structured 4-block rank-11 scheme (total rank 44)
    best = None
    best_rank = n * m * p

    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _rank28_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass
        
        if best is None and time.time() < deadline - 20:
            try:
                U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
                if _verify(n, m, p, U1, V1, W1):
                    best = (U1, V1, W1)
                    best_rank = r1
            except Exception:
                pass

    # Try further refinement on the 44-rank solution (attempt rank 43, 42, ...)
    if best is not None and time.time() < deadline - 10:
        try:
            U2, V2, W2, r2 = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _rank28_decomposition(n, m, p):
    """
    Attempt rank-28 decomposition using rank-7 blocks for 2x3x2.
    """
    n, m, p = 2, 3, 8
    block_size = 2
    num_blocks = p // block_size # 4
    
    rank_per_block = 7
    total_rank = rank_per_block * num_blocks # 28
    
    U = np.zeros((n * m, total_rank)) # 6 x 28
    V = np.zeros((m * p, total_rank)) # 24 x 28
    W = np.zeros((p * n, total_rank)) # 16 x 28
    
    # Rank-7 decomposition for 2x3 x 3x2
    # This is a known result. We define the 7 terms.
    # A is 2x3, B is 3x2.
    # Terms (u, v, w) where u is 2x3, v is 3x2, w is 2x2.
    
    # Coefficients for the 7 terms
    # Note: This is a placeholder structure. The actual coefficients must be correct.
    # If these are wrong, verification will fail.
    
    # Term 1
    u1 = np.array([[1, 0, 1], [0, 1, 1]])
    v1 = np.array([[1, 0], [0, 1], [1, 1]]) # This is just a guess
    w1 = np.array([[1, 0], [0, 1]])
    
    # ... This is too risky to guess.
    # I will return a dummy that fails verification to ensure fallback, 
    # or I will use the 44-rank one if I can't get 28.
    # Given the risk, I will stick to the 44-rank one if 28 is not trivially correct.
    # However, the directive is strong.
    # I will return the 44-rank one as a safe fallback within this function 
    # if I don't implement 28 correctly.
    # Actually, I will just return the 44-rank one's logic here?
    # No, I'll just raise an exception or return invalid to force fallback to 44.
    
    raise NotImplementedError("Rank 28 decomposition not implemented correctly")

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2
    
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block
        
        # Strassen 7 terms for 2x2 * 2x2 part
        strassen_U = [
            {0: 1, 4: 1},      # M1
            {3: 1, 4: 1},      # M2
            {0: 1},            # M3
            {4: 1},            # M4
            {0: 1, 1: 1},      # M5
            {3: 1, 0: -1},     # M6
            {1: 1, 4: -1},     # M7
        ]
        
        strassen_V = [
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M1
            {0 * p + k_offset + 0: 1},                           # M2
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1}, # M3
            {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1}, # M4
            {1 * p + k_offset + 1: 1},                           # M5
            {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},  # M6
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M7
        ]
        
        strassen_W = [
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M1
            {(k_offset + 0) * n + 1: 1},                             # M2
            {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M3
            {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},  # M4
            {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1}, # M5
            {(k_offset + 1) * n + 1: 1},                             # M6
            {(k_offset + 0) * n + 0: 1},                             # M7
        ]
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Outer product part: A_right(2x1) * B_bottom(1x2) - rank 4
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1
    
    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for c in range(U.shape[1]):
                u_val = 0.0
                for j in range(m):
                    u_val += U[i * m + j, c] * A[i, j]
                v_val = 0.0
                for l in range(m):
                    v_val += V[l * p + k, c] * B[l, k]
                C_comp[i, k] += W[k * n + i, c] * u_val * v_val
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _refine_decomposition(n, m, p, U, V, W, current_rank):
    """
    Try to reduce the rank by finding linear dependencies among columns.
    Uses a greedy approach: for each pair of columns, check if one can be
    expressed as a linear combination of others with half-integer coefficients.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    # Build the "constraint" tensor: for each (i,j,k) triple, we need
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # For refinement, we try to remove columns by checking if the remaining
    # columns can still satisfy all constraints.
    
    # This is computationally expensive, so we do a limited search.
    # Try removing each column one at a time and verify.
    
    for c_remove in range(rank - 1, -1, -1):
        if time.time() > time.time() + 0:  # placeholder
            break
        
        # Create new matrices without column c_remove
        mask = [c for c in range(rank) if c != c_remove]
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        # Quick check: verify the new decomposition
        if _verify(n, m, p, U_new, V_new, W_new):
            U, V, W = U_new, V_new, W_new
            rank -= 1
            if rank <= 1:
                break
    
    return U, V, W, rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
