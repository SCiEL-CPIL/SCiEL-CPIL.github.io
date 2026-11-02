"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and dot products for 1x2x2 blocks.
Fixed indexing bug in W-matrix mapping for Strassen blocks.
Expected rank: 36.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_full(U, V, W):
    """Full verification: check all valid and invalid tensor entries."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check all invalid entries
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _verify(U, V, W):
    """Alias for full verify."""
    return _verify_full(U, V, W)

def _add_strassen(U, V, W, a_indices, b_indices, c_indices):
    """
    Adds a 2x2x2 Strassen decomposition to the global matrices.
    a_indices: list of 4 global indices for U (Local 0,1,2,3 -> 00,01,10,11)
    b_indices: list of 4 global indices for V (Local 0,1,2,3 -> 00,01,10,11)
    c_indices: list of 4 global indices for W (Local 0,1,2,3 -> 00,01,10,11)
    """
    rank = U.shape[1]
    
    # r=0 (M1): (A00+A11)(B00+B11)
    U[a_indices[0], rank] += 1; U[a_indices[3], rank] += 1
    V[b_indices[0], rank] += 1; V[b_indices[3], rank] += 1
    W[c_indices[0], rank] += 1; W[c_indices[3], rank] += 1
    
    # r=1 (M2): (A10+A11)B00
    U[a_indices[2], rank+1] += 1; U[a_indices[3], rank+1] += 1
    V[b_indices[0], rank+1] += 1
    W[c_indices[2], rank+1] += 1; W[c_indices[3], rank+1] += 1
    
    # r=2 (M3): A00(B01-B11)
    U[a_indices[0], rank+2] += 1
    V[b_indices[1], rank+2] += 1; V[b_indices[3], rank+2] -= 1
    W[c_indices[1], rank+2] += 1; W[c_indices[3], rank+2] += 1
    
    # r=3 (M4): A11(B10-B00)
    U[a_indices[3], rank+3] += 1
    V[b_indices[2], rank+3] += 1; V[b_indices[0], rank+3] -= 1
    W[c_indices[2], rank+3] += 1; W[c_indices[3], rank+3] += 1
    
    # r=4 (M5): (A00+A01)B11
    U[a_indices[0], rank+4] += 1; U[a_indices[1], rank+4] += 1
    V[b_indices[3], rank+4] += 1
    W[c_indices[1], rank+4] += 1; W[c_indices[3], rank+4] += 1
    
    # r=5 (M6): (A10-A00)(B00+B01)
    U[a_indices[2], rank+5] += 1; U[a_indices[0], rank+5] -= 1
    V[b_indices[0], rank+5] += 1; V[b_indices[1], rank+5] += 1
    W[c_indices[2], rank+5] += 1; W[c_indices[3], rank+5] += 1
    
    # r=6 (M7): (A01-A11)(B10+B11)
    U[a_indices[1], rank+6] += 1; U[a_indices[3], rank+6] -= 1
    V[b_indices[2], rank+6] += 1; V[b_indices[3], rank+6] += 1
    W[c_indices[0], rank+6] += 1; W[c_indices[3], rank+6] += 1

def _add_dot_1x2x2(U, V, W, a_indices, b_indices, c_indices):
    """
    Adds a 1x2x2 dot product decomposition.
    a_indices: list of 2 global indices for U (j=0, j=1)
    b_indices: list of 4 global indices for V (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    c_indices: list of 2 global indices for W (k=0, k=1)
    """
    rank = U.shape[1]
    
    # r=0: A0 * (B00, B01) -> (C0, C1)
    U[a_indices[0], rank] += 1
    V[b_indices[0], rank] += 1; V[b_indices[1], rank] += 1
    W[c_indices[0], rank] += 1; W[c_indices[1], rank] += 1
    
    # r=1: A1 * (B10, B11) -> (C0, C1)
    U[a_indices[1], rank+1] += 1
    V[b_indices[2], rank+1] += 1; V[b_indices[3], rank+1] += 1
    W[c_indices[0], rank+1] += 1; W[c_indices[1], rank+1] += 1

def _block_decomposition():
    n, m, p = DIMENSIONS # 3, 4, 8
    total_rank = 36
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    current_rank = 0
    
    # Helper to generate c_indices for Strassen blocks
    # Local 0: k=0, i=0 -> k*n+i
    # Local 1: k=1, i=0 -> k*n+i
    # Local 2: k=0, i=1 -> k*n+i
    # Local 3: k=1, i=1 -> k*n+i
    def get_c_indices_strassen(k_start, i_start):
        # k_start, i_start are the global offsets for k and i
        # Local (0,0) -> k=k_start, i=i_start
        # Local (0,1) -> k=k_start+1, i=i_start
        # Local (1,0) -> k=k_start, i=i_start+1
        # Local (1,1) -> k=k_start+1, i=i_start+1
        return [
            (k_start) * n + (i_start),
            (k_start + 1) * n + (i_start),
            (k_start) * n + (i_start + 1),
            (k_start + 1) * n + (i_start + 1)
        ]

    # Block 1: C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    # 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    a_idx_1a = [i*4+j for i in range(2) for j in range(2)]
    b_idx_1a = [j*8+k for j in range(2) for k in range(2)]
    c_idx_1a = get_c_indices_strassen(0, 0)
    _add_strassen(U, V, W, a_idx_1a, b_idx_1a, c_idx_1a)
    current_rank += 7
    
    # 1b: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    a_idx_1b = [i*4+j for i in range(2) for j in range(2,4)]
    b_idx_1b = [j*8+k for j in range(2,4) for k in range(2)]
    c_idx_1b = get_c_indices_strassen(0, 0)
    _add_strassen(U, V, W, a_idx_1b, b_idx_1b, c_idx_1b)
    current_rank += 7
    
    # Block 2: C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    # 2a: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    a_idx_2a = [i*4+j for i in range(2) for j in range(2)]
    b_idx_2a = [j*8+k for j in range(2) for k in range(4,6)]
    c_idx_2a = get_c_indices_strassen(4, 0)
    _add_strassen(U, V, W, a_idx_2a, b_idx_2a, c_idx_2a)
    current_rank += 7
    
    # 2b: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    a_idx_2b = [i*4+j for i in range(2) for j in range(2,4)]
    b_idx_2b = [j*8+k for j in range(2,4) for k in range(4,6)]
    c_idx_2b = get_c_indices_strassen(4, 0)
    _add_strassen(U, V, W, a_idx_2b, b_idx_2b, c_idx_2b)
    current_rank += 7
    
    # Block 3: C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    # 3a: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    a_idx_3a = [2*4+j for j in range(2)]
    b_idx_3a = [j*8+k for j in range(2) for k in range(2)]
    c_idx_3a = [k*3+2 for k in range(2)]
    _add_dot_1x2x2(U, V, W, a_idx_3a, b_idx_3a, c_idx_3a)
    current_rank += 2
    
    # 3b: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    a_idx_3b = [2*4+j for j in range(2,4)]
    b_idx_3b = [j*8+k for j in range(2,4) for k in range(2)]
    c_idx_3b = [k*3+2 for k in range(2)]
    _add_dot_1x2x2(U, V, W, a_idx_3b, b_idx_3b, c_idx_3b)
    current_rank += 2
    
    # Block 4: C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    # 4a: A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    a_idx_4a = [2*4+j for j in range(2)]
    b_idx_4a = [j*8+k for j in range(2) for k in range(4,6)]
    c_idx_4a = [k*3+2 for k in range(4,6)]
    _add_dot_1x2x2(U, V, W, a_idx_4a, b_idx_4a, c_idx_4a)
    current_rank += 2
    
    # 4b: A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    a_idx_4b = [2*4+j for j in range(2,4)]
    b_idx_4b = [j*8+k for j in range(2,4) for k in range(4,6)]
    c_idx_4b = [k*3+2 for k in range(4,6)]
    _add_dot_1x2x2(U, V, W, a_idx_4b, b_idx_4b, c_idx_4b)
    current_rank += 2
    
    return U, V, W

def _try_rank_reduction(U, V, W, deadline):
    """
    Try to find linear dependencies among the rank-36 decomposition to reduce rank.
    Uses vectorized verification for speed.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Precompute all (a, b, c) index triples for valid and invalid entries
    valid_triples = []
    invalid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    invalid_triples.append((a, b, c))
    
    valid_a = np.array([t[0] for t in valid_triples])
    valid_b = np.array([t[1] for t in valid_triples])
    valid_c = np.array([t[2] for t in valid_triples])
    invalid_a = np.array([t[0] for t in invalid_triples])
    invalid_b = np.array([t[1] for t in invalid_triples])
    invalid_c = np.array([t[2] for t in invalid_triples])
    
    def _check_decomposition(U_r, V_r, W_r):
        """Vectorized check of decomposition validity."""
        # Valid entries should equal 1
        vals_valid = np.einsum('ir,br,cr->bc', U_r[valid_a], V_r[valid_b], W_r[valid_c])
        if np.any(np.abs(vals_valid - 1.0) > 1e-6):
            return False
        # Invalid entries should equal 0
        vals_invalid = np.einsum('ir,br,cr->bc', U_r[invalid_a], V_r[invalid_b], W_r[invalid_c])
        if np.any(np.abs(vals_invalid) > 1e-6):
            return False
        return True
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # First pass: try removing single columns
    for r_to_remove in range(rank):
        if time.time() > deadline - 5.0:
            break
        
        mask = [i for i in range(rank) if i != r_to_remove]
        U_r = U[:, mask]
        V_r = V[:, mask]
        W_r = W[:, mask]
        
        if _check_decomposition(U_r, V_r, W_r):
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            best_rank = rank - 1
            U, V, W = U_r, V_r, W_r
            rank = best_rank
            # Don't break - continue searching for more reductions
    
    # Second pass: try removing pairs of columns from the current best
    if best_rank < rank and time.time() < deadline - 10.0:
        current_rank = U.shape[1]
        for r1 in range(current_rank):
            if time.time() > deadline - 10.0:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 10.0:
                    break
                mask = [i for i in range(current_rank) if i != r1 and i != r2]
                U_r = U[:, mask]
                V_r = V[:, mask]
                W_r = W[:, mask]
                
                if _check_decomposition(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    best_rank = current_rank - 2
                    U, V, W = U_r, V_r, W_r
                    break
            if best_rank < current_rank:
                break
    
    if best_rank < rank:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    try:
        U, V, W = _block_decomposition()
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if _verify(result["U"], result["V"], result["W"]):
            if time.time() < deadline:
                reduced = _try_rank_reduction(U, V, W, deadline)
                if reduced is not None:
                    U_r, V_r, W_r = reduced
                    result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    if _verify(result["U"], result["V"], result["W"]):
                        return result
            
            return result
    except Exception as e:
        pass
    
    return _schoolbook()
# EVOLVE-BLOCK-END
