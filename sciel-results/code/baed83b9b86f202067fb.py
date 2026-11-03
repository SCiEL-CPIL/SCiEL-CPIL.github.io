"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        for l in range(m):
            T[:, l, :] += np.outer(u[:, l], v[l, :]) * w.T
    
    return np.allclose(T, 1.0, atol=1e-9)

def _build_rectangular_strassen(n, m, p):
    """
    Build rectangular Strassen decomposition for 2x5x6.
    
    Partition:
    - j blocks: {0,1}, {2,3}, {4}
    - k blocks: {0,1,2}, {3,4,5}
    
    For each (j_block, k_block):
    - If j_block has 2 elements and k_block has 3 elements: use 2x2x3 decomposition (rank 9)
    - If j_block has 1 element and k_block has 6 elements: use 2x1x6 decomposition (rank 12)
    
    Total rank: 4*9 + 12 = 48
    """
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: dict {(i,j): val}, v_entries: dict {(j,k): val}, w_entries: dict {(k,i): val}"""
        idx = len(U)
        for i in range(n):
            for j in range(m):
                if not U or len(U) <= idx:
                    U.append([0.0] * (idx + 1))
                U[idx][idx] += u_entries.get((i, j), 0.0)
        
        for j in range(m):
            for k in range(p):
                if not V or len(V) <= idx:
                    V.append([0.0] * (idx + 1))
                V[idx][idx] += v_entries.get((j, k), 0.0)
        
        for k in range(p):
            for i in range(n):
                if not W or len(W) <= idx:
                    W.append([0.0] * (idx + 1))
                W[idx][idx] += w_entries.get((k, i), 0.0)
    
    # Initialize matrices
    U = [[0.0] * 0 for _ in range(n * m)]
    V = [[0.0] * 0 for _ in range(m * p)]
    W = [[0.0] * 0 for _ in range(p * n)]
    
    def _add_term(u_entries, v_entries, w_entries):
        idx = len(U[0])
        for row in U:
            row.append(0.0)
        for row in V:
            row.append(0.0)
        for row in W:
            row.append(0.0)
        
        for (i, j), val in u_entries.items():
            U[i * m + j][idx] += val
        for (j, k), val in v_entries.items():
            V[j * p + k][idx] += val
        for (k, i), val in w_entries.items():
            W[k * n + i][idx] += val
    
    def _strassen_2x2x2(j0, j1, k0, k1):
        """Add 7 Strassen terms for 2x2x2 block."""
        # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
        _add_term({(0, j0): 1, (1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, 0): 1, (k1, 1): 1})
        # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
        _add_term({(1, j0): 1, (1, j1): 1}, {(j0, k0): 1}, {(k0, 1): 1, (k1, 1): -1})
        # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
        _add_term({(0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, 0): 1, (k1, 1): 1})
        # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
        _add_term({(1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, 0): 1, (k0, 1): 1})
        # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
        _add_term({(0, j0): 1, (0, j1): 1}, {(j1, k1): 1}, {(k0, 0): -1, (k1, 0): 1})
        # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
        _add_term({(1, j0): 1, (0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, 1): 1})
        # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
        _add_term({(0, j1): 1, (1, j1): -1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, 0): 1})
    
    def _strassen_2x2x3(j0, j1, k0, k1, k2):
        """
        Add 9 terms for 2x2x3 block.
        Use Strassen on (j0,j1) x (k0,k1) for 7 terms, then handle k2 with 2 more terms.
        """
        _strassen_2x2x2(j0, j1, k0, k1)
        # Handle column k2: C[i,k2] = A[i,j0]*B[j0,k2] + A[i,j1]*B[j1,k2]
        _add_term({(0, j0): 1, (0, j1): 1}, {(j0, k2): 1, (j1, k2): 1}, {(k2, 0): 1})
        _add_term({(1, j0): 1, (1, j1): 1}, {(j0, k2): 1, (j1, k2): 1}, {(k2, 1): 1})
    
    def _strassen_2x1x6(j, k_list):
        """Add 12 terms for 2x1x6 block."""
        for k in k_list:
            _add_term({(0, j): 1}, {(j, k): 1}, {(k, 0): 1})
            _add_term({(1, j): 1}, {(j, k): 1}, {(k, 1): 1})
    
    # Build decomposition
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1, 2), (3, 4, 5)]
    
    for j0, j1 in j_blocks:
        for k0, k1, k2 in k_blocks:
            _strassen_2x2x3(j0, j1, k0, k1, k2)
    
    _strassen_2x1x6(4, list(range(6)))
    
    rank = len(U[0])
    return U, V, W, rank

def _try_eliminate_term(U, V, W, n, m, p, rank, term_idx):
    """Try to eliminate term term_idx. Returns (success, new_U, new_V, new_W, new_rank)."""
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for row in U_new:
        row[term_idx] = 0.0
    for row in V_new:
        row[term_idx] = 0.0
    for row in W_new:
        row[term_idx] = 0.0
    
    if _verify_decomp(U_new, V_new, W_new, n, m, p, rank):
        # Remove the column
        U_final = [[row[j] for j in range(rank) if j != term_idx] for row in U_new]
        V_final = [[row[j] for j in range(rank) if j != term_idx] for row in V_new]
        W_final = [[row[j] for j in range(rank) if j != term_idx] for row in W_new]
        return True, U_final, V_final, W_final, rank - 1
    
    return False, None, None, None, rank

def _snap_to_half_integer(arr):
    """Snap values close to integers or half-integers."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if abs(val - round(val)) < 1e-6:
                new_row.append(float(round(val)))
            elif abs(val - (round(val * 2) / 2.0)) < 1e-6:
                new_row.append(round(val * 2) / 2.0)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin
    
    def _time_left():
        return _deadline - time.time()
    
    # Build rectangular Strassen decomposition
    U, V, W, rank = _build_rectangular_strassen(n, m, p)
    
    if not _verify_decomp(U, V, W, n, m, p, rank):
        # Fallback to known good decomposition
        U, V, W, rank = _build_strassen_fallback(n, m, p)
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to eliminate redundant terms
    current_U, current_V, current_W, current_rank = U, V, W, rank
    
    # Sort terms by norm to try eliminating small ones first
    norms = []
    for r in range(current_rank):
        norm_sq = sum(current_U[i][r] ** 2 for i in range(n * m))
        norm_sq += sum(current_V[l][r] ** 2 for l in range(m * p))
        norm_sq += sum(current_W[k][r] ** 2 for k in range(p * n))
        norms.append(norm_sq)
    
    order = np.argsort(norms)
    
    eliminated = set()
    for idx in order:
        if _time_left() < 2:
            break
        if idx in eliminated:
            continue
        
        success, new_U, new_V, new_W, new_rank = _try_eliminate_term(
            current_U, current_V, current_W, n, m, p, current_rank, idx
        )
        
        if success:
            current_U, current_V, current_W, current_rank = new_U, new_V, new_W, new_rank
            eliminated.add(idx)
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = [row[:] for row in current_U], [row[:] for row in current_V], [row[:] for row in current_W]
    
    # Snap to half-integers
    best_U = _snap_to_half_integer(best_U)
    best_V = _snap_to_half_integer(best_V)
    best_W = _snap_to_half_integer(best_W)
    
    # Final verification
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, best_rank):
        # Fallback
        U, V, W, rank = _build_strassen_fallback(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _build_strassen_fallback(n, m, p):
    """Fallback to standard Strassen decomposition (rank 54)."""
    U = [[0.0] * 0 for _ in range(n * m)]
    V = [[0.0] * 0 for _ in range(m * p)]
    W = [[0.0] * 0 for _ in range(p * n)]
    
    def _add_term(u_entries, v_entries, w_entries):
        idx = len(U[0])
        for row in U:
            row.append(0.0)
        for row in V:
            row.append(0.0)
        for row in W:
            row.append(0.0)
        
        for (i, j), val in u_entries.items():
            U[i * m + j][idx] += val
        for (j, k), val in v_entries.items():
            V[j * p + k][idx] += val
        for (k, i), val in w_entries.items():
            W[k * n + i][idx] += val
    
    def _strassen_2x2x2(j0, j1, k0, k1):
        _add_term({(0, j0): 1, (1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, 0): 1, (k1, 1): 1})
        _add_term({(1, j0): 1, (1, j1): 1}, {(j0, k0): 1}, {(k0, 1): 1, (k1, 1): -1})
        _add_term({(0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, 0): 1, (k1, 1): 1})
        _add_term({(1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, 0): 1, (k0, 1): 1})
        _add_term({(0, j0): 1, (0, j1): 1}, {(j1, k1): 1}, {(k0, 0): -1, (k1, 0): 1})
        _add_term({(1, j0): 1, (0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, 1): 1})
        _add_term({(0, j1): 1, (1, j1): -1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, 0): 1})
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            _strassen_2x2x2(j0, j1, k0, k1)
    
    for k in range(p):
        _add_term({(0, 4): 1}, {(4, k): 1}, {(k, 0): 1})
        _add_term({(1, 4): 1}, {(4, k): 1}, {(k, 1): 1})
    
    rank = len(U[0])
    return U, V, W, rank
# EVOLVE-BLOCK-END

