"""Matrix multiplication 2x5x6 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _to_halfint(x):
    """Convert to integer or half-integer, or None if not representable."""
    if isinstance(x, int):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float):
        rounded_2x = round(2 * x)
        if abs(2 * x - rounded_2x) < 1e-9:
            result = rounded_2x / 2
            if abs(result - round(result)) < 1e-9:
                return int(round(result))
            else:
                return result
        return None
    if hasattr(x, 'numerator') and hasattr(x, 'denominator'):
        if x.denominator == 1:
            return x.numerator
        if x.denominator == 2:
            return x.numerator / 2
        return None
    return None

def _verify(U, V, W, n, m, p):
    """Verify decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if _to_halfint(val) is None:
                    return False
    # Convert to numpy arrays for efficient verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Build tensor T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
    T = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    # Check diagonal entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                if abs(T[a, b, d] - 1.0) > 1e-9:
                    return False
    # Check off-diagonal entries are 0
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                mask[i*m+j, j*p+k, k*n+i] = True
    off_diag = T.copy()
    off_diag[mask] = 0
    if np.any(np.abs(off_diag) > 1e-9):
        return False
    return True

def _strassen_222():
    """
    Strassen's 7-term decomposition for 2x2x2.
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([-1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, -1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]),
    ]
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    for c in range(7):
        u, v, w = terms[c]
        for idx in range(4):
            U[idx][c] = u[idx]
            V[idx][c] = v[idx]
            W[idx][c] = w[idx]
    return U, V, W

def _schoolbook_block(n_b, m_b, p_b):
    """Schoolbook decomposition for n_b x m_b x p_b block."""
    rank = n_b * m_b * p_b
    U = [[0] * rank for _ in range(n_b * m_b)]
    V = [[0] * rank for _ in range(m_b * p_b)]
    W = [[0] * rank for _ in range(p_b * n_b)]
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i * m_b + j][col] = 1
                V[j * p_b + k][col] = 1
                W[k * n_b + i][col] = 1
                col += 1
    return U, V, W

def _get_partitions(n, max_parts=None):
    """Generate all integer partitions of n (as lists, sorted descending)."""
    if max_parts is None:
        max_parts = n
    if n == 0:
        yield []
        return
    for first in range(n, 0, -1):
        if first > n:
            continue
        remaining = n - first
        if remaining < 0:
            continue
        if max_parts - 1 <= 0:
            if remaining == 0:
                yield [first]
            continue
        for rest in _get_partitions(remaining, max_parts - 1):
            yield [first] + rest

def _compute_block_rank(m_b, p_b):
    """Compute the rank for a (2, m_b, p_b) block."""
    if m_b == 2 and p_b == 2:
        return 7  # Strassen
    return 2 * m_b * p_b  # Schoolbook

def _compute_total_rank(partition_j, partition_k):
    """Compute total rank for given partitions."""
    total = 0
    for jb in partition_j:
        for kb in partition_k:
            total += _compute_block_rank(jb, kb)
    return total

def _build_decomposition(partition_j, partition_k, n, m, p):
    """Build the full decomposition from block partitions."""
    # Create J and K blocks
    J_blocks = []
    pos = 0
    for sz in partition_j:
        J_blocks.append(list(range(pos, pos + sz)))
        pos += sz
    
    K_blocks = []
    pos = 0
    for sz in partition_k:
        K_blocks.append(list(range(pos, pos + sz)))
        pos += sz
    
    # Compute total rank
    total_rank = 0
    for J in J_blocks:
        m_b = len(J)
        for K in K_blocks:
            p_b = len(K)
            total_rank += _compute_block_rank(m_b, p_b)
    
    # Build final matrices
    U_final = [[0] * total_rank for _ in range(n * m)]
    V_final = [[0] * total_rank for _ in range(m * p)]
    W_final = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for J in J_blocks:
        m_b = len(J)
        for K in K_blocks:
            p_b = len(K)
            n_b = 2
            if m_b == 2 and p_b == 2:
                U_b, V_b, W_b = _strassen_222()
            else:
                U_b, V_b, W_b = _schoolbook_block(n_b, m_b, p_b)
            block_rank = len(U_b[0])
            
            for i in range(n_b):
                for j_local in range(m_b):
                    global_j = J[j_local]
                    u_global_idx = i * m + global_j
                    u_local_idx = i * m_b + j_local
                    for c in range(block_rank):
                        U_final[u_global_idx][col_offset + c] = U_b[u_local_idx][c]
            
            for j_local in range(m_b):
                global_j = J[j_local]
                for k_local in range(p_b):
                    global_k = K[k_local]
                    v_global_idx = global_j * p + global_k
                    v_local_idx = j_local * p_b + k_local
                    for c in range(block_rank):
                        V_final[v_global_idx][col_offset + c] = V_b[v_local_idx][c]
            
            for k_local in range(p_b):
                global_k = K[k_local]
                for i in range(n_b):
                    w_global_idx = global_k * n + i
                    w_local_idx = k_local * n_b + i
                    for c in range(block_rank):
                        W_final[w_global_idx][col_offset + c] = W_b[w_local_idx][c]
            
            col_offset += block_rank
    
    return U_final, V_final, W_final

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iter=500):
    """
    Try to reduce the rank by attempting to find a decomposition with fewer terms.
    This uses a randomized approach to try to find a valid decomposition.
    """
    current_rank = len(U[0])
    if target_rank >= current_rank:
        return U, V, W, current_rank
    
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to find a lower-rank decomposition using iterative methods
    # This is a heuristic approach - we'll try to zero out some columns
    # and check if the decomposition still holds
    
    # For now, we'll implement a simple approach: try to find linear dependencies
    # in the columns and eliminate them
    
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Try to reduce by 1 at a time
    for target in range(current_rank - 1, target_rank - 1, -1):
        if _time_left() < 5:
            break
        if target < 1:
            break
        
        # Try to find a subset of 'target' columns that still works
        # This is NP-hard in general, so we use a heuristic
        
        # Simple heuristic: try removing one column at a time
        # and checking if we can adjust the remaining columns to compensate
        
        # For efficiency, we'll just try a few random subsets
        np.random.seed(42)
        found = False
        for attempt in range(100):
            if _time_left() < 5:
                break
            
            # Randomly select 'target' columns
            cols = np.random.choice(current_rank, target, replace=False)
            cols = np.sort(cols)
            
            U_sub = U_arr[:, cols]
            V_sub = V_arr[:, cols]
            W_sub = W_arr[:, cols]
            
            # Check if this subset still gives a valid decomposition
            if _verify(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), n, m, p):
                best_U = U_sub.tolist()
                best_V = V_sub.tolist()
                best_W = W_sub.tolist()
                best_rank = target
                found = True
                break
        
        if found:
            continue
        else:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Find the best partition
    best_rank = 999
    best_partition_j = None
    best_partition_k = None
    
    # Generate partitions of 5 and 6
    parts5 = list(_get_partitions(5, 4))
    parts6 = list(_get_partitions(6, 4))
    
    for pj in parts5:
        for pk in parts6:
            r = _compute_total_rank(pj, pk)
            if r < best_rank:
                best_rank = r
                best_partition_j = pj
                best_partition_k = pk
    
    # Build the decomposition
    U, V, W = _build_decomposition(best_partition_j, best_partition_k, n, m, p)
    current_rank = best_rank
    
    # Try to reduce the rank further
    if _time_left() > 10:
        U, V, W, current_rank = _try_rank_reduction(U, V, W, n, m, p, max(1, current_rank - 5))
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to full schoolbook
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

# EVOLVE-BLOCK-END
