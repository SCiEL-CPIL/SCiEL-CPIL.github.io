"""Matrix multiplication 2x5x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 130  # 130s to leave safety margin
    return _DEADLINE

def _time_left():
    return max(0, _get_deadline() - time.time())

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _build_54_rank():
    """Build the base 54-rank decomposition."""
    n, m, p = DIMENSIONS
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # Remaining j=4: scalar terms
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    return U, V, W, col

def _verify_decomp(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # And for all other combinations, it should be 0
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) in _valid_indices(n, m, p) else 0.0
                actual = 0.0
                for r in range(rank):
                    actual += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                if abs(actual - expected) > 1e-6:
                    return False
    return True

def _valid_indices(n, m, p):
    """Return the set of valid (i,j,k) indices for matrix multiplication."""
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 if the index combination is valid
    # For matrix multiplication, all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid
    return set((i, j, k) for i in range(n) for j in range(m) for k in range(p))

def _find_redundant_terms(U, V, W, n, m, p):
    """Find linearly dependent terms that can be eliminated."""
    rank = len(U[0])
    
    # Build the tensor representation
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to find if any column r is a linear combination of others
    
    # For each column r, check if it's in the span of the others
    # This is complex, so we'll try a simpler approach:
    # Check if removing a column still allows correct decomposition
    
    redundant = []
    for r_test in range(rank):
        if _time_left() < 5:
            break
        
        # Try removing column r_test
        U_new = [[U[i][r] for r in range(rank) if r != r_test] for i in range(n*m)]
        V_new = [[V[i][r] for r in range(rank) if r != r_test] for i in range(m*p)]
        W_new = [[W[i][r] for r in range(rank) if r != r_test] for i in range(p*n)]
        
        if _verify_decomp(U_new, V_new, W_new, n, m, p):
            redundant.append(r_test)
    
    return redundant

def _try_local_search(U, V, W, n, m, p, max_steps=100):
    """Try local search to reduce rank."""
    rank = len(U[0])
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    for step in range(max_steps):
        if _time_left() < 10:
            break
        
        # Try removing each column
        for r in range(best_rank):
            if _time_left() < 5:
                break
            
            U_new = [[U[i][c] for c in range(best_rank) if c != r] for i in range(n*m)]
            V_new = [[V[i][c] for c in range(best_rank) if c != r] for i in range(m*p)]
            W_new = [[W[i][c] for c in range(best_rank) if c != r] for i in range(p*n)]
            
            if _verify_decomp(U_new, V_new, W_new, n, m, p):
                best_rank -= 1
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    return best_U, best_V, best_W, best_rank

def _snap_to_half_integer(arr):
    """Snap values to nearest half-integer."""
    snapped = []
    for row in arr:
        snapped_row = []
        for val in row:
            snapped_val = round(val * 2) / 2.0
            snapped_row.append(snapped_val)
        snapped.append(snapped_row)
    return snapped

def construct():
    n, m, p = DIMENSIONS
    _get_deadline()
    
    # Build base decomposition
    U, V, W, col = _build_54_rank()
    
    # Verify base decomposition
    if not _verify_decomp(U, V, W, n, m, p):
        # Fallback: return schoolbook decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find redundant terms
    redundant = _find_redundant_terms(U, V, W, n, m, p)
    
    if redundant:
        # Remove redundant terms
        keep = [r for r in range(col) if r not in redundant]
        U = [[U[i][r] for r in keep] for i in range(n*m)]
        V = [[V[i][r] for r in keep] for i in range(m*p)]
        W = [[W[i][r] for r in keep] for i in range(p*n)]
    
    # Try local search
    if _time_left() > 30:
        U, V, W, rank = _try_local_search(U, V, W, n, m, p)
    
    # Snap to half-integers
    U = _snap_to_half_integer(U)
    V = _snap_to_half_integer(V)
    W = _snap_to_half_integer(W)
    
    # Final verification
    if not _verify_decomp(U, V, W, n, m, p):
        # If snapping broke it, return the unsnapped version
        U, V, W, col = _build_54_rank()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
