"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _tensor_check(U, V, W, n, m, p, tol=1e-9):
    """Fast tensor check: returns True if decomposition is valid."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    # Check a sample of invalid triples for speed
    checked = 0
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = sum(U[y, r] * V[z, r] * W[x, r] for r in range(rank))
                if abs(s) > tol:
                    return False
                checked += 1
                if checked > 5000:
                    return True
    return True

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries
    # Total entries: (p*n) * (n*m) * (m*p) = 12 * 10 * 30 = 3600
    # Rank is ~50, so 3600 * 50 = 180k ops. Very fast.
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                # Check consistency
                if i == i2 and j == j2 and k == k2:
                    # Valid case, already checked
                    continue
                else:
                    s = 0.0
                    for r in range(rank):
                        s += U[y, r] * V[z, r] * W[x, r]
                    if abs(s) > tol:
                        return False
    return True

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 products for Strassen 2x2x2 block
            
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        # C_1,k = A_1,4 B_4,k
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _try_eliminate_term(U, V, W, n, m, p, term_idx, tol=1e-9):
    """
    Try to eliminate term_idx by expressing it as a combination of other terms.
    Returns (new_U, new_V, new_W) if successful, None otherwise.
    """
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    # Remove the term and check if remaining terms still span the tensor
    # Strategy: try to find coefficients c_r such that
    # sum_{r != term_idx} c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = U[:,term_idx] ⊗ V[:,term_idx] ⊗ W[:,term_idx]
    
    # This is a linear system. Build it by flattening.
    # The tensor T has size (n*m) * (m*p) * (p*n)
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor (flattened)
    target = np.outer(np.outer(U[:, term_idx], V[:, term_idx]), W[:, term_idx]).flatten()
    
    # Build matrix A where column r is the flattened tensor for term r
    # A shape: (nm*mp*pn, rank-1)
    A_cols = []
    for r in range(rank):
        if r == term_idx:
            continue
        A_cols.append(np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten())
    
    A = np.column_stack(A_cols)
    
    # Solve A @ coeffs = target
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, target, rcond=1e-10)
    except:
        return None
    
    # Check if solution is exact
    if len(residuals) > 0 and residuals[0] > tol:
        return None
    
    # Check if coefficients are close to half-integers
    # If not, we might still accept if they're rational with small denominator
    max_coeff = np.max(np.abs(coeffs))
    if max_coeff > 10:
        return None
    
    # Create new matrices without the eliminated term
    new_rank = rank - 1
    keep = [r for r in range(rank) if r != term_idx]
    
    new_U = U[:, keep] * coeffs.reshape(1, -1)
    new_V = V[:, keep]
    new_W = W[:, keep]
    
    # Verify
    if _tensor_check(new_U, new_V, new_W, n, m, p, tol):
        return (new_U, new_V, new_W)
    
    return None

def _greedy_elimination(U, V, W, n, m, p, max_rounds=10, tol=1e-9):
    """
    Greedily try to eliminate terms one at a time.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    for round_num in range(max_rounds):
        if _time_left() < 5:
            break
        
        rank = best_U.shape[1]
        if rank <= 1:
            break
        
        improved = False
        for term_idx in range(rank):
            if _time_left() < 3:
                break
            
            result = _try_eliminate_term(best_U, best_V, best_W, n, m, p, term_idx, tol)
            if result is not None:
                new_U, new_V, new_W = result
                if new_U.shape[1] < best_rank:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = new_U.shape[1]
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_U, best_V, best_W

def _snap_to_half_integer(x, tol=1e-6):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _snap_matrix(M, tol=1e-6):
    """Snap all entries of matrix to nearest half-integer if close."""
    return np.array([[_snap_to_half_integer(x) for x in row] for row in M])

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Full verification of decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                else:
                    s = sum(U[y, r] * V[z, r] * W[x, r] for r in range(rank))
                    if abs(s) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank 54 solution
    base_sol = _build_54_rank(n, m, p)
    base_U, base_V, base_W = base_sol["U"], base_sol["V"], base_sol["W"]
    
    # Verify base solution
    if not _verify_decomposition(base_U, base_V, base_W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = base_U.shape[1]
    best_U, best_V, best_W = base_U.copy(), base_V.copy(), base_W.copy()
    
    # Try greedy elimination to reduce rank
    if _time_left() > 20:
        elim_U, elim_V, elim_W = _greedy_elimination(base_U, base_V, base_W, n, m, p, max_rounds=15)
        if elim_U.shape[1] < best_rank:
            best_U, best_V, best_W = elim_U, elim_V, elim_W
            best_rank = elim_U.shape[1]
    
    # Try to find even lower rank by trying different elimination orders
    if _time_left() > 15 and best_rank > 40:
        # Try eliminating from different starting points
        for start_term in range(min(base_U.shape[1], 10)):
            if _time_left() < 5:
                break
            # Create a modified version where we force-eliminate start_term first
            # by scaling it to zero and then running elimination
            mod_U = base_U.copy()
            mod_V = base_V.copy()
            mod_W = base_W.copy()
            # Just try elimination with different orderings
            result = _greedy_elimination(mod_U, mod_V, mod_W, n, m, p, max_rounds=5)
            if result[0].shape[1] < best_rank:
                best_U, best_V, best_W = result
                best_rank = best_U.shape[1]
    
    # Snap to half-integers if possible
    U_snap = _snap_matrix(best_U)
    V_snap = _snap_matrix(best_V)
    W_snap = _snap_matrix(best_W)
    
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
