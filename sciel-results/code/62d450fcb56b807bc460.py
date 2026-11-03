"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 130.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _compute_tensor_error(U, V, W, n, m, p):
    """Compute the maximum error of the tensor decomposition using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    # Check valid entries: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(s - 1.0))
    
    # Check invalid entries: sum_r U[y,r] * V[z,r] * W[x,r] == 0
    # where (i,j,k) doesn't match (i2,j2,k2)
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z % p
                k2 = z // p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = np.dot(U[y, :], V[z, :] * W[x, :])
                max_err = max(max_err, abs(s))
    
    return max_err

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    return _compute_tensor_error(U, V, W, n, m, p) < tol

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
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _snap_to_half_integer(x, tol=1e-6):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminations=3):
    """
    Try to eliminate redundant terms by checking if a term's contribution
    is in the span of other terms' contributions.
    Uses the fact that each term contributes a rank-1 tensor.
    """
    if _time_left() < 10:
        return None
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    current_rank = U_work.shape[1]
    
    eliminated = 0
    cols_to_remove = []
    
    for target_rank in range(current_rank - 1, max(current_rank - max_eliminations - 1, 0) - 1, -1):
        if _time_left() < 10:
            break
        
        # Try to find a column that can be eliminated
        found = False
        for r in range(current_rank - 1, -1, -1):
            if _time_left() < 10:
                break
            
            # Skip if this column is already marked for removal
            if r in cols_to_remove:
                continue
            
            # Check if term r's contribution is in the span of others
            # Build the contribution vector for term r
            # c_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
            
            # Instead of building the full tensor, check linear dependence
            # of the outer products of the three vectors
            
            u_vec = U_work[:, r].copy()
            v_vec = V_work[:, r].copy()
            w_vec = W_work[:, r].copy()
            
            # Check if u_vec, v_vec, w_vec are all non-zero
            if np.allclose(u_vec, 0) or np.allclose(v_vec, 0) or np.allclose(w_vec, 0):
                # This term is already zero, can be removed
                cols_to_remove.append(r)
                eliminated += 1
                found = True
                break
            
            # Try to express u_vec as combination of other u_vecs
            # This is expensive, so use a heuristic: check if the term
            # has small norm relative to others
            
            norm_r = np.linalg.norm(u_vec) * np.linalg.norm(v_vec) * np.linalg.norm(w_vec)
            
            # Compute average norm of other terms
            other_norms = []
            for s in range(current_rank):
                if s == r or s in cols_to_remove:
                    continue
                norm_s = np.linalg.norm(U_work[:, s]) * np.linalg.norm(V_work[:, s]) * np.linalg.norm(W_work[:, s])
                other_norms.append(norm_s)
            
            if other_norms:
                avg_norm = np.mean(other_norms)
                if norm_r < 1e-10 * avg_norm:
                    cols_to_remove.append(r)
                    eliminated += 1
                    found = True
                    break
        
        if not found:
            break
        
        # Remove marked columns
        if cols_to_remove:
            keep_cols = [c for c in range(current_rank) if c not in cols_to_remove]
            U_work = U_work[:, keep_cols]
            V_work = V_work[:, keep_cols]
            W_work = W_work[:, keep_cols]
            current_rank = len(keep_cols)
            cols_to_remove = []
        
        # Verify the reduced decomposition
        if _verify_decomposition(U_work, V_work, W_work, n, m, p):
            return {"U": U_work, "V": V_work, "W": W_work}
        else:
            # Revert
            break
    
    return None

def _try_gauge_fixing(U, V, W, n, m, p):
    """
    Try to reduce rank by gauge fixing: for each term, we can scale
    U[:,r] by alpha, V[:,r] by beta, W[:,r] by gamma as long as
    alpha * beta * gamma = 1. Use this to normalize and check
    for linear dependencies.
    """
    if _time_left() < 15:
        return None
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    current_rank = U_work.shape[1]
    
    # Normalize each term so that ||U[:,r]||_2 = 1
    for r in range(current_rank):
        norm_u = np.linalg.norm(U_work[:, r])
        if norm_u > 1e-10:
            U_work[:, r] /= norm_u
            V_work[:, r] *= norm_u
    
    # Now check for linear dependencies in the U columns
    # If U[:, r] is in the span of other U columns, we might be able to eliminate
    U_mat = U_work.copy()
    
    for r in range(current_rank - 1, -1, -1):
        if _time_left() < 10:
            break
        
        # Check if U[:, r] is in the span of other columns
        other_cols = [c for c in range(current_rank) if c != r]
        if not other_cols:
            break
        
        U_other = U_mat[:, other_cols]
        u_r = U_mat[:, r]
        
        # Solve least squares: U_other @ x = u_r
        try:
            x, residuals, rank, sv = np.linalg.lstsq(U_other, u_r, rcond=1e-10)
            if len(residuals) > 0 and residuals[0] < 1e-8 * np.linalg.norm(u_r):
                # u_r is in the span of other columns
                # Check if we can also adjust V and W accordingly
                # This is complex, so just mark for potential removal
                pass
        except:
            pass
    
    # For now, just return the normalized version
    # The actual elimination would require more careful handling
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank 54 solution
    base_sol = _build_54_rank(n, m, p)
    best_result = base_sol
    best_rank = 54
    
    # Verify the base solution
    if not _verify_decomposition(base_sol["U"], base_sol["V"], base_sol["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by eliminating redundant terms
    if _time_left() > 20:
        result_reduced = _try_eliminate_terms(base_sol["U"], base_sol["V"], base_sol["W"], n, m, p)
        if result_reduced is not None:
            new_rank = result_reduced["U"].shape[1]
            if new_rank < best_rank:
                best_result = result_reduced
                best_rank = new_rank
    
    # Try gauge fixing
    if _time_left() > 15:
        result_gauge = _try_gauge_fixing(base_sol["U"], base_sol["V"], base_sol["W"], n, m, p)
        if result_gauge is not None:
            new_rank = result_gauge["U"].shape[1]
            if new_rank < best_rank:
                best_result = result_gauge
                best_rank = new_rank
    
    # Snap to half-integers
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    
    U_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
    V_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
    W_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
    
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
