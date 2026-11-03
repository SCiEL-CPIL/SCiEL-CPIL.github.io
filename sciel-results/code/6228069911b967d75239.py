"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _rational_solve(A, b):
    """Solve linear system Ax = b over rationals using Gaussian elimination.
    Returns solution vector or None if no unique solution.
    A is (m, n) matrix of Fractions, b is (m,) vector of Fractions.
    """
    m, n = A.shape
    # Augmented matrix
    M = [list(A[i]) + [b[i]] for i in range(m)]
    
    # Gaussian elimination with partial pivoting
    for col in range(min(m, n)):
        # Find pivot
        pivot_row = None
        for row in range(col, m):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            return None  # No unique solution
        
        # Swap rows
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Eliminate below
        pivot_val = M[col][col]
        for row in range(col + 1, m):
            factor = M[row][col] / pivot_val
            for j in range(col, m + 1):
                M[row][j] -= factor * M[col][j]
    
    # Back substitution
    x = [Fraction(0)] * n
    for row in range(min(m, n) - 1, -1, -1):
        # Check if this row is all zeros in coefficient part
        all_zero = True
        for j in range(row, n):
            if M[row][j] != 0:
                all_zero = False
                break
        if all_zero:
            if M[row][n] != 0:
                return None  # Inconsistent
            continue
        
        s = M[row][n]
        for j in range(row + 1, n):
            s -= M[row][j] * x[j]
        x[row] = s / M[row][row]
    
    # Verify solution
    for i in range(m):
        s = Fraction(0)
        for j in range(n):
            s += A[i][j] * x[j]
        if s != b[i]:
            return None
    
    return x

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
            # Standard Strassen 2x2x2 block
            # M1 = (A00 + A11)(B00 + B11)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # M3 = A00(B10 - B00)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M4 = A11(B01 - B11)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
    
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _try_structured_linear_solve(n, m, p, max_rank=54):
    """
    Try to find a lower-rank decomposition by fixing U and V to sparse {0, ±1} 
    patterns and solving for W over rationals.
    
    Uses numpy float64 for speed, then verifies with exact check.
    Returns (U, V, W, rank) if successful, else None.
    """
    # Get the base 54-rank decomposition
    U_base, V_base, W_base = _build_54_rank(n, m, p)
    base_rank = U_base.shape[1]
    
    # Try removing 1 or 2 terms and solving for new W
    for n_remove in [1, 2]:
        if _time_left() < 15:
            break
        
        target_rank = base_rank - n_remove
        
        if n_remove == 1:
            candidates = [tuple([t]) for t in range(base_rank)]
        else:
            # Try pairs of terms that share structure
            candidates = [tuple(t) for t in itertools.combinations(range(base_rank), 2)]
            # Limit to first 200 candidates for time
            candidates = candidates[:200]
        
        for remove_set in candidates:
            if _time_left() < 8:
                break
            
            keep_mask = np.ones(base_rank, dtype=bool)
            for r in remove_set:
                keep_mask[r] = False
            
            U_test = U_base[:, keep_mask].astype(float)
            V_test = V_base[:, keep_mask].astype(float)
            
            # Build linear system: for each (x,y,z) constraint, sum_r U[y,r]*V[z,r]*W[x,r] = target
            # Unknowns: W[x,r] for x in range(p*n), r in range(target_rank)
            n_unknowns = p * n * target_rank
            n_eqs = 0
            
            # Collect equations
            A_rows = []
            b_vals = []
            
            # Valid equations: T[k*n+i, i*m+j, j*p+k] = 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        x = k * n + i
                        y = i * m + j
                        z = j * p + k
                        row = np.zeros(n_unknowns)
                        for r in range(target_rank):
                            row[x * target_rank + r] = U_test[y, r] * V_test[z, r]
                        A_rows.append(row)
                        b_vals.append(1.0)
            
            # Invalid equations: sample to constrain W
            # Only include equations where the coefficient vector is non-zero
            invalid_count = 0
            max_invalid = 300
            for x in range(p*n):
                if invalid_count >= max_invalid:
                    break
                i = x % n
                k = x // n
                for y in range(n*m):
                    if invalid_count >= max_invalid:
                        break
                    i2 = y // m
                    j = y % m
                    if i != i2:
                        # All z are invalid
                        for z in range(m*p):
                            if invalid_count >= max_invalid:
                                break
                            row = np.zeros(n_unknowns)
                            for r in range(target_rank):
                                c = U_test[y, r] * V_test[z, r]
                                if c != 0:
                                    row[x * target_rank + r] = c
                            if np.any(row != 0):
                                A_rows.append(row)
                                b_vals.append(0.0)
                                invalid_count += 1
                    else:
                        for z in range(m*p):
                            if invalid_count >= max_invalid:
                                break
                            j2 = z // p
                            k2 = z % p
                            if j == j2 and k == k2:
                                continue
                            row = np.zeros(n_unknowns)
                            for r in range(target_rank):
                                c = U_test[y, r] * V_test[z, r]
                                if c != 0:
                                    row[x * target_rank + r] = c
                            if np.any(row != 0):
                                A_rows.append(row)
                                b_vals.append(0.0)
                                invalid_count += 1
            
            if not A_rows:
                continue
            
            A_mat = np.array(A_rows)
            b_vec = np.array(b_vals)
            
            # Solve least squares
            try:
                solution, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except Exception:
                continue
            
            # Check if solution is good
            if rank < n_unknowns:
                # Underdetermined - check if residual is small
                pass
            
            residual = np.linalg.norm(A_mat @ solution - b_vec) / max(1, np.linalg.norm(b_vec))
            if residual > 1e-8:
                continue
            
            # Reconstruct W
            W_test = np.zeros((p*n, target_rank))
            for x in range(p*n):
                for r in range(target_rank):
                    W_test[x, r] = solution[x * target_rank + r]
            
            # Verify with full check
            if _verify_full(U_test, V_test, W_test, n, m, p, tol=1e-6):
                return U_test.copy(), V_test.copy(), W_test.copy(), target_rank
    
    return None

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact using vectorized numpy operations.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Build the full tensor T[x,y,z] = sum_r U[y,r] * V[z,r] * W[x,r]
    # T has shape (p*n, n*m, m*p)
    # T[x,y,z] = sum_r W[x,r] * U[y,r] * V[z,r]
    # We can compute this as: for each r, outer(W[:,r], U[:,r]) * V[:,r] broadcast
    
    # Efficient: compute T = sum_r np.einsum('xr,yr,zr->xyz', W, U, V)
    T = np.einsum('xr,yr,zr->xyz', W, U, V)
    
    # Check valid entries: T[k*n+i, i*m+j, j*p+k] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[k*n+i, i*m+j, j*p+k] - 1.0) > tol:
                    return False
    
    # Check invalid entries
    # Valid triple: x=k*n+i, y=i*m+j, z=j*p+k means i==x%n, k==x//n, j==y%m, i2==y//m, j2==z//p, k2==z%p
    # Valid iff i==i2, j==j2, k==k2
    # i.e., x%n == y//m, y%m == z//p, x//n == z%p
    
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            i2 = y // m
            j = y % m
            if i != i2:
                # All z are invalid
                if np.any(np.abs(T[x, y, :]) > tol):
                    return False
            else:
                for z in range(m*p):
                    j2 = z // p
                    k2 = z % p
                    if j == j2 and k == k2:
                        continue
                    if abs(T[x, y, z]) > tol:
                        return False
    return True

def _verify_skip_term(U, V, W, n, m, p, skip_term, tol=1e-9):
    """
    Verify that the decomposition remains valid if term skip_term is set to zero.
    Uses vectorized computation for speed.
    """
    rank = U.shape[1]
    
    # Compute T_full and T_skip using einsum
    T_full = np.einsum('xr,yr,zr->xyz', W, U, V)
    
    # T_skip = T_full - W[:,skip_term] * U[:,skip_term] * V[:,skip_term]
    T_skip = T_full - np.einsum('x,y,z->xyz', W[:, skip_term], U[:, skip_term], V[:, skip_term])
    
    # Check valid entries: T_skip[k*n+i, i*m+j, j*p+k] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T_skip[k*n+i, i*m+j, j*p+k] - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            i2 = y // m
            j = y % m
            if i != i2:
                if np.any(np.abs(T_skip[x, y, :]) > tol):
                    return False
            else:
                for z in range(m*p):
                    j2 = z // p
                    k2 = z % p
                    if j == j2 and k == k2:
                        continue
                    if abs(T_skip[x, y, z]) > tol:
                        return False
    return True

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminations=20):
    """
    Try to eliminate redundant terms one by one.
    Uses iterative greedy approach with vectorized verification.
    Returns (U, V, W, new_rank).
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Precompute the full tensor once
    T = np.einsum('xr,yr,zr->xyz', current_W, current_U, current_V)
    
    # Greedy elimination: repeatedly try to remove terms
    removed = []
    while len(removed) < max_eliminations and _time_left() > 30:
        found = False
        for r in range(current_rank):
            if r in removed:
                continue
            
            # Check if removing term r still satisfies constraints
            # T_without_r = T - W[:,r]*U[:,r]*V[:,r]
            term_tensor = np.einsum('x,y,z->xyz', current_W[:, r], current_U[:, r], current_V[:, r])
            T_reduced = T - term_tensor
            
            # Check valid entries
            valid_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(T_reduced[k*n+i, i*m+j, j*p+k] - 1.0) > 1e-9:
                            valid_ok = False
                            break
                    if not valid_ok:
                        break
                if not valid_ok:
                    break
            
            if not valid_ok:
                continue
            
            # Check invalid entries (only need to check where term_tensor is non-zero)
            # For efficiency, only check entries where T_reduced might be non-zero
            invalid_ok = True
            nz_mask = np.abs(term_tensor) > 1e-12
            if np.any(nz_mask):
                idx = np.argwhere(nz_mask)
                for x, y, z in idx:
                    i = x % n
                    k = x // n
                    i2 = y // m
                    j = y % m
                    j2 = z // p
                    k2 = z % p
                    if i == i2 and j == j2 and k == k2:
                        continue
                    if abs(T_reduced[x, y, z]) > 1e-9:
                        invalid_ok = False
                        break
            
            if invalid_ok:
                # Remove this term
                removed.append(r)
                T = T_reduced
                found = True
                break
        
        if not found:
            break
    
    # Build final matrices
    if removed:
        keep_mask = np.ones(current_rank, dtype=bool)
        for r in removed:
            keep_mask[r] = False
        current_U = current_U[:, keep_mask]
        current_V = current_V[:, keep_mask]
        current_W = current_W[:, keep_mask]
        current_rank = int(np.sum(keep_mask))
    
    return current_U, current_V, current_W, current_rank

def _try_pairwise_elimination(U, V, W, n, m, p):
    """
    Try to eliminate pairs of terms simultaneously.
    Uses vectorized tensor computation for speed.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Precompute full tensor
    T = np.einsum('xr,yr,zr->xyz', current_W, current_U, current_V)
    
    # Try all pairs
    for r1 in range(current_rank):
        if _time_left() < 20:
            break
        for r2 in range(r1 + 1, current_rank):
            # T without terms r1 and r2
            term1 = np.einsum('x,y,z->xyz', current_W[:, r1], current_U[:, r1], current_V[:, r1])
            term2 = np.einsum('x,y,z->xyz', current_W[:, r2], current_U[:, r2], current_V[:, r2])
            T_reduced = T - term1 - term2
            
            # Check valid entries
            valid_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(T_reduced[k*n+i, i*m+j, j*p+k] - 1.0) > 1e-9:
                            valid_ok = False
                            break
                    if not valid_ok:
                        break
                if not valid_ok:
                    break
            
            if not valid_ok:
                continue
            
            # Check invalid entries where the removed terms had support
            combined = term1 + term2
            nz_mask = np.abs(combined) > 1e-12
            if np.any(nz_mask):
                invalid_ok = True
                idx = np.argwhere(nz_mask)
                for x, y, z in idx:
                    i = x % n
                    k = x // n
                    i2 = y // m
                    j = y % m
                    j2 = z // p
                    k2 = z % p
                    if i == i2 and j == j2 and k == k2:
                        continue
                    if abs(T_reduced[x, y, z]) > 1e-9:
                        invalid_ok = False
                        break
                if not invalid_ok:
                    continue
            
            # Success! Remove both
            mask = np.ones(current_rank, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 2
            return current_U, current_V, current_W, current_rank
    
    return current_U, current_V, current_W, current_rank

def _try_linear_combination_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by finding linear dependencies among the terms.
    For each pair (r1, r2), check if we can replace them with a single term
    by finding alpha such that the combined contribution works.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Precompute tensor
    T = np.einsum('xr,yr,zr->xyz', current_W, current_U, current_V)
    
    # For each pair, try to find if one is a scalar multiple of the other
    # (in all three matrices), which would mean redundancy
    for r1 in range(current_rank):
        if _time_left() < 15:
            break
        for r2 in range(r1 + 1, current_rank):
            u1, u2 = current_U[:, r1], current_U[:, r2]
            v1, v2 = current_V[:, r1], current_V[:, r2]
            w1, w2 = current_W[:, r1], current_W[:, r2]
            
            # Check proportionality
            if np.linalg.norm(u1) < 1e-10 or np.linalg.norm(u2) < 1e-10:
                continue
            
            # Try to find alpha = <u1,u2>/<u2,u2>
            alpha = np.dot(u1, u2) / np.dot(u2, u2)
            
            if np.allclose(u1, alpha * u2, atol=1e-8) and \
               np.allclose(v1, alpha * v2, atol=1e-8) and \
               np.allclose(w1, alpha * w2, atol=1e-8):
                # Remove r1 (it's redundant with r2)
                mask = np.ones(current_rank, dtype=bool)
                mask[r1] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                # Rebuild tensor
                T = np.einsum('xr,yr,zr->xyz', current_W, current_U, current_V)
                break
    
    return current_U, current_V, current_W, current_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_full(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try to eliminate redundant terms (single) - do this first as it's faster
    if _time_left() > 50:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
                U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try to eliminate pairs of terms
    if _time_left() > 30:
        U_reduced, V_reduced, W_reduced, new_rank = _try_pairwise_elimination(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
                U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try linear combination reduction
    if _time_left() > 20:
        U_reduced, V_reduced, W_reduced, new_rank = _try_linear_combination_reduction(U, V, W, n, m, p)
        if new_rank < best_rank:
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
                U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try structured linear algebra approach (fix U,V, solve for W) - slower, do last
    if _time_left() > 40:
        result = _try_structured_linear_solve(n, m, p)
        if result is not None:
            U_sol, V_sol, W_sol, new_rank = result
            if new_rank < best_rank:
                if _verify_full(U_sol, V_sol, W_sol, n, m, p):
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = new_rank
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
