"""Rewritten program for 4x4 @ 4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    # M2 = (A21+A22)B11
    terms.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}})
    # M3 = A11(B21-B22)
    terms.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}})
    # M4 = A22(B12-B11)
    terms.append({'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}})
    # M5 = (A11+A12)B22
    terms.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}})
    # M6 = (A21-A11)(B11+B12)
    terms.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
    # M7 = (A12-A22)(B21+B22)
    terms.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}})
    return terms

def _build_4x4x4_strassen():
    """Build a 4x4x4 decomposition using recursive Strassen (rank 49)."""
    n = 4
    terms_2x2 = _strassen_2x2()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            for (bi, bj), coeff_u in term1['U'].items():
                for (li, lj), coeff_v in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i*4 + j][idx] += coeff_u * coeff_v
            
            for (bj, bk), coeff_v in term1['V'].items():
                for (lj, lk), coeff_w in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j*4 + k][idx] += coeff_v * coeff_w
            
            for (bk, bi), coeff_w in term1['W'].items():
                for (lk, li), coeff_u2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k*4 + i][idx] += coeff_w * coeff_u2
    
    return {"U": U, "V": V, "W": W}

def _build_4x4x5_strassen():
    """Build a 4x4x5 decomposition: 4x4x4 (rank 49) + 4x4x1 (rank 4) = rank 53."""
    n, m, p = DIMENSIONS
    
    base = _build_4x4x4_strassen()
    rank_base = 49
    
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy 4x4x4 part (handles k=0..3)
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m):
        for k in range(4):
            for r in range(rank_base):
                V[j*5 + k][r] = base["V"][j*4 + k][r]
    for k in range(4):
        for i in range(n):
            for r in range(rank_base):
                W[k*5 + i][r] = base["W"][k*4 + i][r]
    
    # Add 4x4x1 part: C[i,4] = sum_j A[i,j] * B[j,4] for i=0..3
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i*m + j][r] = 1
        V[j*p + 4][r] = 1
        for i in range(n):
            W[4*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor: R[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1"""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    R = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i,j,k] = np.dot(U_np[i*m+j, :] * V_np[j*p+k, :], W_np[k*n+i, :]) - 1.0
    return R

def _try_eliminate_term(U, V, W, n, m, p, term_idx, time_budget=20):
    """Try to eliminate term_idx by re-optimizing the remaining terms.
    
    Fix all other terms, and see if we can adjust them to compensate for removing term_idx.
    This is a constrained optimization: minimize ||T - T_target|| subject to the constraint.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Create a copy without the target term
    keep = [r for r in range(rank) if r != term_idx]
    U_keep = [row[keep] for row in U]
    V_keep = [row[keep] for row in V]
    W_keep = [row[keep] for row in W]
    
    # Convert to numpy
    U_np = np.array(U_keep, dtype=np.float64)
    V_np = np.array(V_keep, dtype=np.float64)
    W_np = np.array(W_keep, dtype=np.float64)
    
    # We need to find adjustments to U_keep, V_keep, W_keep such that
    # the decomposition still holds. This is a nonlinear problem.
    # Use a simple iterative approach: for each term, try to adjust it.
    
    # Actually, a better approach: use the fact that we can add small perturbations
    # to the remaining terms to compensate.
    
    # Compute the contribution of the removed term
    U_removed = np.array([U[i][term_idx] for i in range(n*m)])
    V_removed = np.array([V[i][term_idx] for i in range(m*p)])
    W_removed = np.array([W[i][term_idx] for i in range(p*n)])
    
    # The removed term contributes: U_removed[i*m+j] * V_removed[j*p+k] * W_removed[k*n+i]
    # We need the remaining terms to compensate for this.
    
    # This is complex. Instead, try a simpler approach:
    # Check if the removed term is "redundant" - i.e., its contribution can be
    # expressed as a linear combination of other terms' contributions.
    
    # For each (i,j,k), the contribution of term r is:
    # C_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We want to see if C_removed can be written as sum_{r != removed} a_r * C_r
    # This is a linear system in a_r.
    
    # Build the system: for each (i,j,k), sum_{r != removed} a_r * C_r[i,j,k] = C_removed[i,j,k]
    
    num_constraints = n * m * p  # 80
    num_unknowns = len(keep)  # rank - 1
    
    # Build constraint matrix
    A_mat = np.zeros((num_constraints, num_unknowns))
    b_vec = np.zeros(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for u_idx, r in enumerate(keep):
                    A_mat[idx, u_idx] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                b_vec[idx] = U[i*m+j][term_idx] * V[j*p+k][term_idx] * W[k*n+i][term_idx]
                idx += 1
    
    # Solve the linear system
    try:
        a_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
    except np.linalg.LinAlgError:
        return None
    
    # Check if the solution is valid (residuals should be small)
    residual_norm = np.linalg.norm(A_mat @ a_sol - b_vec)
    if residual_norm > 1e-6:
        return None
    
    # The solution a_sol gives us the coefficients. But we need to update
    # U, V, W such that the new terms reproduce the same contributions.
    # This is not straightforward because the contributions are products.
    
    # Alternative: just check if we can set the removed term to zero and
    # adjust the other terms proportionally. This is a heuristic.
    
    # For now, return None - this approach is too complex.
    return None

def _try_local_search(U, V, W, n, m, p, target_rank, time_budget=30, seed=42):
    """Try to find a decomposition of target_rank starting from a given solution.
    
    Uses a local search approach: perturb the solution and try to reduce rank.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    current_U = np.array(U, dtype=np.float64)
    current_V = np.array(V, dtype=np.float64)
    current_W = np.array(W, dtype=np.float64)
    current_rank = current_U.shape[1]
    
    if current_rank < target_rank:
        return None
    
    # Strategy: try to merge terms by finding pairs that can be combined
    # This is a heuristic search.
    
    # For each pair of terms, try to see if they can be merged into one term
    # while maintaining the decomposition.
    
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    best_rank = current_rank
    
    # Try to reduce rank by 1
    for attempt in range(min(50, int(time_budget * 10))):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Pick a random term to try to eliminate
        term_idx = random.randint(0, best_rank - 1)
        
        # Try to eliminate this term by adjusting others
        # Simple approach: set this term to zero and check if we can
        # compensate by scaling other terms.
        
        # This is a simplified approach - in practice, we'd need a more
        # sophisticated method.
        
        # Try: can we express the contribution of term_idx as a combination
        # of other terms?
        keep = [r for r in range(best_rank) if r != term_idx]
        
        if len(keep) < target_rank:
            continue
        
        # Build the linear system
        num_constraints = n * m * p
        num_unknowns = len(keep)
        
        A_mat = np.zeros((num_constraints, num_unknowns))
        b_vec = np.zeros(num_constraints)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for u_idx, r in enumerate(keep):
                        A_mat[idx, u_idx] = best_U[i*m+j, r] * best_V[j*p+k, r] * best_W[k*n+i, r]
                    b_vec[idx] = best_U[i*m+j, term_idx] * best_V[j*p+k, term_idx] * best_W[k*n+i, term_idx]
                    idx += 1
        
        try:
            a_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        
        residual_norm = np.linalg.norm(A_mat @ a_sol - b_vec)
        if residual_norm > 1e-8:
            continue
        
        # If we found a solution, we can potentially merge the terms.
        # But the solution a_sol gives linear coefficients, not directly
        # usable for updating U, V, W.
        
        # For now, just record that we found a dependency.
        # A more sophisticated approach would be needed to actually
        # update the matrices.
        
        pass
    
    return None

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_direct_optimization(target_rank, n, m, p, max_iter=200, seed=42, time_limit=25):
    """Try to find a decomposition of given rank using direct optimization.
    
    Uses a smarter initialization and more robust optimization.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize with structured values
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Initialize with small random values
    U[:] = np.random.randn(n*m, target_rank) * 0.1
    V[:] = np.random.randn(m*p, target_rank) * 0.1
    W[:] = np.random.randn(p*n, target_rank) * 0.1
    
    # Normalize each term to have unit norm
    for r in range(target_rank):
        u_norm = np.linalg.norm(U[:, r])
        v_norm = np.linalg.norm(V[:, r])
        w_norm = np.linalg.norm(W[:, r])
        if u_norm > 1e-10 and v_norm > 1e-10 and w_norm > 1e-10:
            U[:, r] /= u_norm
            V[:, r] /= v_norm
            W[:, r] /= w_norm
    
    def loss_and_grad(params):
        U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        loss_val = 0.0
        U_grad = np.zeros_like(U_p)
        V_grad = np.zeros_like(V_p)
        W_grad = np.zeros_like(W_p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    
                    prod = U_p[u_idx, :] * V_p[v_idx, :]
                    s = np.dot(prod, W_p[w_idx, :])
                    resid = s - 1.0
                    
                    loss_val += resid * resid
                    
                    U_grad[u_idx, :] += 2.0 * resid * V_p[v_idx, :] * W_p[w_idx, :]
                    V_grad[v_idx, :] += 2.0 * resid * U_p[u_idx, :] * W_p[w_idx, :]
                    W_grad[w_idx, :] += 2.0 * resid * U_p[u_idx, :] * W_p[w_idx, :]
        
        grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
        return loss_val, grad
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    from scipy.optimize import minimize
    try:
        result = minimize(loss_and_grad, params, jac=True, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-5, 'ftol': 1e-10, 'maxls': 50})
    except Exception:
        return None
    
    if time.time() - start_time > time_limit:
        return None
    
    if result.fun < 1e-6:
        U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Try to snap to half-integers
        for snap_factor in [2.0, 4.0, 8.0]:
            U_snap = np.round(U_opt * snap_factor) / snap_factor
            V_snap = np.round(V_opt * snap_factor) / snap_factor
            W_snap = np.round(W_opt * snap_factor) / snap_factor
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If snapping failed, try without snapping
        U_list = U_opt.tolist()
        V_list = V_opt.tolist()
        W_list = W_opt.tolist()
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based (rank 53)
    try:
        result = _build_4x4x5_strassen()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
            best_rank = rank
    except Exception:
        pass
    
    # Try to find even lower rank through optimization if time permits
    if best_result is not None and time.time() < deadline - 40:
        # Try several target ranks
        for target_rank in range(best_rank - 1, max(best_rank - 8, 30), -1):
            if time.time() > deadline - 20:
                break
            found = False
            for seed in [target_rank, target_rank + 7, target_rank + 13]:
                if time.time() > deadline - 15:
                    break
                try:
                    result = _try_direct_optimization(target_rank, n, m, p, max_iter=150, seed=seed, time_limit=15)
                    if result is not None:
                        best_result = result
                        best_rank = target_rank
                        found = True
                        break
                except Exception:
                    pass
            if found:
                break
    
    if best_result is not None:
        # Ensure all values are proper Python floats/ints
        U = [[float(x) for x in row] for row in best_result["U"]]
        V = [[float(x) for x in row] for row in best_result["V"]]
        W = [[float(x) for x in row] for row in best_result["W"]]
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _build_schoolbook()

# EVOLVE-BLOCK-END
