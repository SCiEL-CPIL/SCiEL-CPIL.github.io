import numpy as np
import time
import sys
from itertools import product
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-7):
    """Check that the decomposition is exact and entries are half-integers."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Check all (i,k) positions: C[i,k] = sum_j A[i,j]B[j,k]
    # We test with random A, B
    for _ in range(20):
        A = np.random.randint(0, 3, (n, m)).astype(float)
        B = np.random.randint(0, 3, (m, p)).astype(float)
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n).T  # W[k*n+i] -> w[i,k]
            C_calc += (u @ A) * 0 + (A @ u) * 0  # placeholder
        # Actually compute properly: the bilinear form
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_l V[j*p+l,c]*B[j,l])
        # But U and V are linear forms on A and B respectively.
        # Let's recompute: for each c, the term is W[k*n+i,c] * L_A(c,A) * L_B(c,B)
        # where L_A(c,A) = sum_{i,j} U[i*m+j,c] * A[i,j]
        # and L_B(c,B) = sum_{j,k} V[j*p+k,c] * B[j,k]
        C_calc = np.zeros((n, p))
        for c in range(r):
            LA = np.sum(U[:, c].reshape(n, m) * A)
            LB = np.sum(V[:, c].reshape(m, p) * B)
            Wcol = W[:, c].reshape(p, n).T  # w[i,k] = W[k*n+i, c]
            C_calc += LA * LB * Wcol
        if not np.allclose(C_calc, C_true, atol=tol):
            return False
    # Check half-integer constraint
    for mat in [U, V, W]:
        for val in mat:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
    return True

def _build_strassen_patterns():
    """
    Build a dictionary of structured rank-1 tensor patterns.
    Pattern 0: schoolbook basis vectors (24 of them)
    Patterns 1+: Strassen-style cross patterns for 2x2 blocks, extended to 2x3x4
    """
    n, m, p = DIMENSIONS
    patterns = []
    
    # Schoolbook basis
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                patterns.append((u, v, w))
    
    # Strassen-style: for 2x2 sub-blocks, use the 7-multiplication scheme
    # For 2x3x4, we can consider overlapping 2x2 blocks
    # Strassen for 2x2: C = AB where A,B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For each 2x2 sub-block of A (rows i,i+1; cols j,j+1) and 2x2 sub-block of B (rows j,j+1; cols k,k+1)
    # We need to handle the fact that A is 2x3 and B is 3x4
    # A sub-block: rows {i,i+1}, cols {j,j+1} for i in 0..1, j in 0..2
    # B sub-block: rows {j,j+1}, cols {k,k+1} for j in 0..2, k in 0..3
    
    for i in range(n - 1):  # 0
        for j in range(m - 1):  # 0, 1
            for k in range(p - 1):  # 0, 1, 2
                # 7 Strassen multiplications for the 2x2 block A[i:i+2, j:j+2] * B[j:j+2, k:k+2]
                # But we need to map these to the full 2x3x4 tensor
                # Each Strassen multiplication corresponds to a linear form on A, a linear form on B,
                # and a matrix (linear form on C)
                
                # M1 = (A[i,j]+A[i+1,j+1]) * (B[j,k]+B[j+1,k+1])
                u1 = np.zeros(n * m)
                v1 = np.zeros(m * p)
                w1 = np.zeros(p * n)
                u1[i * m + j] = 1
                u1[(i+1) * m + (j+1)] = 1
                v1[j * p + k] = 1
                v1[(j+1) * p + (k+1)] = 1
                # Result contributes to C[i,k], C[i,k+1], C[i+1,k], C[i+1,k+1]
                # C[i,k] += A[i,j]*B[j,k]
                # C[i,k+1] += A[i,j]*B[j,k+1]
                # C[i+1,k] += A[i+1,j]*B[j,k]
                # C[i+1,k+1] += A[i+1,j]*B[j,k+1]
                # But Strassen M1 = (A[i,j]+A[i+1,j+1])(B[j,k]+B[j+1,k+1])
                # = A[i,j]B[j,k] + A[i,j]B[j+1,k+1] + A[i+1,j+1]B[j,k] + A[i+1,j+1]B[j+1,k+1]
                # This doesn't directly give the 4 products we want.
                # The Strassen scheme is designed so that the 7 M's combine to give all 4 products.
                # For our purposes, we just need the individual linear forms.
                
                # Let's just add the 7 Strassen linear forms as patterns
                # M1
                u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
                u[i * m + j] = 1; u[(i+1) * m + (j+1)] = 1
                v[j * p + k] = 1; v[(j+1) * p + (k+1)] = 1
                # The output matrix for M1 in Strassen is:
                # C[i,k] += M1, C[i,k+1] -= M3, C[i+1,k] -= M4, C[i+1,k+1] += M1
                # This is complex. Let's just add the linear forms without worrying about the exact output.
                # The output W will be determined by the linear solve.
                # Actually, we need to specify what C entries this pattern contributes to.
                # For simplicity, let's just add the linear forms and let the solver figure out W.
                # But W is also part of the pattern.
                
                # Let me take a different approach: just add the schoolbook patterns and a few
                # "mixed" patterns where U and V have two non-zero entries each.
                
    # Add mixed patterns: U has 2 non-zero entries, V has 2 non-zero entries
    for i1 in range(n):
        for j1 in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    if (i1, j1) >= (i2, j2):
                        continue
                    for j3 in range(m):
                        for k1 in range(p):
                            for k2 in range(p):
                                if (j3, k1) >= (j3, k2):
                                    continue
                                # Skip if too many
                                if len(patterns) > 200:
                                    break
                                u = np.zeros(n * m)
                                v = np.zeros(m * p)
                                w = np.zeros(p * n)
                                u[i1 * m + j1] = 1
                                u[i2 * m + j2] = 1
                                v[j3 * p + k1] = 1
                                v[j3 * p + k2] = 1
                                # W: this pattern contributes to C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
                                # But we don't know the exact coefficients yet
                                # Let's set W to 1 for all 4 positions
                                w[k1 * n + i1] = 1
                                w[k2 * n + i1] = 1
                                w[k1 * n + i2] = 1
                                w[k2 * n + i2] = 1
                                patterns.append((u, v, w))
                                if len(patterns) > 200:
                                    break
                            if len(patterns) > 200:
                                break
                        if len(patterns) > 200:
                            break
                    if len(patterns) > 200:
                        break
                if len(patterns) > 200:
                    break
            if len(patterns) > 200:
                break
        if len(patterns) > 200:
            break
    return patterns

def _solve_linear_combination(patterns, target_rank):
    """
    Find a linear combination of patterns that gives a valid decomposition
    with at most target_rank non-zero coefficients.
    """
    n, m, p = DIMENSIONS
    num_patterns = len(patterns)
    
    # Build the tensor equation matrix
    # For each (i,k) position and each pattern c, the coefficient is:
    # sum_j U_c[i*m+j] * A[i,j] * sum_l V_c[j*p+l] * B[j,l] * W_c[k*n+i]
    # This is bilinear in A and B, so we need to check it for all A, B.
    # Equivalently, for each (i,j,k) triple, the coefficient must be 1.
    # For pattern c, the contribution to (i,j,k) is:
    # U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # So we need: sum_c alpha_c * U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all (i,j,k)
    
    # Build the linear system: for each (i,j,k), one equation
    # num_equations = n * m * p = 24
    # unknowns: alpha_1, ..., alpha_num_patterns
    # We want a sparse solution (at most target_rank non-zero alphas)
    
    A_mat = np.zeros((n * m * p, num_patterns))
    for eq, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        for c in range(num_patterns):
            u, v, w = patterns[c]
            A_mat[eq, c] = u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    b = np.ones(n * m * p)
    
    # Try to find a sparse solution
    # Use l1 minimization: minimize ||alpha||_1 s.t. A_mat @ alpha = b
    # Or use a greedy approach
    
    # First, try least squares
    alpha_ls, residuals, rank, sv = np.linalg.lstsq(A_mat, b, rcond=None)
    
    # Check if the LS solution is close to half-integers
    alpha_ls_rounded = np.round(alpha_ls * 2) / 2
    if np.allclose(A_mat @ alpha_ls_rounded, b, atol=1e-7):
        # Find non-zero alphas
        nonzero = np.where(np.abs(alpha_ls_rounded) > 1e-7)[0]
        if len(nonzero) <= target_rank:
            # Build the decomposition
            U = np.zeros((n * m, len(nonzero)))
            V = np.zeros((m * p, len(nonzero)))
            W = np.zeros((p * n, len(nonzero)))
            for idx, c in enumerate(nonzero):
                u, v, w = patterns[c]
                U[:, idx] = alpha_ls_rounded[c] * u
                V[:, idx] = alpha_ls_rounded[c] * v
                W[:, idx] = alpha_ls_rounded[c] * w
            return U, V, W
    
    # Try greedy: pick patterns that best satisfy the equations
    residual = b.copy()
    selected = []
    for _ in range(target_rank):
        if _time_left() < 5:
            break
        best_c = -1
        best_res = np.inf
        for c in range(num_patterns):
            if c in selected:
                continue
            # Try adding pattern c with optimal coefficient
            # Minimize ||residual - alpha * A_mat[:, c]||^2
            col = A_mat[:, c]
            if np.linalg.norm(col) < 1e-10:
                continue
            alpha = np.dot(residual, col) / np.dot(col, col)
            new_res = residual - alpha * col
            res = np.linalg.norm(new_res)
            if res < best_res:
                best_res = res
                best_c = c
                best_alpha = alpha
        if best_c >= 0:
            selected.append(best_c)
            residual = residual - best_alpha * A_mat[:, best_c]
            if best_res < 1e-7:
                break
    
    if len(selected) > 0 and best_res < 1e-7:
        # Check if alphas are half-integers
        U = np.zeros((n * m, len(selected)))
        V = np.zeros((m * p, len(selected)))
        W = np.zeros((p * n, len(selected)))
        for idx, c in enumerate(selected):
            u, v, w = patterns[c]
            # Recompute alpha
            # This is approximate; let's just use the greedy alphas
            # For now, skip this and return None
            pass
        return None
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Get schoolbook solution
    U_sb, V_sb, W_sb = _build_schoolbook()
    
    # Build pattern dictionary
    patterns = _build_strassen_patterns()
    
    # Try to find a lower-rank solution via linear combination
    for target_rank in [rank_schoolbook - 1, rank_schoolbook - 2, rank_schoolbook - 3]:
        if _time_left() < 10:
            break
        sol = _solve_linear_combination(patterns, target_rank)
        if sol is not None:
            U, V, W = sol
            if _verify(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: try gradient-based search from the current program
    def try_gradient_search(target_rank):
        num_vars = (n * m + m * p + p * n) * target_rank
        
        def unpack_vars(x):
            U_flat = x[:n * m * target_rank]
            V_flat = x[n * m * target_rank: n * m * target_rank + m * p * target_rank]
            W_flat = x[n * m * target_rank + m * p * target_rank:]
            U = U_flat.reshape((n * m, target_rank))
            V = V_flat.reshape((m * p, target_rank))
            W = W_flat.reshape((p * n, target_rank))
            return U, V, W
        
        def objective(x):
            U, V, W = unpack_vars(x)
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        res += (val - 1.0) ** 2
            return res
        
        np.random.seed(42 + target_rank)
        best_solution = None
        
        for restart in range(50):
            if _time_left() < 5:
                break
            
            if restart % 3 == 0:
                x0 = np.zeros(num_vars)
                U0 = np.zeros((n * m, target_rank))
                V0 = np.zeros((m * p, target_rank))
                W0 = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    U0[:, c] = U_sb[:, c]
                    V0[:, c] = V_sb[:, c]
                    W0[:, c] = W_sb[:, c]
                x0 = np.concatenate([
                    (U0 + np.random.randn(n * m, target_rank) * 0.1).flatten(),
                    (V0 + np.random.randn(m * p, target_rank) * 0.1).flatten(),
                    (W0 + np.random.randn(p * n, target_rank) * 0.1).flatten()
                ])
            elif restart % 3 == 1:
                x0 = np.random.randn(num_vars) * 0.5
            else:
                x0 = np.random.randn(num_vars)
                x0[np.random.choice(num_vars, int(num_vars * 0.7), replace=False)] = 0
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e-10:
                    U, V, W = unpack_vars(res.x)
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if _verify(U_snap, V_snap, W_snap):
                        return (U_snap, V_snap, W_snap)
            except Exception:
                pass
        
        return None
    
    for target_rank in [rank_schoolbook - 1, rank_schoolbook - 2]:
        if _time_left() < 10:
            break
        sol = try_gradient_search(target_rank)
        if sol is not None:
            U, V, W = sol
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
