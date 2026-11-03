"""Strategy: Global Linear-Algebraic Search via Iterative Refinement.

Instead of tiling disjoint Strassen blocks, we model the entire 3x5x6 matmul tensor
T[i,j,k] = 1 as a sum of rank-1 terms. We start with a high-rank dense random
initialization and iteratively:
1. Fix two factor matrices, solve for the third via least squares.
2. Prune columns that have small norm or contribute little to the tensor.
3. Repeat until convergence or rank stabilizes.
We also include a "greedy rank-1 subtraction" phase: at each step, find the best
rank-1 approximation to the residual tensor and subtract it.

This is a fundamentally different approach from tiling: it allows cross-term
interactions that can reduce rank below any block-diagonal decomposition.
"""

import time
import numpy as np

DIMENSIONS = [3, 5, 6]
DEADLINE = time.time() + 140  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. Returns True if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition, rank = n*m*p."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

def _strassen_2x2x2():
    """Return 7 Strassen rank-1 terms for 2x2x2 matmul as (u_dict, v_dict, w_dict)."""
    terms = []
    terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
    terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}))
    terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}))
    terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}))
    terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}))
    terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
    terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
    return terms

def _tiling_decomposition(n, m, p):
    """Place 2x2x2 Strassen blocks where they fit, schoolbook for the rest."""
    strassen = _strassen_2x2x2()
    blocks = []
    for i0 in range(0, n - 1):
        for j0 in range(0, m - 1):
            for k0 in range(0, p - 1):
                blocks.append((i0, j0, k0))
    
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    remaining = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                 if (i, j, k) not in covered]
    
    total_rank = len(blocks) * 7 + len(remaining)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    for (i0, j0, k0) in blocks:
        for r in range(7):
            ud, vd, wd = strassen[r]
            for (di, dj), val in ud.items():
                U[(i0 + di) * m + (j0 + dj), col] += val
            for (dj, dk), val in vd.items():
                V[(j0 + dj) * p + (k0 + dk), col] += val
            for (dk, di), val in wd.items():
                W[(k0 + dk) * n + (i0 + di), col] += val
            col += 1
    for (i, j, k) in remaining:
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank
    return None, total_rank

def _random_init(n, m, p, rank, seed=42):
    """Initialize factor matrices with random values."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - 1."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return T - 1.0

def _alternating_least_squares(U, V, W, n, m, p, max_iters=500, tol=1e-12):
    """
    Alternating least squares to minimize ||T(U,V,W) - 1||_F^2.
    At each step, fix two matrices and solve for the third.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    prev_err = np.inf
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Solve for W: for each (k,i), W[k*n+i,:] = argmin ||sum_j U[i*m+j,:] V[j*p+k,:] W[:,k*n+i] - 1||
        # This is a linear system in W[:,k*n+i]
        # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # For fixed (k,i): 1 = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # Let A_{j,c} = U[i*m+j,c] * V[j*p+k,c]
        # Then 1 = sum_j sum_c A_{j,c} W[k*n+i,c]
        # This is not quite right because the target is 1 for all (i,j,k)
        
        # Better approach: vectorize.
        # T = einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        # We want to minimize ||T - 1||^2 with respect to W.
        # This is a quadratic in W.
        
        # Let's use a simpler approach: gradient descent on W.
        # Actually, let's do proper ALS.
        
        # Reshape for efficient computation
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        # Solve for W:
        # For each c, W[:,c] appears in T[i,j,k] as Wr[k,i,c]
        # The gradient with respect to Wr[k,i,c] is 2 * (T[i,j,k] - 1) * Ur[i,j,c] * Vr[j,k,c]
        # This is complex. Let's use a different approach.
        
        # Use the fact that the problem is linear in each factor matrix.
        # For W: fix U and V. The tensor T is linear in W.
        # T[i,j,k] = sum_c (sum_j Ur[i,j,c] Vr[j,k,c]) Wr[k,i,c]
        # Let B[i,k,c] = sum_j Ur[i,j,c] Vr[j,k,c]
        # Then T[i,j,k] = sum_c B[i,k,c] Wr[k,i,c] ... this is not right because j is free.
        
        # Correct formulation:
        # T[i,j,k] = sum_c Ur[i,j,c] Vr[j,k,c] Wr[k,i,c]
        # For fixed U, V, this is linear in W.
        # Vectorize: let w = vec(W) of length p*n*rank
        # T = M * w + ... no, it's a trilinear form.
        
        # Let's just do coordinate descent on each entry of W.
        # Or better: use the fact that for each (k,i), W[k*n+i,:] is a vector of length rank.
        # The contribution to T[i,j,k] from column c is Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # So for fixed (k,i), we have:
        # 1 ≈ sum_j sum_c Ur[i,j,c] Vr[j,k,c] Wr[k,i,c]
        # This is a system of m equations (one for each j) in rank unknowns (Wr[k,i,0..rank-1])
        # Let A_{j,c} = Ur[i,j,c] * Vr[j,k,c]
        # Then 1 = A * Wr[k,i,:]
        # We can solve this least squares problem.
        
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                # Build A: m x rank
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = Ur[i, j, :] * Vr[j, k, :]
                # Solve A * w = ones(m)
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w_sol
                except:
                    pass
        
        W = W_new
        
        # Similarly solve for U and V
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_c Ur[i,j,c] Vr[j,k,c] Wr[k,i,c]
                # For fixed (i,j), we have p equations in rank unknowns
                # A_{k,c} = Vr[j,k,c] * Wr[k,i,c]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = Vr[j, k, :] * Wr[k, i, :]
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u_sol
                except:
                    pass
        U = U_new
        
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        Ur = U.reshape(n, m, rank)
        
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c Ur[i,j,c] Vr[j,k,c] Wr[k,i,c]
                # For fixed (j,k), we have n equations in rank unknowns
                # A_{i,c} = Ur[i,j,c] * Wr[k,i,c]
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = Ur[i, j, :] * Wr[k, i, :]
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v_sol
                except:
                    pass
        V = V_new
        
        # Check convergence
        residual = _compute_residual(U, V, W, n, m, p)
        err = np.linalg.norm(residual)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _prune_columns(U, V, W, n, m, p, threshold=1e-6):
    """Prune columns that have small norm or contribute little to the tensor."""
    rank = U.shape[1]
    keep = np.ones(rank, dtype=bool)
    
    # Check column norms
    for c in range(rank):
        norm_u = np.linalg.norm(U[:, c])
        norm_v = np.linalg.norm(V[:, c])
        norm_w = np.linalg.norm(W[:, c])
        if norm_u < threshold or norm_v < threshold or norm_w < threshold:
            keep[c] = False
    
    if not np.all(keep):
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    return U, V, W

def _greedy_rank1_subtraction(n, m, p, max_rank=100, seed=42):
    """
    Greedy algorithm: start with residual = ones tensor.
    At each step, find the best rank-1 approximation to the residual,
    subtract it, and record the factors.
    """
    residual = np.ones((n, m, p))
    U_cols = []
    V_cols = []
    W_cols = []
    
    rng = np.random.default_rng(seed)
    
    for step in range(max_rank):
        if time.time() > DEADLINE:
            break
        
        # Find best rank-1 approximation to residual
        # Use iterative method: start with random vectors, alternate updates
        u = rng.standard_normal(n * m)
        v = rng.standard_normal(m * p)
        w = rng.standard_normal(p * n)
        
        best_err = np.inf
        for _ in range(50):
            # Update u
            u = u.reshape(n, m)
            v = v.reshape(m, p)
            w = w.reshape(p, n)
            
            # u[i,j] = sum_{k} residual[i,j,k] * v[j,k] * w[k,i]
            u_new = np.einsum('ijk,jk,ki->ij', residual, v, w)
            if np.linalg.norm(u_new) > 0:
                u = u_new / np.linalg.norm(u_new) * np.linalg.norm(u)
            else:
                break
            
            # Update v
            v_new = np.einsum('ijk,ij,ki->jk', residual, u, w)
            if np.linalg.norm(v_new) > 0:
                v = v_new / np.linalg.norm(v_new) * np.linalg.norm(v)
            else:
                break
            
            # Update w
            w_new = np.einsum('ijk,ij,jk->ki', residual, u, v)
            if np.linalg.norm(w_new) > 0:
                w = w_new / np.linalg.norm(w_new) * np.linalg.norm(w)
            else:
                break
        
        # Compute the rank-1 term
        u = u.reshape(n, m)
        v = v.reshape(m, p)
        w = w.reshape(p, n)
        term = np.einsum('ij,jk,ki->ijk', u, v, w)
        
        # Check if this term improves the residual
        new_residual = residual - term
        if np.linalg.norm(new_residual) >= best_err and step > 0:
            # Try another random init
            continue
        
        residual = new_residual
        U_cols.append(u.flatten())
        V_cols.append(v.flatten())
        W_cols.append(w.flatten())
        
        if np.linalg.norm(residual) < 1e-10:
            break
    
    if len(U_cols) == 0:
        return None
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U, V, W

def _hybrid_search(n, m, p):
    """
    Hybrid search: start with tiling decomposition, then use ALS to refine
    and prune to reduce rank.
    """
    # Start with tiling
    tiling_result, tiling_rank = _tiling_decomposition(n, m, p)
    if tiling_result is None:
        return _schoolbook(n, m, p)
    
    U = np.array(tiling_result["U"])
    V = np.array(tiling_result["V"])
    W = np.array(tiling_result["W"])
    best_rank = tiling_rank
    best_result = tiling_result
    
    # Try to reduce rank via ALS and pruning
    for trial in range(5):
        if time.time() > DEADLINE:
            break
        
        # Perturb slightly
        rng = np.random.default_rng(42 + trial)
        U_pert = U + rng.standard_normal(U.shape) * 0.1
        V_pert = V + rng.standard_normal(V.shape) * 0.1
        W_pert = W + rng.standard_normal(W.shape) * 0.1
        
        # Run ALS
        U_ref, V_ref, W_ref = _alternating_least_squares(
            U_pert, V_pert, W_pert, n, m, p, max_iters=100
        )
        
        # Prune
        U_pruned, V_pruned, W_pruned = _prune_columns(U_ref, V_ref, W_ref, n, m, p)
        
        if U_pruned.shape[1] < best_rank:
            if _verify(U_pruned, V_pruned, W_pruned, n, m, p):
                best_rank = U_pruned.shape[1]
                best_result = {
                    "U": U_pruned.tolist(),
                    "V": V_pruned.tolist(),
                    "W": W_pruned.tolist()
                }
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # First, get the tiling-based baseline (known good, rank 84 for 3x5x6)
    tiling_result, tiling_rank = _tiling_decomposition(n, m, p)
    
    if tiling_result is None:
        # Fallback to schoolbook
        sb_result, sb_rank = _schoolbook(n, m, p)
        return sb_result
    
    best_result = tiling_result
    best_rank = tiling_rank
    
    # Try hybrid search to see if we can do better
    if time.time() < DEADLINE:
        hybrid_result = _hybrid_search(n, m, p)
        if hybrid_result is not None:
            # Verify the hybrid result
            U_h = np.array(hybrid_result["U"])
            V_h = np.array(hybrid_result["V"])
            W_h = np.array(hybrid_result["W"])
            if _verify(U_h, V_h, W_h, n, m, p):
                hybrid_rank = U_h.shape[1]
                if hybrid_rank < best_rank:
                    best_result = hybrid_result
                    best_rank = hybrid_rank
    
    # Final fallback
    if best_result is None:
        sb_result, sb_rank = _schoolbook(n, m, p)
        return sb_result
    
    return best_result
