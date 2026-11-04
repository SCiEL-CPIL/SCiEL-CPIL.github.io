"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy: Use Strassen decomposition on 2x3x2 sub-blocks, then attempt rank reduction
through systematic linear algebra and search.
"""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

# Strassen decomposition for 2x3 x 3x2 tensor (rank 11)
_STRASSEN_232_U = [
    {(0,0): 1, (1,2): 1},
    {(1,0): 1, (1,1): 1},
    {(0,0): 1},
    {(1,2): 1},
    {(0,0): 1, (0,1): 1},
    {(1,0): 1, (0,0): -1},
    {(0,1): 1, (1,2): -1},
    {(0,1): 1},
    {(1,0): 1},
    {(0,0): 1, (0,1): 1},
    {(1,0): 1, (1,1): 1},
]
_STRASSEN_232_V = [
    {(0,0): 1, (1,1): 1},
    {(0,0): 1},
    {(1,0): 1, (0,0): -1},
    {(1,1): 1, (0,1): -1},
    {(1,1): 1},
    {(0,0): 1, (0,1): 1},
    {(1,0): 1, (1,1): 1},
    {(0,1): 1},
    {(1,0): 1},
    {(1,1): 1, (1,0): -1},
    {(0,1): 1, (0,0): -1},
]
_STRASSEN_232_W = [
    {(0,0): 1, (1,1): 1},
    {(0,1): 1, (1,1): -1},
    {(0,0): 1, (0,1): 1},
    {(0,1): 1, (1,1): 1},
    {(0,0): 1, (1,1): 1},
    {(0,0): 1, (0,1): 1},
    {(0,1): 1, (1,1): 1},
    {(0,1): 1},
    {(1,0): 1},
    {(1,0): 1, (1,1): -1},
    {(1,0): 1, (1,1): 1},
]

def _block_decomposition_232(n, m, p):
    """Build rank-44 decomposition by 4 blocks of 11 terms each using 2x3x2 Strassen."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b
        col = b * 11
        for r in range(11):
            for (i, j), val in _STRASSEN_232_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_232_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_232_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val

    return {"U": U, "V": V, "W": W}

def _compress_scalar(U, V, W, n, m, p):
    """Remove scalar-multiple terms."""
    rank = len(U[0])
    if rank <= 1:
        return U, V, W, rank
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    removed = set()
    new_rank = rank
    for _ in range(100):
        if new_rank <= 1:
            break
        found = False
        for r1 in range(new_rank):
            if r1 in removed:
                continue
            for r2 in range(r1 + 1, new_rank):
                if r2 in removed:
                    continue
                u1 = U_np[r1]
                u2 = U_np[r2]
                nz = np.nonzero(u2)[0]
                if len(nz) == 0:
                    continue
                a = u1[nz[0]] / u2[nz[0]]
                if not np.allclose(u1, a * u2, atol=1e-9):
                    continue
                v1 = V_np[r1]
                v2 = V_np[r2]
                if not np.allclose(v1, a * v2, atol=1e-9):
                    continue
                w1 = W_np[r1]
                w2 = W_np[r2]
                if not np.allclose(w1, a * w2, atol=1e-9):
                    continue
                removed.add(r2)
                found = True
                break
            if found:
                break
        if not found:
            break
        new_rank -= 1
    
    if not removed:
        return U, V, W, rank
    
    keep = [i for i in range(rank) if i not in removed]
    return [U[i] for i in keep], [V[i] for i in keep], [W[i] for i in keep], len(keep)

def _compress_tensor(U, V, W, n, m, p, max_iters=50, tol=1e-8):
    """Compress by removing tensor-linearly dependent terms."""
    rank = len(U[0])
    if rank <= 1:
        return U, V, W, rank
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    nm, mp, pn = n*m, m*p, p*n
    tensors = np.zeros((rank, nm*mp*pn))
    for r in range(rank):
        t = np.outer(np.outer(U_np[r], V_np[r]), W_np[r])
        tensors[r] = t.flatten()
    
    keep = list(range(rank))
    for _ in range(max_iters):
        if len(keep) <= 1:
            break
        found = False
        idx = len(keep) - 1
        r = keep[idx]
        if idx > 0:
            A = tensors[keep[:idx]]
            b = tensors[r]
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=tol)
                if residuals and len(residuals) > 0:
                    res = np.linalg.norm(b - A @ x)
                    if res < tol * max(1.0, np.linalg.norm(b)):
                        keep.pop(idx)
                        found = True
            except Exception:
                pass
        if not found:
            break
    
    if len(keep) == rank:
        return U, V, W, rank
    
    return [U[i] for i in keep], [V[i] for i in keep], [W[i] for i in keep], len(keep)

def _try_reduce_rank(U, V, W, n, m, p, target_rank, timeout=5):
    """Try to reduce to target_rank by zeroing out some terms."""
    start_time = time.time()
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    # Try removing terms one at a time and checking validity
    U_work = [row[:] for row in U]
    V_work = [row[:] for row in V]
    W_work = [row[:] for row in W]
    
    removed = set()
    while len(removed) < current_rank - target_rank:
        if time.time() - start_time > timeout:
            break
        found = False
        for r in range(current_rank):
            if r in removed:
                continue
            # Try removing term r
            U_test = [row[:] for row in U_work]
            V_test = [row[:] for row in V_work]
            W_test = [row[:] for row in W_work]
            
            # Zero out column r
            for i in range(len(U_test)):
                U_test[i][r] = 0
            for i in range(len(V_test)):
                V_test[i][r] = 0
            for i in range(len(W_test)):
                W_test[i][r] = 0
            
            if _validate(U_test, V_test, W_test, n, m, p):
                removed.add(r)
                found = True
                break
        
        if not found:
            break
    
    if not removed:
        return U, V, W, current_rank
    
    keep = [i for i in range(current_rank) if i not in removed]
    return [U[i] for i in keep], [V[i] for i in keep], [W[i] for i in keep], len(keep)

def _find_232_decomp_rank(target_rank, deadline):
    """Search for a 2x3x2 decomposition with given target rank using random init + Newton."""
    start = time.time()
    n, m, p = 2, 3, 2
    # Indices: i in [0,1], j in [0,1,2], k in [0,1]
    # U: (i,j) -> 6 entries per term
    # V: (j,k) -> 6 entries per term  
    # W: (k,i) -> 4 entries per term
    
    best_result = None
    
    for trial in range(50):
        if time.time() > deadline:
            break
        np.random.seed(trial * 12345 + 678)
        R = target_rank
        # Random initialization
        U = np.random.randn(R, 6) * 0.5
        V = np.random.randn(R, 6) * 0.5
        W = np.random.randn(R, 4) * 0.5
        
        # Newton iteration to solve sum_r U[r,i,j]*V[r,j,k]*W[r,k,i] = 1
        # Define F(U,V,W) = tensor - target
        # Use Gauss-Newton / Levenberg-Marquardt style
        
        for it in range(200):
            if time.time() > deadline:
                break
            # Compute residual
            resid = np.zeros((2, 3, 2))
            for r in range(R):
                for i in range(2):
                    for j in range(3):
                        for k in range(2):
                            resid[i,j,k] += U[r,i*3+j] * V[r,j*2+k] * W[r,k*2+i]
            resid -= 1.0  # target is 1 everywhere
            
            if np.max(np.abs(resid)) < 1e-12:
                # Found solution
                best_result = (U, V, W)
                break
            
            # Compute Jacobian (simplified: use finite differences or analytical)
            # For speed, use a simplified update
            # Update rule: minimize ||F||^2
            # dF/dU[r,i,j] = V[r,j,k]*W[r,k,i] for each (i,j,k)
            # This is complex, so use a simpler approach:
            # Perturb and use least squares on the linearized system
            
            # Linearize: F(U+delta_U) ≈ F(U) + sum_r delta_U[r,:] * (V[r,:] ⊗ W[r,:])
            # Build the linear system
            # Number of equations: 12
            # Number of unknowns: R*6 + R*6 + R*4 = 16R
            
            # Actually, let's use a different approach: 
            # Fix V and W, solve for U, then fix U and W, solve for V, etc.
            
            # Fix V, W: solve for U
            # sum_r U[r,i,j] * V[r,j,k] * W[r,k,i] = 1 for all (i,j,k)
            # For fixed (i,k): sum_j U[r,i,j] * V[r,j,k] * W[r,k,i] = 1
            # This is still coupled. Let's use coordinate descent.
            
            # Update U: for each (i,j), U[r,i,j] appears in equations for all k
            # sum_r U[r,i,j] * V[r,j,k] * W[r,k,i] = 1 for all k
            # For fixed (i,j): V[:,j,:] * W[:,k,i] gives coefficients
            # Actually: for fixed (i,j), the contribution to equation (i,j,k) is U[r,i,j]*V[r,j,k]*W[r,k,i]
            # Sum over r: sum_r U[r,i,j]*V[r,j,k]*W[r,k,i] = 1
            # This is a linear system in U[:,i,j] if we fix others... but it's coupled across r.
            
            # Let's just do a simple gradient descent with line search
            # Gradient of 0.5*||F||^2 w.r.t. U[r,i,j]:
            # sum_k resid[i,j,k] * V[r,j,k] * W[r,k,i]
            
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            for r in range(R):
                for i in range(2):
                    for j in range(3):
                        for k in range(2):
                            gU[r, i*3+j] += resid[i,j,k] * V[r,j*2+k] * W[r,k*2+i]
                            gV[r, j*2+k] += resid[i,j,k] * U[r,i*3+j] * W[r,k*2+i]
                            gW[r, k*2+i] += resid[i,j,k] * U[r,i*3+j] * V[r,j*2+k]
            
            # Line search
            lr = 0.01
            for ls in range(20):
                U_new = U - lr * gU
                V_new = V - lr * gV
                W_new = W - lr * gW
                
                resid_new = np.zeros((2, 3, 2))
                for r in range(R):
                    for i in range(2):
                        for j in range(3):
                            for k in range(2):
                                resid_new[i,j,k] += U_new[r,i*3+j] * V_new[r,j*2+k] * W_new[r,k*2+i]
                resid_new -= 1.0
                
                if np.max(np.abs(resid_new)) < np.max(np.abs(resid)):
                    U, V, W = U_new, V_new, W_new
                    resid = resid_new
                    break
                else:
                    lr *= 0.5
            
            if np.max(np.abs(resid)) < 1e-12:
                best_result = (U, V, W)
                break
    
        if best_result is not None:
            break
    
    if best_result is None:
        return None
    
    U, V, W = best_result
    # Convert to list of dicts
    U_list = []
    V_list = []
    W_list = []
    for r in range(R):
        Ud = {}
        Vd = {}
        Wd = {}
        for i in range(2):
            for j in range(3):
                if abs(U[r, i*3+j]) > 1e-10:
                    Ud[(i,j)] = round(U[r, i*3+j], 10)
        for j in range(3):
            for k in range(2):
                if abs(V[r, j*2+k]) > 1e-10:
                    Vd[(j,k)] = round(V[r, j*2+k], 10)
        for k in range(2):
            for i in range(2):
                if abs(W[r, k*2+i]) > 1e-10:
                    Wd[(k,i)] = round(W[r, k*2+i], 10)
        U_list.append(Ud)
        V_list.append(Vd)
        W_list.append(Wd)
    
    return (U_list, V_list, W_list)

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115

    # Start with rank-44 Strassen block decomposition
    best = None
    try:
        cand = _block_decomposition_232(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            # First compress scalar-multiple terms
            U_c, V_c, W_c, rank_c = _compress_scalar(cand["U"], cand["V"], cand["W"], n, m, p)
            if _validate(U_c, V_c, W_c, n, m, p):
                # Then compress tensor-linearly dependent terms
                U_t, V_t, W_t, rank_t = _compress_tensor(U_c, V_c, W_c, n, m, p)
                if _validate(U_t, V_t, W_t, n, m, p):
                    best = {"U": U_t, "V": V_t, "W": W_t}
                else:
                    best = {"U": U_c, "V": V_c, "W": W_c}
            else:
                best = cand
    except Exception:
        pass

    # Try to find a better 2x3x2 decomposition with rank 10
    if time.time() < deadline - 30:
        try:
            decomp_10 = _find_232_decomp_rank(10, deadline - 10)
            if decomp_10 is not None:
                U_10, V_10, W_10 = decomp_10
                # Build full decomposition with rank 4*10 = 40
                nblocks = 4
                rank = nblocks * 10
                U_full = [[0] * rank for _ in range(n * m)]
                V_full = [[0] * rank for _ in range(m * p)]
                W_full = [[0] * rank for _ in range(p * n)]
                
                for b in range(nblocks):
                    k0 = 2 * b
                    col = b * 10
                    for r in range(10):
                        for (i, j), val in U_10[r].items():
                            U_full[i * m + j][col + r] = val
                        for (j, kp), val in V_10[r].items():
                            V_full[j * p + (k0 + kp)][col + r] = val
                        for (kp, i), val in W_10[r].items():
                            W_full[(k0 + kp) * n + i][col + r] = val
                
                if _validate(U_full, V_full, W_full, n, m, p):
                    best = {"U": U_full, "V": V_full, "W": W_full}
        except Exception:
            pass

    if best is not None:
        return best

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
