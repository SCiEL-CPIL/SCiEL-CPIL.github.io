"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _tensor_error(U_np, V_np, W_np, T):
    M1 = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]
    R = np.einsum('abr,cr->abc', M1, W_np)
    err = np.sum((R - T) ** 2)
    return err

def _snap_to_half_int(U, V, W):
    U = [[round(x * 2) / 2 for x in row] for row in U]
    V = [[round(x * 2) / 2 for x in row] for row in V]
    W = [[round(x * 2) / 2 for x in row] for row in W]
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _try_polish_gradient(U, V, W, T, rank, deadline, initial_err, max_iters=20000, seed=None):
    """
    Gradient-based local search refinement with adaptive learning rate and momentum.
    """
    if seed is not None:
        np.random.seed(seed)
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_err = initial_err
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    lr = 0.05
    momentum = 0.9
    vU = np.zeros_like(U_np)
    vV = np.zeros_like(V_np)
    vW = np.zeros_like(W_np)
    
    # Precompute dimensions for speed
    nm, mp, pn = U_np.shape[0], V_np.shape[0], W_np.shape[0]
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gradients
        M1 = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]  # (nm, mp, rank)
        R = np.einsum('abr,cr->abc', M1, W_np)                # (nm, mp, pn)
        residual = R - T  # (nm, mp, pn)
        
        # Gradients
        temp_VW = V_np[:, np.newaxis, :] * W_np[np.newaxis, :, :]  # (mp, pn, rank)
        grad_U = 2.0 * np.einsum('abc,bcr->ar', residual, temp_VW)
        
        temp_UW = U_np[:, np.newaxis, :] * W_np[np.newaxis, :, :]  # (nm, pn, rank)
        grad_V = 2.0 * np.einsum('abc,acr->br', residual, temp_UW)
        
        temp_UV = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]  # (nm, mp, rank)
        grad_W = 2.0 * np.einsum('abc,abr->cr', residual, temp_UV)
        
        # Update with momentum
        vU = momentum * vU - lr * grad_U
        vV = momentum * vV - lr * grad_V
        vW = momentum * vW - lr * grad_W
        
        U_np = U_np + vU
        V_np = V_np + vV
        W_np = W_np + vW
        
        new_err = _tensor_error(U_np, V_np, W_np, T)
        
        if new_err < best_err:
            best_err = new_err
            best_U = U_np.copy()
            best_V = V_np.copy()
            best_W = W_np.copy()
            
            if best_err < 1e-10:
                # Try snapping
                U_snapped, V_snapped, W_snapped = _snap_to_half_int(
                    best_U.tolist(), best_V.tolist(), best_W.tolist()
                )
                U_s_np = np.array(U_snapped)
                V_s_np = np.array(V_snapped)
                W_s_np = np.array(W_snapped)
                if _tensor_error(U_s_np, V_s_np, W_s_np, T) < 1e-10:
                    return U_snapped, V_snapped, W_snapped, True
        else:
            # Revert if error increased too much
            if new_err > best_err * 1.5:
                U_np = best_U.copy()
                V_np = best_V.copy()
                W_np = best_W.copy()
                vU *= 0.5
                vV *= 0.5
                vW *= 0.5
                lr *= 0.8
            else:
                # Small increase, just dampen
                vU *= 0.9
                vV *= 0.9
                vW *= 0.9
        
        # Decay learning rate slowly
        lr *= 0.9998
        
        # Periodic restart of momentum if stuck
        if iteration % 500 == 0 and iteration > 0:
            vU *= 0.5
            vV *= 0.5
            vW *= 0.5
    
    # Final snap attempt
    U_snapped, V_snapped, W_snapped = _snap_to_half_int(
        best_U.tolist(), best_V.tolist(), best_W.tolist()
    )
    U_s_np = np.array(U_snapped)
    V_s_np = np.array(V_snapped)
    W_s_np = np.array(W_snapped)
    if _tensor_error(U_s_np, V_s_np, W_s_np, T) < 1e-10:
        return U_snapped, V_snapped, W_snapped, True
    
    return U_snapped, V_snapped, W_snapped, False

def _random_reparametrization(U, V, W, rank, n, m, p, T):
    """
    Apply random linear transformations to the factor spaces to escape local minima.
    This is a change of basis in the rank-r decomposition space.
    """
    # Generate random invertible matrices
    A = np.random.randn(rank, rank)
    B = np.random.randn(rank, rank)
    C = np.random.randn(rank, rank)
    
    # Ensure they are invertible (determinant not too small)
    for _ in range(10):
        if abs(np.linalg.det(A)) > 1e-6 and abs(np.linalg.det(B)) > 1e-6 and abs(np.linalg.det(C)) > 1e-6:
            break
        A = np.random.randn(rank, rank)
        B = np.random.randn(rank, rank)
        C = np.random.randn(rank, rank)
    
    # Apply transformations: U' = U @ A, V' = V @ B, W' = W @ C
    # But we need U' @ A^{-1} @ B^{-1} @ C^{-1} to preserve the tensor
    # Actually, for tensor decomposition T = sum_r U[:,r] V[:,r] W[:,r],
    # if we do U' = U @ G, V' = V @ G^{-1}, W' = W @ I, it preserves T
    # More generally, U' = U @ G1, V' = V @ G2, W' = W @ G3 where G1 G2 G3 = I
    
    # Simpler: just perturb the columns slightly
    U_pert = U + 0.1 * np.random.randn(U.shape)
    V_pert = V + 0.1 * np.random.randn(V.shape)
    W_pert = W + 0.1 * np.random.randn(W.shape)
    
    return U_pert, V_pert, W_pert

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s kill
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    T = _build_tensor(n, m, p)
    
    # Try ranks from 24 down to 18
    for target_rank in [24, 23, 22, 21, 20, 19, 18]:
        if time.time() > deadline:
            break
        
        # If rank is 24, it's already exact with schoolbook
        if target_rank == 24:
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
        # Initialize with schoolbook truncated to target_rank
        U, V, W = _schoolbook(n, m, p)
        U = U[:, :target_rank]
        V = V[:, :target_rank]
        W = W[:, :target_rank]
        
        initial_err = _tensor_error(U, V, W, T)
        
        # If rank is 24, it's already exact
        if initial_err < 1e-10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try multiple random initializations and gradient-based refinement
        best_result = None
        best_err = initial_err
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Create a perturbed initialization
            U_init = U + 0.05 * np.random.randn(U.shape)
            V_init = V + 0.05 * np.random.randn(V.shape)
            W_init = W + 0.05 * np.random.randn(W.shape)
            
            # Also try the original
            if attempt == 0:
                U_init, V_init, W_init = U, V, W
            
            init_err = _tensor_error(U_init, V_init, W_init, T)
            
            # Try gradient-based refinement
            U_out, V_out, W_out, success = _try_polish_gradient(
                U_init, V_init, W_init, T, target_rank, deadline, init_err,
                max_iters=5000, seed=42 + attempt
            )
            
            if success:
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # Check if we found a better (but not exact) solution
            final_err = _tensor_error(np.array(U_out), np.array(V_out), np.array(W_out), T)
            if final_err < best_err:
                best_err = final_err
                best_result = (U_out, V_out, W_out)
        
        # If we found a good approximation, try to snap it
        if best_result is not None:
            U_out, V_out, W_out = best_result
            U_snapped, V_snapped, W_snapped = _snap_to_half_int(U_out, V_out, W_out)
            U_s_np = np.array(U_snapped)
            V_s_np = np.array(V_snapped)
            W_s_np = np.array(W_snapped)
            if _tensor_error(U_s_np, V_s_np, W_s_np, T) < 1e-10:
                return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
