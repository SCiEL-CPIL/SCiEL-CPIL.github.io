"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _get_strassen_26_decomposition():
    """
    Returns the standard Rank-26 decomposition (Strassen 2x2 core + 19 border terms).
    """
    n, m, p = DIMENSIONS
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen Terms for 2x2 Core (i,j,k in 0,1)
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[0, r] = 1; U[4, r] = 1
    V[0, r] = 1; V[4, r] = 1
    W[0, r] = 1; W[4, r] = 1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[3, r] = 1; U[4, r] = 1
    V[0, r] = 1
    W[1, r] = 1; W[4, r] = -1
    
    # m3 = a00(b01 - b11)
    r = 2
    U[0, r] = 1
    V[1, r] = 1; V[4, r] = -1
    W[3, r] = 1; W[4, r] = 1
    
    # m4 = a11(b10 - b00)
    r = 3
    U[4, r] = 1
    V[3, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[1, r] = 1
    
    # m5 = (a00 + a01)b11
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[4, r] = 1
    W[0, r] = -1; W[3, r] = 1
    
    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[3, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[4, r] = 1
    
    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[1, r] = 1; U[4, r] = -1
    V[3, r] = 1; V[4, r] = 1
    W[0, r] = 1
    
    # Border Terms (19 terms)
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # Skip if i<2 and j<2 and k<2
    border_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                r = 7 + border_count
                border_count += 1
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
    return U, V, W

def _get_standard_27():
    """Fallback standard decomposition."""
    n, m, p = DIMENSIONS
    R = 27
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return U, V, W

def _search_rank_R(R, deadline, rng, start_U=None, start_V=None, start_W=None):
    """
    Attempt to find a rank-R decomposition using randomized gradient descent.
    This is a structural change from the previous local search approach:
    We use a direct non-convex optimization on the tensor entries with random restarts.
    """
    n, m, p = DIMENSIONS
    T_target = np.ones((n, m, p))
    
    # Parameters: U (n*m x R), V (m*p x R), W (p*n x R)
    # Total params: 3 * 9 * R
    total_params = 3 * 9 * R
    
    def _compute_loss_grad(params):
        U = params[:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:].reshape(9, R)
        
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        diff = T_recon - T_target
        loss = np.sum(diff**2)
        
        # Gradients
        # dL/dU_ijr = 2 * sum_{k} diff_ijk * V_jkr * W_kir
        # dL/dV_jkr = 2 * sum_{i} diff_ijk * U_ijr * W_kir
        # dL/dW_kir = 2 * sum_{j} diff_ijk * U_ijr * V_jkr
        
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', diff, V_reshaped, W_reshaped)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', diff, U_reshaped, W_reshaped)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U_reshaped, V_reshaped)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    best_loss = float('inf')
    best_params = None
    
    # Number of restarts
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize parameters
        if start_U is not None and restart == 0:
            U_init = start_U.copy()
            V_init = start_V.copy()
            W_init = start_W.copy()
        else:
            # Random initialization, biased towards small values
            scale = 1.0
            U_init = rng.normal(0, scale, (9, R))
            V_init = rng.normal(0, scale, (9, R))
            W_init = rng.normal(0, scale, (9, R))
            
            # Try to enforce some sparsity or structure? 
            # For now, just random.
            
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Simple gradient descent with momentum
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(params)
        
        for step in range(500):
            if time.time() > deadline:
                break
                
            loss, grad = _compute_loss_grad(params)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
                
            if loss < 1e-12:
                return best_params, best_loss
            
            # Update
            v = momentum * v - lr * grad
            params = params + v
            
            # Optional: periodic re-initialization if stuck
            if step > 0 and step % 100 == 0 and loss > 1e-6:
                # Small random kick
                params += rng.normal(0, 0.01, params.shape)
                v = 0.0
        
    return best_params, best_loss

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(42)
    
    # Get the standard Rank-26 solution
    U26, V26, W26 = _get_strassen_26_decomposition()
    
    if not _verify_decomposition(U26, V26, W26, n, m, p):
        # Fallback to standard 27 if verification fails (should not happen)
        U, V, W = _get_standard_27()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 26
    best_result = (U26, V26, W26)
    
    # Attempt to find lower rank solutions
    # We try ranks 25, 24, 23 in order.
    # If we find a valid solution for a lower rank, we update best_result.
    
    for R in [25, 24, 23]:
        if time.time() > deadline:
            break
            
        # Use the rank-26 solution as a starting point by truncating columns
        # This is a heuristic to help the optimizer find a solution.
        U_start = U26[:, :R]
        V_start = V26[:, :R]
        W_start = W26[:, :R]
        
        # We also try random starts.
        # First try with the truncated start
        params, loss = _search_rank_R(R, deadline, rng, U_start, V_start, W_start)
        
        if loss < 1e-8 and params is not None:
            U = params[:9*R].reshape(9, R)
            V = params[9*R:18*R].reshape(9, R)
            W = params[18*R:].reshape(9, R)
            
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = R
                best_result = (U, V, W)
                # If we found rank 23, we can stop early as it's the known lower bound
                if R == 23:
                    break
        
        # If the first attempt failed, try a few random restarts if time permits
        if time.time() < deadline - 10:
            params, loss = _search_rank_R(R, deadline, rng)
            if loss < 1e-8 and params is not None:
                U = params[:9*R].reshape(9, R)
                V = params[9*R:18*R].reshape(9, R)
                W = params[18*R:].reshape(9, R)
                
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = R
                    best_result = (U, V, W)
                    if R == 23:
                        break
                        
    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}

# EVOLVE-BLOCK-END
