"""
Improved search for minimal rank tensor decomposition of 3x4x7 matrix multiplication.
Uses structured initialization and efficient local search.
"""
import time
import numpy as np
import random

# Target dimensions
n, m, p = 3, 4, 7

# Pre-compute the target tensor
T_target = np.ones((n, m, p), dtype=np.float64)

def compute_error(U, V, W):
    """Compute squared error of decomposition."""
    r = U.shape[1]
    U_resh = U.reshape(n, m, r)
    V_resh = V.reshape(m, p, r)
    W_resh = W.reshape(p, n, r)
    
    # Use batched matrix multiplication for efficiency
    # approx[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We can compute this more efficiently by fixing k and using matrix mult
    approx = np.zeros((n, m, p), dtype=np.float64)
    for k in range(p):
        # For fixed k: approx[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Let A[i,j] = sum_c U[i,j,c] * V[j,k,c]
        # Let B[i] = W[k,i,c] for each c
        # Actually, let's use einsum with optimized path
        pass
    
    # Better: use the original einsum but with optimize=True
    approx = np.einsum('ijc,jkc,kic->ijk', U_resh, V_resh, W_resh, optimize=True)
    error = np.sum((approx - T_target) ** 2)
    return error

def snap_to_half_int(matrix):
    """Snap values to nearest multiple of 0.5."""
    return np.round(matrix * 2.0) / 2.0

def verify_decomposition(U, V, W):
    """Check if decomposition is exact."""
    return compute_error(U, V, W) < 1e-10

def initialize_structured(r, seed):
    """
    Initialize with structured random values based on known constructions.
    Uses DFT-inspired structure for better initialization.
    """
    np.random.seed(seed)
    
    # Use a DFT-inspired initialization
    # For the 4-element dimension (m), use 4-point DFT structure
    # For other dimensions, use random but structured values
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Create DFT matrix for size m=4
    dft_m = np.zeros((m, m), dtype=complex)
    for a in range(m):
        for b in range(m):
            dft_m[a, b] = np.exp(2j * np.pi * a * b / m)
    
    # For each rank component, assign a structured pattern
    for c in range(r):
        # Assign a random "frequency" to this component
        freq = c % m
        
        # U: for each (i,j), set value based on DFT structure
        for i in range(n):
            for j in range(m):
                # Use a combination of DFT and random
                val = np.real(dft_m[j, freq]) + 0.5 * np.random.randn()
                U[i * m + j, c] = val
        
        # V: for each (j,k), set value based on DFT structure
        for j in range(m):
            for k in range(p):
                val = np.real(dft_m[j, freq]) * np.exp(2j * np.pi * k * freq / p).real + 0.5 * np.random.randn()
                V[j * p + k, c] = val
        
        # W: for each (k,i), set value
        for k in range(p):
            for i in range(n):
                val = np.exp(-2j * np.pi * (k * freq / p + i * freq / n)).real + 0.5 * np.random.randn()
                W[k * n + i, c] = val
    
    # Scale to reasonable magnitudes
    U *= 0.5
    V *= 0.5
    W *= 0.5
    
    return U, V, W

def local_search(U, V, W, time_limit, seed):
    """
    Perform local search to improve the decomposition.
    Uses coordinate descent on individual entries.
    """
    random.seed(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    r = U.shape[1]
    best_error = compute_error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute indices for efficiency
    U_indices = [(i * m + j, c) for i in range(n) for j in range(m) for c in range(r)]
    V_indices = [(j * p + k, c) for j in range(m) for k in range(p) for c in range(r)]
    W_indices = [(k * n + i, c) for k in range(p) for i in range(n) for c in range(r)]
    
    step = 0
    while time.time() < deadline:
        # Randomly pick which matrix to update
        mat_choice = random.choice(['U', 'V', 'W'])
        
        if mat_choice == 'U':
            idx, c = random.choice(U_indices)
            old_val = U[idx, c]
            # Try small perturbations
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                U[idx, c] = old_val + delta
                err = compute_error(U, V, W)
                if err < best_error - 1e-12:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[idx, c] = old_val
        elif mat_choice == 'V':
            idx, c = random.choice(V_indices)
            old_val = V[idx, c]
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                V[idx, c] = old_val + delta
                err = compute_error(U, V, W)
                if err < best_error - 1e-12:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[idx, c] = old_val
        else:  # W
            idx, c = random.choice(W_indices)
            old_val = W[idx, c]
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                W[idx, c] = old_val + delta
                err = compute_error(U, V, W)
                if err < best_error - 1e-12:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[idx, c] = old_val
        
        step += 1
        if step % 1000 == 0:
            # Try snapping to half-integers
            U_snap = snap_to_half_int(best_U)
            V_snap = snap_to_half_int(best_V)
            W_snap = snap_to_half_int(best_W)
            if verify_decomposition(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
    
    # Final snap attempt
    if best_error < 1e-3:
        U_snap = snap_to_half_int(best_U)
        V_snap = snap_to_half_int(best_V)
        W_snap = snap_to_half_int(best_W)
        if verify_decomposition(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
    
    return None

def gradient_descent(U, V, W, time_limit):
    """
    Perform gradient descent to reduce error with adaptive learning rate.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    r = U.shape[1]
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    lr = 0.01
    momentum = 0.9
    best_error = compute_error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Track consecutive non-improvements for adaptive lr
    no_improve_count = 0
    patience = 50
    
    while time.time() < deadline:
        U_resh = U.reshape(n, m, r)
        V_resh = V.reshape(m, p, r)
        W_resh = W.reshape(p, n, r)
        
        approx = np.einsum('ijc,jkc,kic->ijk', U_resh, V_resh, W_resh, optimize=True)
        D = approx - T_target
        
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', D, V_resh, W_resh, optimize=True).reshape(n*m, r)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', D, U_resh, W_resh, optimize=True).reshape(m*p, r)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', D, U_resh, V_resh, optimize=True).reshape(p*n, r)
        
        vU = momentum * vU + (1 - momentum) * dU
        vV = momentum * vV + (1 - momentum) * dV
        vW = momentum * vW + (1 - momentum) * dW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        err = compute_error(U, V, W)
        if err < best_error - 1e-12:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            no_improve_count = 0
            if best_error < 1e-4:
                U_snap = snap_to_half_int(best_U)
                V_snap = snap_to_half_int(best_V)
                W_snap = snap_to_half_int(best_W)
                if verify_decomposition(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                lr *= 0.5
                no_improve_count = 0
                # Reset momentum
                vU *= 0.5
                vV *= 0.5
                vW *= 0.5
    
    if best_error < 1e-3:
        U_snap = snap_to_half_int(best_U)
        V_snap = snap_to_half_int(best_V)
        W_snap = snap_to_half_int(best_W)
        if verify_decomposition(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
    
    return None

def search_rank(r, time_limit, seed):
    """
    Attempt to find a decomposition of rank r.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Try multiple initializations with different strategies
    for init_seed in range(5):
        if time.time() > deadline:
            break
        
        U, V, W = initialize_structured(r, seed + init_seed)
        
        # First do gradient descent
        gd_time = time_limit * 0.3
        gd_result = gradient_descent(U, V, W, gd_time)
        if gd_result is not None:
            return gd_result
        
        # Then do local search with the best result from GD
        ls_time = time_limit * 0.2
        ls_result = local_search(U, V, W, ls_time, seed + init_seed)
        if ls_result is not None:
            return ls_result
    
    return None

def schoolbook_decomposition():
    """
    Returns the standard rank 84 decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Entry point. Tries to find a rank < 84 decomposition.
    Falls back to schoolbook if not found within time limit.
    """
    start_time = time.time()
    overall_deadline = 140.0  # Leave 10s margin for safety (150s limit)
    
    # Try ranks from 83 down to 70
    # Based on known bounds, 3x4x7 should be achievable around 70-75
    ranks_to_try = list(range(83, 69, -1))
    
    best_result = None
    
    for r in ranks_to_try:
        if time.time() > overall_deadline:
            break
        
        # Allocate time based on rank - lower ranks get more time
        time_per_rank = 2.5 if r > 75 else 4.0
        if time.time() + time_per_rank > overall_deadline:
            time_per_rank = overall_deadline - time.time()
            if time_per_rank < 1.0:
                break
        
        # Try seeds for this rank
        for seed in range(4):
            if time.time() > overall_deadline:
                break
            res = search_rank(r, time_per_rank / 4.0, seed)
            if res is not None:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no improvement found, return schoolbook
    U, V, W = schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
