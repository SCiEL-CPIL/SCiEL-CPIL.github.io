import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensors: U[i,j,c], V[j,k,c], W[k,i,c]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum for efficiency
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    # Check if T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _snap_to_half_integer(U, V, W):
    """Snap values to the nearest half-integer (0, 0.5, 1, 1.5, ...)."""
    U_snapped = np.round(U * 2) / 2
    V_snapped = np.round(V * 2) / 2
    W_snapped = np.round(W * 2) / 2
    return U_snapped, V_snapped, W_snapped

def _compute_loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for tensor decomposition.
    
    Loss = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
    """
    rank = U.shape[1]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    # Error
    E = T - 1.0
    loss = np.sum(E * E)
    
    # Gradients
    # dL/dU[i,j,c] = 2 * sum_{k} E[i,j,k] * V[j,k,c] * W[k,i,c]
    # dL/dV[j,k,c] = 2 * sum_{i} E[i,j,k] * U[i,j,c] * W[k,i,c]
    # dL/dW[k,i,c] = 2 * sum_{j} E[i,j,k] * U[i,j,c] * V[j,k,c]
    
    # For U: sum over k of E[i,j,k] * V[j,k,c] * W[k,i,c]
    # = einsum('ijk,jkc,kic->ijc', E, V_tensor, W_tensor)
    grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', E, V_tensor, W_tensor)
    
    # For V: sum over i of E[i,j,k] * U[i,j,c] * W[k,i,c]
    # = einsum('ijk,ijc,kic->jkc', E, U_tensor, W_tensor)
    grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', E, U_tensor, W_tensor)
    
    # For W: sum over j of E[i,j,k] * U[i,j,c] * V[j,k,c]
    # = einsum('ijk,ijc,jkc->kic', E, U_tensor, V_tensor)
    grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', E, U_tensor, V_tensor)
    
    return loss, grad_U.reshape(-1, rank), grad_V.reshape(-1, rank), grad_W.reshape(-1, rank)

def _gradient_descent(U, V, W, n, m, p, max_iter=200, lr=0.01, tol=1e-10, deadline=None):
    """Perform gradient descent with adaptive learning rate to minimize the decomposition error."""
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = np.inf
    
    # Momentum with adaptive beta
    v_U, v_V, v_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    beta = 0.85
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        # Update momentum with adaptive beta (increase as we converge)
        progress = min(it / max_iter, 1.0)
        beta_current = beta + 0.1 * progress
        
        v_U = beta_current * v_U + (1 - beta_current) * grad_U
        v_V = beta_current * v_V + (1 - beta_current) * grad_V
        v_W = beta_current * v_W + (1 - beta_current) * grad_W
        
        # Adaptive learning rate with cosine annealing
        grad_norm = max(np.linalg.norm(grad_U), np.linalg.norm(grad_V), np.linalg.norm(grad_W))
        if grad_norm > 0:
            base_lr = lr * 100.0 / (grad_norm + 1e-8)
            # Cosine annealing: start high, end low
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            effective_lr = base_lr * (0.5 + 0.5 * cosine_factor)
            effective_lr = min(effective_lr, 0.5)
        else:
            break
        
        U -= effective_lr * v_U
        V -= effective_lr * v_V
        W -= effective_lr * v_W
    
    return best_U, best_V, best_W, best_loss

def _polish_solution(U, V, W, n, m, p, deadline, tol=1e-14, max_iter=50):
    """High-precision polish pass with very small steps."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = np.inf
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        grad_norm = max(np.linalg.norm(grad_U), np.linalg.norm(grad_V), np.linalg.norm(grad_W))
        if grad_norm < 1e-15:
            break
        
        # Very small step for polishing
        step_size = 0.1 / (grad_norm + 1e-10)
        step_size = min(step_size, 0.01)
        
        U -= step_size * grad_U
        V -= step_size * grad_V
        W -= step_size * grad_W
    
    return best_U, best_V, best_W, best_loss

def _try_random_search(n, m, p, target_rank, num_trials, deadline):
    """Try random initializations and gradient descent to find valid decompositions."""
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        # Random initialization with different scales and sparsity patterns
        scale = np.random.uniform(0.1, 2.0)
        
        # Mix of dense and sparse initializations
        if np.random.random() < 0.3:
            # Sparse initialization
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            for c in range(target_rank):
                # Randomly select a few non-zero entries per column
                n_nonzero_U = np.random.randint(1, min(3, n * m) + 1)
                n_nonzero_V = np.random.randint(1, min(3, m * p) + 1)
                n_nonzero_W = np.random.randint(1, min(3, p * n) + 1)
                
                U_idx = np.random.choice(n * m, n_nonzero_U, replace=False)
                U[U_idx, c] = scale * np.random.randn(n_nonzero_U)
                
                V_idx = np.random.choice(m * p, n_nonzero_V, replace=False)
                V[V_idx, c] = scale * np.random.randn(n_nonzero_V)
                
                W_idx = np.random.choice(p * n, n_nonzero_W, replace=False)
                W[W_idx, c] = scale * np.random.randn(n_nonzero_W)
        else:
            # Dense initialization
            U = scale * np.random.randn(n * m, target_rank)
            V = scale * np.random.randn(m * p, target_rank)
            W = scale * np.random.randn(p * n, target_rank)
        
        # Run gradient descent
        U_opt, V_opt, W_opt, final_loss = _gradient_descent(
            U, V, W, n, m, p, max_iter=150, lr=0.05, tol=1e-12, deadline=deadline
        )
        
        # If close to solution, try polishing
        if final_loss < 1e-4:
            U_opt, V_opt, W_opt, final_loss = _polish_solution(
                U_opt, V_opt, W_opt, n, m, p, deadline, tol=1e-14, max_iter=30
            )
        
        # Check if the continuous solution is close to an exact one
        if final_loss < 1e-6:
            # Try snapping to half-integers
            U_snap, V_snap, W_snap = _snap_to_half_integer(U_opt, V_opt, W_opt)
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
        
        # Also check the continuous solution directly (in case it's effectively exact)
        if final_loss < 1e-10:
            return U_opt, V_opt, W_opt

    return None

def _try_structured_patterns(n, m, p, deadline, best_rank):
    """Try various structured patterns that might give lower rank."""
    # Based on known results, try ranks below the schoolbook
    # For 4x5x5, schoolbook is 100. Try progressively lower ranks.
    candidate_ranks = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations for this rank
        result = _try_random_search(n, m, p, target_rank, 5, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find better decompositions using gradient-based search
    result = _try_structured_patterns(n, m, p, deadline, best_rank)
    if result is not None:
        U_try, V_try, W_try = result
        if U_try.shape[1] < best_rank:
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                best_rank = U_try.shape[1]
    
    # If time remains, try a few more random searches with different seeds
    if time.time() < deadline - 25:
        for seed in [123, 456, 789, 111, 222]:
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            result = _try_random_search(n, m, p, best_rank - 1, 3, deadline)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = U_try.shape[1]
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
