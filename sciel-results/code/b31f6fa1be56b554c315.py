"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check dimensions
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_val = 1.0
                decomp_val = 0.0
                for c in range(rank):
                    decomp_val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(tensor_val - decomp_val) > tol:
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

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _reconstruct_tensor(U, V, W, rank):
    """Reconstruct the tensor from factor matrices."""
    T = np.zeros((len(U), len(V), len(W)))
    for t in range(rank):
        T += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
    return T

def _try_gradient_descent(T_target, n, m, p, target_rank, num_trials=10, max_iter=500, deadline=None):
    """Try to find a rank-target_rank decomposition using gradient descent."""
    np.random.seed(42)
    
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            return None
        
        # Random initialization with small values
        U_t = np.random.randn(n*m, target_rank) * 0.1
        V_t = np.random.randn(m*p, target_rank) * 0.1
        W_t = np.random.randn(p*n, target_rank) * 0.1
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            # Compute reconstruction
            T_recon = np.zeros_like(T_target)
            for t in range(target_rank):
                T_recon += np.outer(np.outer(U_t[:, t], V_t[:, t]), W_t[:, t])
            
            error = T_target - T_recon
            loss = np.sum(error**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U_t)
            dV = np.zeros_like(V_t)
            dW = np.zeros_like(W_t)
            for t in range(target_rank):
                dU[:, t] = np.einsum('abc,bc->a', error, V_t[:, t] * W_t[:, t])
                dV[:, t] = np.einsum('abc,ac->b', error, U_t[:, t] * W_t[:, t])
                dW[:, t] = np.einsum('abc,ab->c', error, U_t[:, t] * V_t[:, t])
            
            # Update with adaptive learning rate
            U_t -= lr * dU
            V_t -= lr * dV
            W_t -= lr * dW
            
            # Adaptive learning rate
            if it % 20 == 0 and it > 0:
                lr *= 0.9
            
            # Periodic restart if stuck
            if it % 100 == 99 and loss > 1e-6:
                lr = 0.01
                U_t += np.random.randn(n*m, target_rank) * 0.01
                V_t += np.random.randn(m*p, target_rank) * 0.01
                W_t += np.random.randn(p*n, target_rank) * 0.01
        
        # Final check
        T_recon = np.zeros_like(T_target)
        for t in range(target_rank):
            T_recon += np.outer(np.outer(best_U[:, t], best_V[:, t]), best_W[:, t])
        
        if np.allclose(T_recon, T_target, atol=1e-10, rtol=0):
            return best_U, best_V, best_W
    
    return None

def _try_structured_decomposition(n, m, p, deadline=None):
    """
    Try to construct a structured decomposition by exploiting the block structure.
    For 3x3 x 3x6, we can view B as [B1, B2] where B1, B2 are 3x3.
    Then C = [A*B1, A*B2], which is two 3x3x3 multiplications.
    
    We try to find a rank-r decomposition for 3x3x3 and combine two copies.
    If we can find rank 23 for 3x3x3, total rank is 46.
    """
    if deadline and time.time() > deadline:
        return None
    
    # Try to find a low-rank decomposition for 3x3x3
    # Then combine two copies for the 3x3x6 case
    
    # For 3x3x3, the tensor has 27 entries
    T_333 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_333[i*3+j, j*3+k, k*3+i] = 1.0
    
    # Try ranks from 23 down to 20 for 3x3x3
    for target_rank_333 in range(23, 19, -1):
        if deadline and time.time() > deadline:
            return None
        
        result = _try_gradient_descent(T_333, 3, 3, 3, target_rank_333, 
                                       num_trials=5, max_iter=300, deadline=deadline)
        if result is not None:
            U_333, V_333, W_333 = result
            
            # Combine two copies for 3x3x6
            # U for 3x3x6: shape (9, 2*target_rank_333)
            # First 9 columns from first copy, next 9 from second copy
            U_336 = np.zeros((9, 2*target_rank_333))
            V_336 = np.zeros((18, 2*target_rank_333))
            W_336 = np.zeros((18, 2*target_rank_333))
            
            # First copy handles B1 (columns 0-2 of B, i.e., k=0,1,2)
            U_336[:, :target_rank_333] = U_333
            # V for first copy: rows j*p+k where k=0,1,2 -> rows 0,1,2, 3,4,5, 6,7,8
            V_336[:9, :target_rank_333] = V_333
            # W for first copy: rows k*n+i where k=0,1,2 -> rows 0,1,2, 3,4,5, 6,7,8
            W_336[:9, :target_rank_333] = W_333
            
            # Second copy handles B2 (columns 3-5 of B, i.e., k=3,4,5)
            U_336[:, target_rank_333:] = U_333
            # V for second copy: rows j*p+k where k=3,4,5 -> rows 9,10,11, 12,13,14, 15,16,17
            V_336[9:, target_rank_333:] = V_333
            # W for second copy: rows k*n+i where k=3,4,5 -> rows 9,10,11, 12,13,14, 15,16,17
            W_336[9:, target_rank_333:] = W_333
            
            # Verify
            T_target = _build_target_tensor(3, 3, 6)
            T_recon = np.zeros((9, 18, 18))
            for t in range(2*target_rank_333):
                T_recon += np.outer(np.outer(U_336[:, t], V_336[:, t]), W_336[:, t])
            
            if np.allclose(T_recon, T_target, atol=1e-10, rtol=0):
                return U_336, V_336, W_336, 2*target_rank_333
    
    return None

def _try_direct_search(n, m, p, deadline=None):
    """Try to find a low-rank decomposition directly for 3x3x6."""
    T_target = _build_target_tensor(n, m, p)
    
    # Try ranks from 46 down to 40
    for target_rank in range(46, 39, -1):
        if deadline and time.time() > deadline:
            return None
        
        result = _try_gradient_descent(T_target, n, m, p, target_rank,
                                       num_trials=3, max_iter=200, deadline=deadline)
        if result is not None:
            U, V, W = result
            return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Best known: schoolbook gives rank 54
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured decomposition first (exploit block structure)
    result = _try_structured_decomposition(n, m, p, deadline)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try direct search if we have time
    if time.time() < deadline:
        result = _try_direct_search(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best

# EVOLVE-BLOCK-END
