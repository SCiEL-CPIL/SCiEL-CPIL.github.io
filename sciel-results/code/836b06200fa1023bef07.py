import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return U, V, W, rank

def _try_rank_search(n, m, p, target_rank, time_limit=10.0):
    """
    Try to find a decomposition of the given rank using random restarts and gradient descent.
    """
    start_time = time.time()
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build target tensor
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_loss = float('inf')
    best_params = None
    
    # Try multiple restarts
    max_restarts = 50
    steps_per_restart = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = np.random.randn(n_m, target_rank) * 0.5
        V = np.random.randn(m_p, target_rank) * 0.5
        W = np.random.randn(p_n, target_rank) * 0.5
        
        # Adam optimizer
        lr = 0.05
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        for step in range(steps_per_restart):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_dec - T
            loss = np.sum(diff ** 2)
            
            if loss < 1e-10:
                return U, V, W, target_rank
            
            # Gradients
            dU = np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Adam update
            t += 1
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            
            m_U = b1 * m_U + (1 - b1) * dU
            v_U = b2 * v_U + (1 - b2) * (dU ** 2)
            m_U_hat = m_U / (1 - b1 ** t)
            v_U_hat = v_U / (1 - b2 ** t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = b1 * m_V + (1 - b1) * dV
            v_V = b2 * v_V + (1 - b2) * (dV ** 2)
            m_V_hat = m_V / (1 - b1 ** t)
            v_V_hat = v_V / (1 - b2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = b1 * m_W + (1 - b1) * dW
            v_W = b2 * v_W + (1 - b2) * (dW ** 2)
            m_W_hat = m_W / (1 - b1 ** t)
            v_W_hat = v_W / (1 - b2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())
        
        if best_loss < 1e-10:
            return best_params[0], best_params[1], best_params[2], target_rank

    if best_params and best_loss < 1e-10:
        return best_params[0], best_params[1], best_params[2], target_rank
        
    return None

def _try_block_decomposition():
    """
    Exploit the 3×2 block structure: split B (3x6) into three 3x2 blocks.
    For each 3x2 block, we multiply A (2x3) by B_block (3x2) to get C_block (2x2).
    
    If we can find a rank-r decomposition for 2x3x2, then for 2x3x6 we need
    rank = 3*r (one independent copy per block, since the blocks are independent).
    
    The key question: what is the minimal rank for 2x3x2?
    - Schoolbook: 2*3*2 = 12
    - Known: rank(2x3x2) is at most 10 (from various constructions)
    - Let's try to find rank 8 or 10 for 2x3x2 via optimization
    
    Actually, let's think more carefully. For 2x3x2:
    - n=2, m=3, p=2
    - Input sizes: A is 2x3 (6 entries), B is 3x2 (6 entries), C is 2x2 (4 entries)
    - The tensor has 4 non-zero entries (one for each output element)
    
    Let's try to find a rank-10 decomposition for 2x3x2, which would give
    rank 30 for 2x3x6. Or rank 8 giving rank 24.
    """
    n, m, p = 2, 3, 2  # Subproblem dimensions
    
    # Target tensor for 2x3x2
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from 8 upward
    for R in [8, 9, 10, 11, 12]:
        # Multiple random restarts
        for seed in range(50):
            np.random.seed(seed)
            
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Adam optimization
            lr = 0.05
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            m_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_U = np.zeros_like(U)
            v_V = np.zeros_like(V)
            v_W = np.zeros_like(W)
            
            best_loss = np.inf
            converged = False
            
            for it in range(5000):
                T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
                diff = T_dec - T
                loss = np.sum(diff ** 2)
                
                if loss < 1e-10:
                    converged = True
                    break
                
                best_loss = min(best_loss, loss)
                
                # Gradients
                dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
                dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
                dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
                
                # Adam update
                it1 = it + 1
                m_U = beta1 * m_U + (1 - beta1) * dU
                m_V = beta1 * m_V + (1 - beta1) * dV
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_U = beta2 * v_U + (1 - beta2) * dU ** 2
                v_V = beta2 * v_V + (1 - beta2) * dV ** 2
                v_W = beta2 * v_W + (1 - beta2) * dW ** 2
                m_U_hat = m_U / (1 - beta1 ** it1)
                m_V_hat = m_V / (1 - beta1 ** it1)
                m_W_hat = m_W / (1 - beta1 ** it1)
                v_U_hat = v_U / (1 - beta2 ** it1)
                v_V_hat = v_V / (1 - beta2 ** it1)
                v_W_hat = v_W / (1 - beta2 ** it1)
                
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if converged:
                # Verify
                T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_dec, T, atol=1e-10):
                    # Build full 2x3x6 decomposition
                    n_full, m_full, p_full = 2, 3, 6
                    R_full = 3 * R
                    
                    U_full = np.zeros((n_full * m_full, R_full))
                    V_full = np.zeros((m_full * p_full, R_full))
                    W_full = np.zeros((p_full * n_full, R_full))
                    
                    for block in range(3):
                        offset = block * R
                        U_full[:, offset:offset+R] = U
                        
                        for j in range(m_full):
                            for k in range(2):
                                V_full[j * p_full + (block * 2 + k), offset:offset+R] = V[j * 2 + k]
                        
                        for k in range(2):
                            for i in range(n_full):
                                W_full[(block * 2 + k) * n_full + i, offset:offset+R] = W[k * n_full + i]
                    
                    if _verify_decomposition(U_full, V_full, W_full, n_full, m_full, p_full):
                        return U_full, V_full, W_full, R_full
    
    return None

def _try_direct_optimization():
    """
    Try direct optimization on the full 2x3x6 problem with various ranks.
    """
    n, m, p = 2, 3, 6
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from 20 upward (schoolbook is 36)
    for R in [20, 22, 24, 26, 28, 30]:
        for seed in range(20):
            np.random.seed(seed)
            
            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3
            
            lr = 0.03
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            m_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_U = np.zeros_like(U)
            v_V = np.zeros_like(V)
            v_W = np.zeros_like(W)
            
            converged = False
            
            for it in range(3000):
                T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
                diff = T_dec - T
                loss = np.sum(diff ** 2)
                
                if loss < 1e-10:
                    converged = True
                    break
                
                # Gradients
                dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
                dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
                dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
                
                # Adam update
                it1 = it + 1
                m_U = beta1 * m_U + (1 - beta1) * dU
                m_V = beta1 * m_V + (1 - beta1) * dV
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_U = beta2 * v_U + (1 - beta2) * dU ** 2
                v_V = beta2 * v_V + (1 - beta2) * dV ** 2
                v_W = beta2 * v_W + (1 - beta2) * dW ** 2
                m_U_hat = m_U / (1 - beta1 ** it1)
                m_V_hat = m_V / (1 - beta1 ** it1)
                m_W_hat = m_W / (1 - beta1 ** it1)
                v_U_hat = v_U / (1 - beta2 ** it1)
                v_V_hat = v_V / (1 - beta2 ** it1)
                v_W_hat = v_W / (1 - beta2 ** it1)
                
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if converged:
                if _verify_decomposition(U, V, W, n, m, p):
                    return U, V, W, R
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    best_solution = None
    best_rank = n * m * p  # 36
    
    # Strategy 1: Try block decomposition (2x3x2 subproblem)
    if time.time() < deadline - 60:
        block_result = _try_block_decomposition()
        if block_result is not None:
            U, V, W, rank = block_result
            if rank < best_rank:
                best_solution = (U, V, W, rank)
                best_rank = rank
    
    # Strategy 2: Try direct optimization on full problem
    if best_solution is None and time.time() < deadline - 30:
        direct_result = _try_direct_optimization()
        if direct_result is not None:
            U, V, W, rank = direct_result
            if rank < best_rank:
                best_solution = (U, V, W, rank)
                best_rank = rank
    
    # Strategy 3: Try generic rank search with lower ranks
    if best_solution is None:
        ranks_to_try = [12, 14, 16, 18, 20, 24]
        for R in ranks_to_try:
            if time.time() > deadline - 20:
                break
                
            time_slice = 10.0
            if R == 12:
                time_slice = 20.0
            
            result = _try_rank_search(n, m, p, R, time_limit=time_slice)
            
            if result is not None:
                U, V, W, rank = result
                if _verify_decomposition(U, V, W, n, m, p):
                    best_solution = (U, V, W, rank)
                    best_rank = rank
                    break
    
    # If no improvement found, fall back to schoolbook
    if best_solution is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best_solution = (U, V, W, rank)
        best_rank = rank
    
    U, V, W, R = best_solution
    
    # Convert to lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
