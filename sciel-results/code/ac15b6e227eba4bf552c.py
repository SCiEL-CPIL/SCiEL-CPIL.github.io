"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    if not np.allclose(S, 1.0, atol=tol):
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

def _loss(U, V, W, n, m, p):
    """Compute ||S - 1||^2."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    diff = S - 1.0
    return np.sum(diff ** 2)

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to round to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol):
        return U_r, V_r, W_r
    return None

def _local_search(U, V, W, n, m, p, max_iter=5000, deadline=None):
    """
    Coordinate descent local search. For each coordinate, try small perturbations
    and keep the best. This works well when the solution is close to half-integers.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U_arr[0])
    
    best_loss = _loss(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p)
    
    # Perturbation candidates: try values in a local grid
    # Focus on coordinates that are close to half-integers
    perturbation_sizes = [0.5, 0.25, 0.125, 0.0625]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try perturbing each coordinate of U, V, W
        for param_name, arr in [('U', U_arr), ('V', V_arr), ('W', W_arr)]:
            rows, cols = arr.shape
            for r in range(rows):
                for c in range(cols):
                    old_val = arr[r, c]
                    if abs(old_val) < 1e-10:
                        continue  # Skip zeros, focus on non-zero entries
                    
                    # Try small perturbations
                    for delta in perturbation_sizes:
                        for sign in [1, -1]:
                            arr[r, c] = old_val + sign * delta
                            new_loss = _loss(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p)
                            if new_loss < best_loss - 1e-12:
                                best_loss = new_loss
                                improved = True
                            else:
                                arr[r, c] = old_val
        
        if not improved:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), best_loss

def _gaussian_elimination_merge(U, V, W, n, m, p, target_rank, deadline=None):
    """
    Try to find a rank-`target_rank` decomposition by starting from a larger rank
    and using linear algebra to merge terms.
    
    Approach: We know the tensor T[i,j,k] = 1 for all (i,j,k).
    We want to find U, V, W with rank columns such that:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    Strategy: Use a randomized approach with rank-revealing decomposition.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Start with schoolbook and try to reduce rank
    sb = _schoolbook(n, m, p)
    U_sb = np.array(sb['U'], dtype=float)
    V_sb = np.array(sb['V'], dtype=float)
    W_sb = np.array(sb['W'], dtype=float)
    
    # Build the tensor as a vector
    tensor_vec = np.ones(n * m * p)
    
    # For each candidate rank, try to find a decomposition
    # Use a greedy approach: pick columns that cover the most uncovered tensor entries
    
    # Actually, let's use a different approach: 
    # Randomly initialize and use Adam with restarts
    best_result = None
    best_loss = np.inf
    
    num_trials = 30
    for trial in range(num_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(trial * 7919 + target_rank)
        
        # Initialize with sparse random patterns
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Each column gets a random sparse pattern
        for c in range(target_rank):
            # Randomly choose which (i,j) positions are non-zero in U
            num_nonzero_U = max(1, min(n * m, np.random.randint(1, 4)))
            idx_U = np.random.choice(n * m, size=num_nonzero_U, replace=False)
            U[idx_U, c] = np.random.choice([0.5, 1.0, -0.5, -1.0], size=num_nonzero_U)
            
            num_nonzero_V = max(1, min(m * p, np.random.randint(1, 4)))
            idx_V = np.random.choice(m * p, size=num_nonzero_V, replace=False)
            V[idx_V, c] = np.random.choice([0.5, 1.0, -0.5, -1.0], size=num_nonzero_V)
            
            num_nonzero_W = max(1, min(p * n, np.random.randint(1, 4)))
            idx_W = np.random.choice(p * n, size=num_nonzero_W, replace=False)
            W[idx_W, c] = np.random.choice([0.5, 1.0, -0.5, -1.0], size=num_nonzero_W)
        
        # Run Adam optimization
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        m_U = np.zeros_like(U_t)
        v_U = np.zeros_like(U_t)
        m_V = np.zeros_like(V_t)
        v_V = np.zeros_like(V_t)
        m_W = np.zeros_like(W_t)
        v_W = np.zeros_like(W_t)
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = 0.02
        t = 0
        best_local_loss = np.inf
        best_U = U_t.copy()
        best_V = V_t.copy()
        best_W = W_t.copy()
        
        max_iter = 1500
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            t += 1
            S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            diff = S - 1.0
            loss = np.sum(diff ** 2)
            
            if loss < best_local_loss:
                best_local_loss = loss
                best_U = U_t.copy()
                best_V = V_t.copy()
                best_W = W_t.copy()
            
            if loss < 1e-10:
                break
            
            dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_t, W_t)
            dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_t, W_t)
            dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_t, V_t)
            
            m_U = beta1 * m_U + (1 - beta1) * dU
            v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
            U_t -= lr * (m_U / (1 - beta1 ** t)) / (np.sqrt(v_U / (1 - beta2 ** t)) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
            V_t -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
            W_t -= lr * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)
        
        if best_local_loss < best_loss:
            best_loss = best_local_loss
            U_opt = best_U.reshape(n*m, -1)
            V_opt = best_V.reshape(m*p, -1)
            W_opt = best_W.reshape(p*n, -1)
            
            # Try to round
            result = _try_round_and_verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8)
            if result is not None:
                return {'U': result[0], 'V': result[1], 'W': result[2]}
            
            # If close but not exact, do local search
            if best_local_loss < 1e-6:
                U_ls, V_ls, W_ls, ls_loss = _local_search(
                    U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p,
                    max_iter=2000, deadline=deadline
                )
                if ls_loss < 1e-8:
                    result = _try_round_and_verify(U_ls, V_ls, W_ls, n, m, p, tol=1e-8)
                    if result is not None:
                        return {'U': result[0], 'V': result[1], 'W': result[2]}
    
    return None

def _structured_sparse_init(n, m, p, rank, seed=None):
    """
    Create a structured sparse initialization that respects the matrix multiplication structure.
    Each column corresponds to a "pattern" that covers multiple (i,j,k) tuples.
    """
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each column, create a pattern that tries to cover multiple tensor entries
    # The key: for matrix multiplication, U[i,j] * V[j,k] * W[k,i] should contribute
    # to the (i,j,k) entry.
    
    for c in range(rank):
        # Choose a "center" (i0, j0, k0)
        i0 = c % n
        j0 = (c // n) % m
        k0 = (c // (n * m)) % p
        
        # U: non-zero for (i0, j0) and possibly neighbors
        U[i0 * m + j0, c] = 1.0
        if m > 1:
            j1 = (j0 + 1) % m
            U[i0 * m + j1, c] = np.random.choice([0.5, -0.5])
        
        # V: non-zero for (j0, k0) and possibly neighbors
        V[j0 * p + k0, c] = 1.0
        if p > 1:
            k1 = (k0 + 1) % p
            V[j0 * p + k1, c] = np.random.choice([0.5, -0.5])
        
        # W: non-zero for (k0, i0) and possibly neighbors
        W[k0 * n + i0, c] = 1.0
        if n > 1:
            i1 = (i0 + 1) % n
            W[k0 * n + i1, c] = np.random.choice([0.5, -0.5])
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to best_rank - 1
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Method 1: Gaussian elimination merge with Adam
        result = _gaussian_elimination_merge(n, m, p, target_rank, deadline=deadline - 10)
        if result is not None:
            best = result
            best_rank = target_rank
            found = True
            break
        
        # Method 2: Structured sparse init + optimization
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            seed = target_rank * 10000 + attempt
            U_init, V_init, W_init = _structured_sparse_init(n, m, p, target_rank, seed=seed)
            
            # Quick Adam optimization
            U_t = np.array(U_init, dtype=float).reshape(n, m, -1)
            V_t = np.array(V_init, dtype=float).reshape(m, p, -1)
            W_t = np.array(W_init, dtype=float).reshape(p, n, -1)
            
            m_U = np.zeros_like(U_t)
            v_U = np.zeros_like(U_t)
            m_V = np.zeros_like(V_t)
            v_V = np.zeros_like(V_t)
            m_W = np.zeros_like(W_t)
            v_W = np.zeros_like(W_t)
            
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            lr = 0.03
            t = 0
            best_local_loss = np.inf
            best_U = U_t.copy()
            best_V = V_t.copy()
            best_W = W_t.copy()
            
            for it in range(1000):
                if time.time() > deadline - 10:
                    break
                
                t += 1
                S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                diff = S - 1.0
                loss = np.sum(diff ** 2)
                
                if loss < best_local_loss:
                    best_local_loss = loss
                    best_U = U_t.copy()
                    best_V = V_t.copy()
                    best_W = W_t.copy()
                
                if loss < 1e-10:
                    break
                
                dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_t, W_t)
                dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_t, W_t)
                dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_t, V_t)
                
                m_U = beta1 * m_U + (1 - beta1) * dU
                v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
                U_t -= lr * (m_U / (1 - beta1 ** t)) / (np.sqrt(v_U / (1 - beta2 ** t)) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * dV
                v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
                V_t -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
                W_t -= lr * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)
            
            if best_local_loss < 1e-8:
                U_opt = best_U.reshape(n*m, -1).tolist()
                V_opt = best_V.reshape(m*p, -1).tolist()
                W_opt = best_W.reshape(p*n, -1).tolist()
                
                result = _try_round_and_verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-8)
                if result is not None:
                    best = {'U': result[0], 'V': result[1], 'W': result[2]}
                    best_rank = target_rank
                    found = True
                    break
        
        if best_rank == target_rank:
            break
    
    return best
