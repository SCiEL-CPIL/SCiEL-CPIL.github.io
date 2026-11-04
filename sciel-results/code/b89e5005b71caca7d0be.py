import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, num_tests=5, seed=42):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        # Efficient check: C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can vectorize this slightly or just loop for clarity in verification
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-8, rtol=1e-8):
            return False
    return True

def _prune_and_verify(U, V, W, n, m, p, threshold=1e-4):
    """
    Identify terms (columns) that are close to zero in all three matrices.
    If a column r has max(abs(U[:,r]), abs(V[:,r]), abs(W[:,r])) < threshold, 
    it contributes negligibly. We can remove it.
    """
    # Calculate magnitude of each term
    # A term is "dead" if its components are small
    # Note: It's possible for a term to be important even if individual components are small 
    # if they multiply to something non-negligible, but in gradient descent for 
    # factorization, dead terms usually decay in all components.
    
    max_u = np.max(np.abs(U), axis=0)
    max_v = np.max(np.abs(V), axis=0)
    max_w = np.max(np.abs(W), axis=0)
    
    # A term is likely zero if all its parts are small
    # Let's be conservative: if the max norm across all three is small, it's dead.
    term_magnitudes = np.maximum(np.maximum(max_u, max_v), max_w)
    
    active_indices = np.where(term_magnitudes > threshold)[0]
    
    if len(active_indices) == 0:
        return None, None, None, 0
        
    U_pruned = U[:, active_indices]
    V_pruned = V[:, active_indices]
    W_pruned = W[:, active_indices]
    
    # Verify the pruned version
    if _verify_decomposition(U_pruned, V_pruned, W_pruned, n, m, p, num_tests=10, seed=99):
        return U_pruned, V_pruned, W_pruned, len(active_indices)
    
    # If pruning fails verification, maybe the threshold was too aggressive 
    # or the solution wasn't fully converged. Return original or try lower threshold?
    # For safety, if pruning breaks it, we return None to signal "no valid low rank found here"
    return None, None, None, 0

def _optimize_decomposition(n, m, p, initial_rank, learning_rate=0.01, steps=2000, time_limit=100):
    """
    Use gradient descent to find a decomposition.
    We start with a random initialization (or structured) and minimize the error.
    """
    start_time = time.time()
    
    # Initialize U, V, W
    # Random initialization often gets stuck in local minima.
    # Let's initialize with the schoolbook rank (n*m*p) and then let it shrink?
    # Or initialize with a rank slightly above the expected minimum.
    # Let's try initializing with rank 48 (schoolbook) and letting it collapse.
    
    # Actually, a better strategy for finding *minimal* rank is to start with a 
    # specific structured guess that is already low rank, or use random init 
    # with a rank close to the target.
    
    # Let's try random init with rank = initial_rank
    U = np.random.randn(n*m, initial_rank) * 0.1
    V = np.random.randn(m*p, initial_rank) * 0.1
    W = np.random.randn(p*n, initial_rank) * 0.1
    
    # To help convergence, we can add a small regularization to encourage sparsity
    # or just rely on the fact that if a term isn't needed, its gradient will push it to 0.
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        # Compute current product
        # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # We need the gradient of the error with respect to U, V, W.
        # Error E = sum_{i,k} (C_computed[i,k] - C_target[i,k])^2
        # This is expensive to compute fully every step for random A,B.
        # Instead, we fix a few test matrices A, B and minimize the error on them.
        
        # Let's use a fixed set of test matrices for the optimization
        rng = np.random.RandomState(0)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_target = A_test @ B_test
        
        # Compute C_computed
        C_computed = np.zeros((n, p))
        # Efficient computation using einsum or loops
        # C[i,k] = sum_j U[i,j,:] . (V[j,k,:] * W[k,i,:])
        # Let's precompute M[j,k,i] = V[j*p+k, :] * W[k*n+i, :] ? No, W depends on i too.
        
        # Let's just do the loops for gradient calculation as it's small dimensions.
        # n=2, m=4, p=6. Rank ~ 30-50.
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Compute C_computed and accumulate gradients
        # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # dE/dU[i*m+j, r] = 2 * (C[i,k] - C_target[i,k]) * V[j*p+k, r] * W[k*n+i, r]
        # Note: U[i*m+j, r] only appears in C[i, k] for all k? No, for a fixed i,j, it appears in all k.
        # Actually, U[i*m+j, r] contributes to C[i, k] for all k.
        
        # Let's compute the error matrix E[i,k] = C_computed[i,k] - C_target[i,k]
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    # Dot product of vectors of length rank
                    u_vec = U[i*m + j, :]
                    v_vec = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    s += np.dot(u_vec, v_vec * w_val)
                C_comp[i, k] = s
        
        E = C_comp - C_target
        
        # Now compute gradients
        # dE/dU[i*m+j, r] = 2 * sum_k E[i,k] * V[j*p+k, r] * W[k*n+i, r]
        # dE/dV[j*p+k, r] = 2 * sum_i E[i,k] * U[i*m+j, r] * W[k*n+i, r]
        # dE/dW[k*n+i, r] = 2 * sum_j E[i,k] * U[i*m+j, r] * V[j*p+k, r]
        
        for i in range(n):
            for k in range(p):
                e_ik = 2 * E[i, k]
                if abs(e_ik) < 1e-8: continue
                
                for j in range(m):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    
                    # Gradient for U[u_idx, r]
                    # dU[u_idx, r] += e_ik * V[v_idx, r] * W[w_idx, r]
                    dU[u_idx, :] += e_ik * V[v_idx, :] * W[w_idx, :]
                    
                    # Gradient for V[v_idx, r]
                    # dV[v_idx, r] += e_ik * U[u_idx, r] * W[w_idx, r]
                    dV[v_idx, :] += e_ik * U[u_idx, :] * W[w_idx, :]
                    
                    # Gradient for W[w_idx, r]
                    # dW[w_idx, r] += e_ik * U[u_idx, r] * V[v_idx, :]
                    dW[w_idx, :] += e_ik * U[u_idx, :] * V[v_idx, :]
        
        # Update
        U -= learning_rate * dU
        V -= learning_rate * dV
        W -= learning_rate * dW
        
        # Optional: Project to encourage sparsity? 
        # Simple thresholding: set small values to 0
        # This can help escape plateaus where small non-zero values persist.
        U[np.abs(U) < 1e-5] = 0
        V[np.abs(V) < 1e-5] = 0
        W[np.abs(W) < 1e-5] = 0
        
        # Check convergence
        if np.max(np.abs(E)) < 1e-6:
            # Converged
            break
            
    # Prune
    U_pruned, V_pruned, W_pruned, rank_found = _prune_and_verify(U, V, W, n, m, p, threshold=1e-3)
    
    if rank_found > 0:
        return U_pruned, V_pruned, W_pruned, rank_found
        
    # If pruning didn't yield a valid low rank, try a more aggressive prune or return None
    # Let's try threshold 1e-4
    U_pruned, V_pruned, W_pruned, rank_found = _prune_and_verify(U, V, W, n, m, p, threshold=1e-4)
    if rank_found > 0:
        return U_pruned, V_pruned, W_pruned, rank_found
        
    return None, None, None, 0

def _build_trivial(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m*p) + j * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    return U, V, W

def _build_block_strassen(n, m, p):
    """Block-strassen decomposition. Rank 42 for 2x4x6."""
    rank = 42
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 2x2x2
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    U_s[0, 0] = 1; U_s[3, 0] = 1
    U_s[2, 1] = 1; U_s[3, 1] = 1
    U_s[0, 2] = 1
    U_s[3, 3] = 1
    U_s[0, 4] = 1; U_s[1, 4] = 1
    U_s[2, 5] = 1; U_s[0, 5] = -1
    U_s[1, 6] = 1; U_s[3, 6] = -1
    
    V_s[0, 0] = 1; V_s[3, 0] = 1
    V_s[0, 1] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    V_s[3, 4] = 1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    
    W_s[0, 0] = 1; W_s[0, 3] = 1; W_s[0, 4] = -1; W_s[0, 6] = 1
    W_s[1, 1] = 1; W_s[1, 3] = 1
    W_s[2, 2] = 1; W_s[2, 4] = 1
    W_s[3, 0] = 1; W_s[3, 1] = -1; W_s[3, 2] = 1; W_s[3, 5] = 1
    
    for a_block in range(2):
        for b_block in range(3):
            base_r = (a_block * 3 + b_block) * 7
            a_col_offset = a_block * 2
            b_row_offset = a_block * 2
            b_col_offset = b_block * 2
            
            for r_s in range(7):
                r = base_r + r_s
                for i in range(2):
                    for j in range(2):
                        full_j = j + a_col_offset
                        U[i * m + full_j, r] = U_s[i * 2 + j, r_s]
                for j in range(2):
                    for k in range(2):
                        full_j = j + b_row_offset
                        full_k = k + b_col_offset
                        V[full_j * p + full_k, r] = V_s[j * 2 + k, r_s]
                for k in range(2):
                    for i in range(2):
                        full_k = k + b_col_offset
                        W[full_k * n + i, r] = W_s[k * 2 + i, r_s]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # 1. Try Gradient Descent Optimization
    # We want to find a rank lower than 42.
    # Let's try optimizing from a rank of 45 (slightly above Strassen) 
    # and see if it collapses to something lower.
    # Or start from 48 (schoolbook) and collapse.
    
    # Strategy: Run optimization with different initial ranks and learning rates.
    configs = [
        {'rank': 45, 'lr': 0.05, 'steps': 3000},
        {'rank': 48, 'lr': 0.02, 'steps': 5000},
        {'rank': 40, 'lr': 0.01, 'steps': 3000}, # Start below 42? Might be hard to converge to valid solution.
    ]
    
    for cfg in configs:
        if time.time() > deadline - 20:
            break
            
        U_opt, V_opt, W_opt, rank_opt = _optimize_decomposition(
            n, m, p, 
            initial_rank=cfg['rank'], 
            learning_rate=cfg['lr'], 
            steps=cfg['steps'],
            time_limit=deadline - time.time() - 10
        )
        
        if rank_opt < best_rank:
            best_rank = rank_opt
            best_U = U_opt
            best_V = V_opt
            best_W = W_opt
            
            # If we found something significantly better, we can stop early
            if best_rank <= 36:
                break

    # 2. Fallback to known good solutions
    # Try Block Strassen (Rank 42)
    U_s, V_s, W_s = _build_block_strassen(n, m, p)
    if 42 < best_rank:
        best_rank = 42
        best_U = U_s
        best_V = V_s
        best_W = W_s
        
    # Try Schoolbook (Rank 48)
    U_t, V_t, W_t = _build_trivial(n, m, p)
    if 48 < best_rank:
        best_rank = 48
        best_U = U_t
        best_V = V_t
        best_W = W_t

    # Final verification of the best candidate
    if best_U is not None:
        if _verify_decomposition(best_U, best_V, best_W, n, m, p, num_tests=20, seed=123):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        else:
            # If verification failed (shouldn't happen if _prune_and_verify worked),
            # fall back to trivial
            U_t, V_t, W_t = _build_trivial(n, m, p)
            return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    else:
        U_t, V_t, W_t = _build_trivial(n, m, p)
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
