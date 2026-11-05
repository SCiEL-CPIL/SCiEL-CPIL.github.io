"""
Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition.
Uses a fundamentally different approach: direct algebraic elimination + structured search
for low-rank decompositions with integer/half-integer entries.
"""
import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n * m, m * p, p * n

    T = np.ones((n, m, p))

    def check_exact(U, V, W, rank):
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def schoolbook_init():
        U = np.zeros((n, m, rank_schoolbook))
        V = np.zeros((m, p, rank_schoolbook))
        W = np.zeros((p, n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U, V, W

    def try_structured_rank(rank, max_iters=500000, seed_base=0):
        """
        Try to find a rank decomposition using structured search:
        1. Start from schoolbook-like initialization
        2. Use alternating minimization / gradient descent
        3. Attempt snapping to integer/half-integer grid
        """
        if time.time() > deadline - 5:
            return None
        
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        # Multiple strategies for initialization
        strategies = []
        
        # Strategy 0: Pure random
        strategies.append(lambda rng: (
            rng.randn(n, m, rank) * 0.5,
            rng.randn(m, p, rank) * 0.5,
            rng.randn(p, n, rank) * 0.5
        ))
        
        # Strategy 1: Sparse random (most entries zero)
        def sparse_random(rng):
            U = rng.randn(n, m, rank)
            V = rng.randn(m, p, rank)
            W = rng.randn(p, n, rank)
            # Zero out ~50% of entries
            for arr in [U, V, W]:
                mask = rng.random(arr.shape) < 0.5
                arr[mask] = 0
            return U, V, W
        strategies.append(sparse_random)
        
        # Strategy 2: Based on schoolbook with random perturbations
        def perturbed_schoolbook(rng):
            U = np.zeros((n, m, rank))
            V = np.zeros((m, p, rank))
            W = np.zeros((p, n, rank))
            # Initialize with schoolbook pattern for first min(rank, n*m*p) entries
            cnt = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if cnt >= rank:
                            break
                        U[i, j, cnt] = 1
                        V[j, k, cnt] = 1
                        W[k, i, cnt] = 1
                        cnt += 1
            # Add random perturbations to remaining columns
            for c in range(cnt, rank):
                U[:, :, c] = rng.randn(n, m) * 0.1
                V[:, :, c] = rng.randn(m, p) * 0.1
                W[:, :, c] = rng.randn(p, n) * 0.1
            return U, V, W
        strategies.append(perturbed_schoolbook)
        
        # Strategy 3: Binary-like initialization
        def binary_init(rng):
            U = rng.choice([0, 0.5, 1.0], size=(n, m, rank))
            V = rng.choice([0, 0.5, 1.0], size=(m, p, rank))
            W = rng.choice([0, 0.5, 1.0], size=(p, n, rank))
            return U, V, W
        strategies.append(binary_init)
        
        # Strategy 4: Kronecker-like structure
        def kronecker_init(rng):
            U = np.zeros((n, m, rank))
            V = np.zeros((m, p, rank))
            W = np.zeros((p, n, rank))
            # Try to create structured blocks
            block_size = max(1, rank // (n * m))
            for i in range(n):
                for j in range(m):
                    start = i * m * block_size + j * block_size
                    end = min(start + block_size, rank)
                    if start < rank:
                        U[i, j, start:end] = 1.0
            block_size_v = max(1, rank // (m * p))
            for j in range(m):
                for k in range(p):
                    start = j * p * block_size_v + k * block_size_v
                    end = min(start + block_size_v, rank)
                    if start < rank:
                        V[j, k, start:end] = 1.0
            block_size_w = max(1, rank // (p * n))
            for k in range(p):
                for i in range(n):
                    start = k * n * block_size_w + i * block_size_w
                    end = min(start + block_size_w, rank)
                    if start < rank:
                        W[k, i, start:end] = 1.0
            return U, V, W
        strategies.append(kronecker_init)
        
        best_err = np.inf
        best_solution = None
        
        num_strategies = len(strategies)
        per_strategy_iters = max(1, max_iters // num_strategies)
        
        for strat_idx, init_func in enumerate(strategies):
            if time.time() > deadline - 10:
                break
                
            for attempt in range(per_strategy_iters):
                if time.time() > deadline - 10:
                    break
                    
                seed = seed_base + strat_idx * 1000 + attempt
                rng = np.random.RandomState(seed)
                
                U, V, W = init_func(rng)
                
                # Alternating minimization: fix two, optimize one
                # For each rank component c, we can solve linear systems
                for iter_count in range(200):
                    if time.time() > deadline - 10:
                        break
                    
                    # Check if already exact
                    T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    err = np.sum((T_rec - T) ** 2)
                    if err < 1e-10:
                        return (U, V, W)
                    
                    if err < best_err:
                        best_err = err
                        best_solution = (U.copy(), V.copy(), W.copy())
                    
                    # Alternate: optimize U given V, W
                    # For each (i,j), we want sum_c U[i,j,c] * V[j,?,c] * W[?,i,c] = 1
                    # This is complex; use gradient descent step instead
                    lr = 0.01 / (1 + iter_count * 0.01)
                    
                    # Gradient computation
                    T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    err_tensor = T_rec - T
                    
                    grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err_tensor, V, W)
                    grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err_tensor, U, W)
                    grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err_tensor, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    # Periodic snapping attempt
                    if iter_count % 50 == 0 and iter_count > 0:
                        U_s = np.round(U * 2) / 2
                        V_s = np.round(V * 2) / 2
                        W_s = np.round(W * 2) / 2
                        T_rec_s = np.einsum('ijc,jkc,kic->ijk', U_s, V_s, W_s)
                        err_s = np.sum((T_rec_s - T) ** 2)
                        if err_s < 1e-9:
                            return (U_s, V_s, W_s)
                        if err_s < err * 0.5:
                            U, V, W = U_s, V_s, W_s
        
        # Final snapping on best solution
        if best_solution is not None and best_err < 1e-3:
            U, V, W = best_solution
            for scale in [2, 4, 8, 16]:
                U_s = np.round(U * scale) / scale
                V_s = np.round(V * scale) / scale
                W_s = np.round(W * scale) / scale
                T_rec_s = np.einsum('ijc,jkc,kic->ijk', U_s, V_s, W_s)
                if np.allclose(T_rec_s, T, atol=1e-9):
                    return (U_s, V_s, W_s)
        
        return None

    def try_l_bfgs_rank(rank, num_restarts=15, time_budget=8.0):
        """
        Use L-BFGS-B optimization with analytic gradients.
        """
        if time.time() > deadline - 10:
            return None
        
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        def objective_and_grad(x):
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U + size_V]
            W_flat = x[size_U + size_V:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            loss = np.sum(err ** 2)
            
            grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
            grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
            grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad
        
        best_result = None
        best_err = np.inf
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            restart_deadline = time.time() + time_budget
            
            # Diverse initialization
            np.random.seed(r * 7919 + 104729)
            
            if r % 4 == 0:
                U_init = np.random.randn(n, m, rank) * 0.5
                V_init = np.random.randn(m, p, rank) * 0.5
                W_init = np.random.randn(p, n, rank) * 0.5
            elif r % 4 == 1:
                U_init = np.random.choice([0, 0.5, 1.0], size=(n, m, rank))
                V_init = np.random.choice([0, 0.5, 1.0], size=(m, p, rank))
                W_init = np.random.choice([0, 0.5, 1.0], size=(p, n, rank))
            elif r % 4 == 2:
                # Sparse
                U_init = np.random.randn(n, m, rank)
                V_init = np.random.randn(m, p, rank)
                W_init = np.random.randn(p, n, rank)
                for arr in [U_init, V_init, W_init]:
                    mask = np.random.random(arr.shape) < 0.6
                    arr[mask] = 0
                arr_scale = 1.0
            else:
                U_init = np.random.randn(n, m, rank) * 0.2
                V_init = np.random.randn(m, p, rank) * 0.2
                W_init = np.random.randn(p, n, rank) * 0.2
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = opt.minimize(objective_and_grad, x0, method='L-BFGS-B',
                                   jac=True,
                                   options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
                
                if time.time() > restart_deadline:
                    continue
                    
                err = res.fun
                if err < best_err:
                    best_err = err
                    U_sol = res.x[:size_U].reshape(n, m, rank)
                    V_sol = res.x[size_U:size_U + size_V].reshape(m, p, rank)
                    W_sol = res.x[size_U + size_V:].reshape(p, n, rank)
                    best_result = (U_sol, V_sol, W_sol)
                
                if err < 1e-8:
                    # Try snapping
                    for scale in [2, 4, 8]:
                        U_s = np.round(U_sol * scale) / scale
                        V_s = np.round(V_sol * scale) / scale
                        W_s = np.round(W_sol * scale) / scale
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_s, V_s, W_s)
                        if np.allclose(T_rec, T, atol=1e-9):
                            return (U_s, V_s, W_s)
                        
            except Exception:
                continue
        
        # Final snapping on best
        if best_result is not None and best_err < 1e-4:
            U_sol, V_sol, W_sol = best_result
            for scale in [2, 4, 8, 16, 32]:
                U_s = np.round(U_sol * scale) / scale
                V_s = np.round(V_sol * scale) / scale
                W_s = np.round(W_sol * scale) / scale
                T_rec = np.einsum('ijc,jkc,kic->ijk', U_s, V_s, W_s)
                if np.allclose(T_rec, T, atol=1e-9):
                    return (U_s, V_s, W_s)
        
        return None

    def try_hybrid_rank(rank):
        """
        Combine structured search and L-BFGS-B for a given rank.
        """
        if time.time() > deadline - 15:
            return None
        
        # First try structured approach (faster per iteration)
        result = try_structured_rank(rank, max_iters=100000, seed_base=0)
        if result is not None:
            return result
        
        if time.time() > deadline - 15:
            return None
        
        # Then try L-BFGS-B
        result = try_l_bfgs_rank(rank, num_restarts=10, time_budget=6.0)
        if result is not None:
            return result
        
        return None

    # Strategy:
    # 1. Schoolbook as fallback
    # 2. Try rank 71, 70, 69 in order
    # 3. Return best found
    
    U_sb, V_sb, W_sb = schoolbook_init()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_schoolbook
    
    # Try decreasing ranks
    for target_rank in [71, 70, 69, 68]:
        if time.time() > deadline - 20:
            break
        
        if target_rank >= best_rank:
            continue
            
        result = try_hybrid_rank(target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a lower rank, stop
    
    U, V, W = best_result
    
    # Convert to required format: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n+i, r] convention means W is indexed by [k*n+i, r]
    U_out = U.reshape(NM, best_rank).tolist()
    V_out = V.reshape(MP, best_rank).tolist()
    W_out = W.reshape(PN, best_rank).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
