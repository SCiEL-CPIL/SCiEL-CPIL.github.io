import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # Note: The task description says "output factor follows the official trace convention W[k*n+i,r]"
            # In the tensor T[u,v,w], u corresponds to (i,j), v to (j,k), w to (k,i)?
            # Standard MatMul tensor: T[i,j,k] = 1 if output (i,k) receives input (i,j) and (j,k).
            # The interface defines U (n*m, rank), V (m*p, rank), W (p*n, rank).
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            # This matches the standard definition where the output index is (i, k) mapped to k*n + i.
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T_rec = np.zeros(T.shape)
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_rec, T, atol=tol)

def _calculate_error(U, V, W, T):
    """Calculate the squared error between reconstructed tensor and target."""
    T_rec = np.zeros(T.shape)
    for r in range(U.shape[1]):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.sum((T_rec - T) ** 2)

def _local_search_integer(T, n, m, p, rank, time_limit=10, seed=0):
    """
    Perform gradient-based local search on integer lattice to find exact decomposition.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize with small random integers
    U = np.random.randint(-2, 3, size=(n * m, rank))
    V = np.random.randint(-2, 3, size=(m * p, rank))
    W = np.random.randint(-2, 3, size=(p * n, rank))
    
    def compute_error(U, V, W):
        T_rec = np.zeros(T.shape)
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.sum((T_rec - T) ** 2)

    current_error = compute_error(U, V, W)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 20
    
    lr = 0.5
    
    while time.time() - start_time < time_limit:
        # Compute Residual
        T_rec = np.zeros(T.shape)
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        R = T - T_rec
        current_error = np.sum(R**2)
        
        if current_error < 1e-9:
            return U, V, W, 0.0
            
        if current_error < best_error:
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                if current_error > 10:
                    U = np.random.randint(-2, 3, size=(n * m, rank))
                    V = np.random.randint(-2, 3, size=(m * p, rank))
                    W = np.random.randint(-2, 3, size=(p * n, rank))
                    no_improve_count = 0
                    continue
                else:
                    break

        # Compute Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            dU[:, r] = -2.0 * np.einsum('uvw,v,w->u', R, V[:, r], W[:, r])
            dV[:, r] = -2.0 * np.einsum('uvw,u,w->v', R, U[:, r], W[:, r])
            dW[:, r] = -2.0 * np.einsum('uvw,u,v->w', R, U[:, r], V[:, r])
            
        # Update with Rounding
        U_new = np.round(U + lr * dU)
        V_new = np.round(V + lr * dV)
        W_new = np.round(W + lr * dW)
        
        new_error = compute_error(U_new, V_new, W_new)
        
        if new_error < current_error:
            U, V, W = U_new, V_new, W_new
            lr *= 0.9
        else:
            lr *= 0.5
            if lr < 0.01:
                # Try random perturbation
                idx = np.random.randint(n * m)
                r_idx = np.random.randint(rank)
                U[idx, r_idx] += np.random.choice([-1, 1])
                pass

    return best_U, best_V, best_W, best_error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    T = _build_target_tensor(n, m, p)

    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    best_result, best_rank = schoolbook()

    # Known optimal rank for 3x3x5 is 38.
    # We try to find it.
    target_ranks = [38, 39, 40, 41, 42, 43, 44]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple restarts
        n_restarts = 5
        # Allocate time per rank
        time_per_rank = 20
        time_per_restart = time_per_rank / n_restarts
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                U, V, W, err = _local_search_integer(T, n, m, p, target_rank, 
                                                     time_limit=time_per_restart, 
                                                     seed=restart * 100 + target_rank)
                if err < 1e-6:
                    # Verify
                    if _verify_decomposition(U, V, W, T, tol=1e-6):
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        best_rank = target_rank
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            # If we found 38, we can stop as it's the known minimum
            if target_rank == 38:
                break
            # If we found higher, we still try lower ones? 
            # The loop goes from 38 upwards, so if 38 fails, we try 39 etc.
            # But if 38 succeeds, we break.
            # If 38 fails, we continue to 39.
            # This is correct.

    return best_result
