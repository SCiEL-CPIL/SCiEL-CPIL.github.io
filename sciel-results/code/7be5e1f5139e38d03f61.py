"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _get_target_tensor(n, m, p):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_iter_fast(U, V, W, T, rank):
    """Fast ALS iteration for 9x9x9 tensor."""
    # Update U
    for r in range(rank):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - T_approx + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        b = np.einsum('abc,b,c->a', R, V[:,r], W[:,r])
        A = np.dot(V[:,r]**2, W[:,r]**2)
        if A > 1e-12:
            U[:,r] = b / A
    
    # Update V
    for r in range(rank):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - T_approx + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        b = np.einsum('abc,a,c->b', R, U[:,r], W[:,r])
        A = np.dot(U[:,r]**2, W[:,r]**2)
        if A > 1e-12:
            V[:,r] = b / A
    
    # Update W
    for r in range(rank):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - T_approx + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        b = np.einsum('abc,a,b->c', R, U[:,r], V[:,r])
        A = np.dot(U[:,r]**2, V[:,r]**2)
        if A > 1e-12:
            W[:,r] = b / A
    
    return U, V, W

def _find_rank23(n, m, p, deadline, max_restarts=400):
    """Find rank-23 decomposition."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = _get_target_tensor(n, m, p)
    target_rank = 23
    
    best_error = float('inf')
    best_solution = None
    
    U0, V0, W0 = _schoolbook(n, m, p)
    rng = random.Random(7)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        mode = restart % 8
        
        if mode == 0:
            drops = rng.sample(range(27), 4)
            keep = [i for i in range(27) if i not in drops]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            U += np.random.randn(dim_a, target_rank) * 0.001
            V += np.random.randn(dim_b, target_rank) * 0.001
            W += np.random.randn(dim_c, target_rank) * 0.001
        elif mode == 1:
            drops = rng.sample(range(27), 4)
            keep = [i for i in range(27) if i not in drops]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            U += np.random.randn(dim_a, target_rank) * 0.01
            V += np.random.randn(dim_b, target_rank) * 0.01
            W += np.random.randn(dim_c, target_rank) * 0.01
        elif mode == 2:
            drops = rng.sample(range(27), 4)
            keep = [i for i in range(27) if i not in drops]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            U += np.random.randn(dim_a, target_rank) * 0.05
            V += np.random.randn(dim_b, target_rank) * 0.05
            W += np.random.randn(dim_c, target_rank) * 0.05
        elif mode == 3:
            U = np.random.randn(dim_a, target_rank) * 0.3
            V = np.random.randn(dim_b, target_rank) * 0.3
            W = np.random.randn(dim_c, target_rank) * 0.3
        elif mode == 4:
            drops = rng.sample(range(27), 4)
            keep = [i for i in range(27) if i not in drops]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            for _ in range(50):
                c = rng.randint(0, target_rank-1)
                U[rng.randint(0, dim_a-1), c] += np.random.randn() * 0.1
                V[rng.randint(0, dim_b-1), c] += np.random.randn() * 0.1
                W[rng.randint(0, dim_c-1), c] += np.random.randn() * 0.1
        elif mode == 5:
            drops = rng.sample(range(27), 4)
            keep = [i for i in range(27) if i not in drops]
            alpha = rng.uniform(0.4, 0.6)
            U = U0[:, keep].copy() * alpha + np.random.randn(dim_a, target_rank) * (1-alpha) * 0.4
            V = V0[:, keep].copy() * alpha + np.random.randn(dim_b, target_rank) * (1-alpha) * 0.4
            W = W0[:, keep].copy() * alpha + np.random.randn(dim_c, target_rank) * (1-alpha) * 0.4
        elif mode == 6:
            # Scale perturbation
            drops = rng.sample(range(27), 4)
            keep = [i for i in range(27) if i not in drops]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            for c in range(target_rank):
                s = rng.uniform(0.5, 1.5)
                U[:, c] *= s
                V[:, c] /= s
        else:
            # Large random perturbation
            drops = rng.sample(range(27), 4)
            keep = [i for i in range(27) if i not in drops]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            U += np.random.randn(dim_a, target_rank) * 0.2
            V += np.random.randn(dim_b, target_rank) * 0.2
            W += np.random.randn(dim_c, target_rank) * 0.2
        
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        
        max_iters = 350
        prev_err = float('inf')
        plateau = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U_curr, V_curr, W_curr = _als_iter_fast(U_curr, V_curr, W_curr, T, target_rank)
            
            if it % 4 == 0:
                T_approx = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                err = np.sum((T_approx - T)**2)
                
                if err < best_error:
                    best_error = err
                    best_solution = (U_curr.copy(), V_curr.copy(), W_curr.copy())
                
                if err < 1e-10:
                    return U_curr, V_curr, W_curr
                
                if err > prev_err * 0.99:
                    plateau += 1
                else:
                    plateau = 0
                prev_err = err
                
                if plateau > 80:
                    break
        
        if best_error < 1e-8:
            return best_solution
    
    if best_error < 1e-6 and best_solution is not None:
        return best_solution
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Try rank 23
    if time.time() < deadline:
        result = _find_rank23(n, m, p, deadline, max_restarts=500)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

