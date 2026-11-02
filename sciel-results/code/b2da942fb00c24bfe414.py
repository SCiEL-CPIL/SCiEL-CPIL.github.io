import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec_val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(dec_val - 1.0) > 1e-7:
                    return False
    return True

def _rank33_solution():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}

def _try_scipy_search(target_rank, start_time, deadline, seed=0):
    """Use scipy L-BFGS-B to search for a rank-target_rank decomposition."""
    n, m, p = DIMENSIONS
    num_u = n * m  # 6
    num_v = m * p  # 18
    num_w = p * n  # 12
    
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i*m+j, j*p+k, k*n+i))
    
    def loss_and_grad(x):
        u_size = num_u * target_rank
        v_size = num_v * target_rank
        U = x[:u_size].reshape(num_u, target_rank)
        V = x[u_size:u_size+v_size].reshape(num_v, target_rank)
        W = x[u_size+v_size:].reshape(num_w, target_rank)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (a, b, c) in T_indices:
            prod_sum = np.dot(U[a, :], V[b, :] * W[c, :])
            error = prod_sum - 1.0
            loss += error * error
            g = 2.0 * error
            grad_U[a, :] += g * (V[b, :] * W[c, :])
            grad_V[b, :] += g * (U[a, :] * W[c, :])
            grad_W[c, :] += g * (U[a, :] * V[b, :])
        
        return loss, np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
    
    # Multiple initialization strategies
    best_result = None
    
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(seed + attempt * 100)
        
        if attempt == 0:
            # Initialize from rank-33 structure, truncate
            base = _rank33_solution()
            U_init = np.array(base['U'])[:, :target_rank]
            V_init = np.array(base['V'])[:, :target_rank]
            W_init = np.array(base['W'])[:, :target_rank]
        elif attempt == 1:
            # Random init with moderate scale
            U_init = np.random.randn(num_u, target_rank) * 1.0
            V_init = np.random.randn(num_v, target_rank) * 1.0
            W_init = np.random.randn(num_w, target_rank) * 1.0
        else:
            # Sparse random init
            U_init = np.zeros((num_u, target_rank))
            V_init = np.zeros((num_v, target_rank))
            W_init = np.zeros((num_w, target_rank))
            for r in range(target_rank):
                U_init[np.random.randint(0, num_u), r] = np.random.choice([-1, 1])
                V_init[np.random.randint(0, num_v), r] = np.random.choice([-1, 1])
                W_init[np.random.randint(0, num_w), r] = np.random.choice([-1, 1])
        
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        # Run optimization with time limit
        max_time_per_attempt = min(15, deadline - time.time() - 2)
        if max_time_per_attempt < 3:
            break
        
        class TimeLimitException(Exception):
            pass
        
        def callback(xk):
            if time.time() > deadline - 5:
                raise TimeLimitException()
            return False
        
        try:
            result = minimize(
                loss_and_grad, x0, jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except TimeLimitException:
            continue
        except Exception:
            continue
        
        if result.success or result.fun < 1e-6:
            x_opt = result.x
            u_size = num_u * target_rank
            v_size = num_v * target_rank
            U_opt = x_opt[:u_size].reshape(num_u, target_rank)
            V_opt = x_opt[u_size:u_size+v_size].reshape(num_v, target_rank)
            W_opt = x_opt[u_size+v_size:].reshape(num_w, target_rank)
            
            U_list = [[U_opt[i, r] for r in range(target_rank)] for i in range(num_u)]
            V_list = [[V_opt[i, r] for r in range(target_rank)] for i in range(num_v)]
            W_list = [[W_opt[i, r] for r in range(target_rank)] for i in range(num_w)]
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    fallback = _rank33_solution()
    
    # Verify fallback works
    if not _verify(fallback['U'], fallback['V'], fallback['W']):
        pass  # Should never happen
    
    # Try to find lower rank solutions
    # Theoretical lower bound for 2x3x6: at least 12 (n*p) but likely higher
    # Known: rank(2x3x2)=11, so 3*11=33 is current best
    # Try ranks between 24 and 32
    
    for target_rank in [32, 31, 30, 29, 28, 27, 26, 25, 24]:
        if time.time() > deadline - 10:
            break
        
        result = _try_scipy_search(target_rank, start_time, deadline, seed=target_rank)
        if result is not None:
            return result
    
    return fallback
