import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Keep safety margin from 150s limit
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    trivial_rank = n * m * p  # 24
    
    # Trivial solution for 2x3x4
    U_trivial = np.zeros((nm, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((mp, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((pn, trivial_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()
    
    def check_valid(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    def compute_error(U, V, W, rank):
        """Compute the error for given U, V, W matrices"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    err += (s - 1.0)**2
        return err

    def error_func_factory(target_rank):
        """Create error function for given target_rank"""
        def error_func(x):
            U = x[:nm*target_rank].reshape(nm, target_rank)
            V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            return compute_error(U, V, W, target_rank)
        return error_func

    def try_optimization(target_rank, U_init, V_init, W_init, max_iter=1000, maxfun=10000):
        """Try to optimize from given initial values"""
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        error_func = error_func_factory(target_rank)
        
        try:
            res = minimize(error_func, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'maxfun': maxfun, 'ftol': 1e-12})
            
            if res.fun < 1e-8:
                U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                # Try snapping to half-integers
                U_s, V_s, W_s = snap_to_half_integer(U_sol, V_sol, W_sol)
                if check_valid(U_s, V_s, W_s, target_rank, tol=1e-9):
                    return U_s, V_s, W_s, True
                
                # If snapping fails, try to polish with exact solution
                # Use higher precision optimization
                x_refined = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
                try:
                    res2 = minimize(error_func, x_refined, method='L-BFGS-B',
                                   options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-15})
                    if res2.fun < 1e-12:
                        U_sol2 = res2.x[:nm*target_rank].reshape(nm, target_rank)
                        V_sol2 = res2.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                        W_sol2 = res2.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                        
                        U_s2, V_s2, W_s2 = snap_to_half_integer(U_sol2, V_sol2, W_sol2)
                        if check_valid(U_s2, V_s2, W_s2, target_rank, tol=1e-9):
                            return U_s2, V_s2, W_s2, True
                except Exception:
                    pass
                    
        except Exception:
            pass
            
        return None, None, None, False

    def generate_structured_init(target_rank, seed):
        """Generate structured initial values that are more likely to converge"""
        np.random.seed(seed)
        
        # Start with a partially structured initialization
        U_init = np.zeros((nm, target_rank))
        V_init = np.zeros((mp, target_rank))
        W_init = np.zeros((pn, target_rank))
        
        # First, place some of the trivial solution components
        # This helps the optimizer start closer to a valid solution
        col = 0
        for i in range(n):
            for j in range(m):
                if col < target_rank:
                    U_init[i * m + j, col] = 1.0
                    col += 1
        
        # Add random components for the remaining rank
        for r in range(col, target_rank):
            U_init[:, r] = np.random.randn(nm) * 0.1
            V_init[:, r] = np.random.randn(mp) * 0.1
            W_init[:, r] = np.random.randn(pn) * 0.1
        
        # For the first 'col' components, add small random perturbations
        # to help escape from the trivial solution structure
        for r in range(min(col, target_rank)):
            U_init[:, r] += np.random.randn(nm) * 0.05
            V_init[:, r] += np.random.randn(mp) * 0.05
            W_init[:, r] += np.random.randn(pn) * 0.05
            
        return U_init, V_init, W_init

    def generate_random_init(target_rank, seed):
        """Generate random initial values"""
        np.random.seed(seed)
        U_init = np.random.randn(nm, target_rank) * 0.3
        V_init = np.random.randn(mp, target_rank) * 0.3
        W_init = np.random.randn(pn, target_rank) * 0.3
        return U_init, V_init, W_init

    # Try to find solutions for ranks from 16 up to 23
    # Focus on lower ranks first as they are harder but more valuable
    for target_rank in range(16, 24):
        if time.time() > deadline - 20.0:
            break
        
        # Try multiple initializations with different strategies
        found = False
        
        # Strategy 1: Structured initialization (partially use trivial solution)
        for seed in range(5):
            if time.time() > deadline - 15.0:
                break
                
            U_init, V_init, W_init = generate_structured_init(target_rank, seed + 100)
            U_sol, V_sol, W_sol, success = try_optimization(target_rank, U_init, V_init, W_init)
            
            if success:
                best_rank = target_rank
                best_U = U_sol
                best_V = V_sol
                best_W = W_sol
                found = True
                break
        
        if found:
            break
        
        # Strategy 2: Pure random initialization
        for seed in range(5):
            if time.time() > deadline - 15.0:
                break
                
            U_init, V_init, W_init = generate_random_init(target_rank, seed + 200)
            U_sol, V_sol, W_sol, success = try_optimization(target_rank, U_init, V_init, W_init)
            
            if success:
                best_rank = target_rank
                best_U = U_sol
                best_V = V_sol
                best_W = W_sol
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
