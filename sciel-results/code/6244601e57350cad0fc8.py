import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    def check_valid(U, V, W, rank, tol=1e-9):
        # Verify the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    def snap_to_integer(U, V, W):
        U_s = np.round(U)
        V_s = np.round(V)
        W_s = np.round(W)
        return U_s, V_s, W_s

    # Strategy:
    # 1. Try to find a rank-19 decomposition using random search + local optimization.
    #    The known lower bound for M(2,3,4) is 19.
    #    We can exploit symmetry: M(2,3,4) can be viewed as two M(2,3,2) blocks sharing the U factor?
    #    No, U is shared, V and W are different.
    #    Actually, we can split B into B1 (3x2) and B2 (3x2).
    #    C1 = A B1, C2 = A B2.
    #    If we have a rank-R decomposition for M(2,3,2), we can use it for both blocks.
    #    U is the same for both.
    #    V1 and V2 are the V factors for the respective blocks.
    #    W1 and W2 are the W factors.
    #    Total rank = R + R = 2R.
    #    Best known for M(2,3,2) is 9 (Larsen 1990? or similar).
    #    If we find rank 9 for M(2,3,2), total rank is 18.
    #    Wait, is M(2,3,2) rank 9?
    #    M(2,2,2) is 7.
    #    M(2,3,2) >= M(2,2,2) = 7.
    #    Trivial is 12.
    #    If we can find rank 9 for M(2,3,2), then M(2,3,4) can be done in rank 18.
    #    But maybe we can do better by mixing?
    #    Let's first try to find a good decomposition for M(2,3,2).
    
    # Search for M(2,3,2) decomposition
    n_s, m_s, p_s = 2, 3, 2
    nm_s = n_s * m_s  # 6
    mp_s = m_s * p_s  # 6
    pn_s = p_s * n_s  # 4
    
    found_m232 = False
    best_m232_rank = 12
    best_m232_U = None
    best_m232_V = None
    best_m232_W = None
    
    for target_rank in [9, 10, 11]:
        if time.time() > deadline - 20.0:
            break
            
        # Multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 20.0:
                break
                
            np.random.seed(42 + restart)
            U_init = np.random.rand(nm_s, target_rank) * 0.5 - 0.25
            V_init = np.random.rand(mp_s, target_rank) * 0.5 - 0.25
            W_init = np.random.rand(pn_s, target_rank) * 0.5 - 0.25
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def error_func(x):
                U = x[:nm_s*target_rank].reshape(nm_s, target_rank)
                V = x[nm_s*target_rank:nm_s*target_rank+mp_s*target_rank].reshape(mp_s, target_rank)
                W = x[nm_s*target_rank+mp_s*target_rank:].reshape(pn_s, target_rank)
                
                err = 0.0
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m_s + j, r] * V[j*p_s + k, r] * W[k*n_s + i, r]
                            err += (s - 1.0)**2
                return err
                
            try:
                from scipy.optimize import minimize
                res = minimize(error_func, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 2000})
                if res.fun < 1e-8:
                    U_sol = res.x[:nm_s*target_rank].reshape(nm_s, target_rank)
                    V_sol = res.x[nm_s*target_rank:nm_s*target_rank+mp_s*target_rank].reshape(mp_s, target_rank)
                    W_sol = res.x[nm_s*target_rank+mp_s*target_rank:].reshape(pn_s, target_rank)
                    
                    # Try snapping to half-integers
                    U_s, V_s, W_s = snap_to_half_integer(U_sol, V_sol, W_sol)
                    
                    # Check validity
                    valid = True
                    for i in range(n_s):
                        for j in range(m_s):
                            for k in range(p_s):
                                s = 0.0
                                for r in range(target_rank):
                                    s += U_s[i*m_s + j, r] * V_s[j*p_s + k, r] * W_s[k*n_s + i, r]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid:
                        continue
                        
                    # Also try snapping to integers
                    U_i, V_i, W_i = snap_to_integer(U_sol, V_sol, W_sol)
                    valid_i = True
                    for i in range(n_s):
                        for j in range(m_s):
                            for k in range(p_s):
                                s = 0.0
                                for r in range(target_rank):
                                    s += U_i[i*m_s + j, r] * V_i[j*p_s + k, r] * W_i[k*n_s + i, r]
                                if abs(s - 1.0) > 1e-9:
                                    valid_i = False
                                    break
                            if not valid_i: break
                        if not valid_i: break
                    if valid_i:
                        U_s, V_s, W_s = U_i, V_i, W_i
                        # Prefer integer solutions if valid, but half-integer is fine too.
                        # The problem allows integer/half-integer.
                        
                    best_m232_rank = target_rank
                    best_m232_U = U_s
                    best_m232_V = V_s
                    best_m232_W = W_s
                    found_m232 = True
                    break
            except Exception:
                pass
        
        if found_m232:
            break

    if found_m232:
        # Construct 2x3x4 solution from two M(2,3,2) blocks
        rank_total = 2 * best_m232_rank
        U_total = np.zeros((nm, rank_total), dtype=np.float64)
        V_total = np.zeros((mp, rank_total), dtype=np.float64)
        W_total = np.zeros((pn, rank_total), dtype=np.float64)
        
        # Block 1: k=0,1
        U_total[:, 0:best_m232_rank] = best_m232_U
        # V indices for k=0,1 are j*4 + 0 and j*4 + 1
        # best_m232_V has indices j*2 + k for k=0,1
        for j in range(m):
            V_total[j*4 + 0, 0:best_m232_rank] = best_m232_V[j*2 + 0, :]
            V_total[j*4 + 1, 0:best_m232_rank] = best_m232_V[j*2 + 1, :]
        # W indices for k=0,1 are k*2 + i
        W_total[0:4, 0:best_m232_rank] = best_m232_W
        
        # Block 2: k=2,3
        U_total[:, best_m232_rank:rank_total] = best_m232_U
        for j in range(m):
            V_total[j*4 + 2, best_m232_rank:rank_total] = best_m232_V[j*2 + 0, :]
            V_total[j*4 + 3, best_m232_rank:rank_total] = best_m232_V[j*2 + 1, :]
        W_total[4:8, best_m232_rank:rank_total] = best_m232_W
        
        if check_valid(U_total, V_total, W_total, rank_total):
            if rank_total < best_rank:
                best_rank = rank_total
                best_U = U_total
                best_V = V_total
                best_W = W_total

    # If we didn't find a better M(2,3,2), or if the combined rank is not better than trivial,
    # try direct search for M(2,3,4) with rank 20, 21, 22, 23
    # This is computationally expensive, so we only do it if we have time.
    
    if best_rank == 24 and time.time() < deadline - 30.0:
        for target_rank in [20, 21, 22, 23]:
            if time.time() > deadline - 10.0:
                break
                
            # Use a smarter initialization
            # Start from trivial and zero out some columns?
            # Or random.
            np.random.seed(123)
            U_init = np.random.rand(nm, target_rank) * 0.5 - 0.25
            V_init = np.random.rand(mp, target_rank) * 0.5 - 0.25
            W_init = np.random.rand(pn, target_rank) * 0.5 - 0.25
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def error_func_direct(x):
                U = x[:nm*target_rank].reshape(nm, target_rank)
                V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                            err += (s - 1.0)**2
                return err
                
            try:
                from scipy.optimize import minimize
                res = minimize(error_func_direct, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
                if res.fun < 1e-8:
                    U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                    V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    U_s, V_s, W_s = snap_to_half_integer(U_sol, V_sol, W_sol)
                    if check_valid(U_s, V_s, W_s, target_rank):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = U_s
                            best_V = V_s
                            best_W = W_s
            except Exception:
                pass

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
