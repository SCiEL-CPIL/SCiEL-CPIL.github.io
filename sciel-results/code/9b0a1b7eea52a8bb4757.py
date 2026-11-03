import time
import numpy as np
import random

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_decomposition(U, V, W, rank):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m + j] * V_np[j*p + k] * W_np[k*n + i])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def get_residuals(U, V, W, rank):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m + j] * V_np[j*p + k] * W_np[k*n + i])
                    total_err += (val - 1.0) ** 2
        return total_err

    def snap_to_half_int(x):
        return np.round(x * 2) / 2

    def refine_decomposition(U, V, W, rank, time_limit):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        U = snap_to_half_int(U)
        V = snap_to_half_int(V)
        W = snap_to_half_int(W)
        
        current_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
        
        if current_err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        while time.time() < time_limit:
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                idx = random.randint(0, nm - 1)
                r = random.randint(0, rank - 1)
                old_val = U[idx, r]
                neighbors = [old_val - 0.5, old_val + 0.5]
                best_new_val = old_val
                best_new_err = current_err
                for nv in neighbors:
                    if nv < -2 or nv > 2: continue
                    U[idx, r] = nv
                    new_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_new_val = nv
                    U[idx, r] = old_val
                
                if best_new_val != old_val:
                    U[idx, r] = best_new_val
                    current_err = best_new_err
            elif mat_idx == 1:
                idx = random.randint(0, mp - 1)
                r = random.randint(0, rank - 1)
                old_val = V[idx, r]
                neighbors = [old_val - 0.5, old_val + 0.5]
                best_new_val = old_val
                best_new_err = current_err
                for nv in neighbors:
                    if nv < -2 or nv > 2: continue
                    V[idx, r] = nv
                    new_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_new_val = nv
                    V[idx, r] = old_val
                
                if best_new_val != old_val:
                    V[idx, r] = best_new_val
                    current_err = best_new_err
            else:
                idx = random.randint(0, pn - 1)
                r = random.randint(0, rank - 1)
                old_val = W[idx, r]
                neighbors = [old_val - 0.5, old_val + 0.5]
                best_new_val = old_val
                best_new_err = current_err
                for nv in neighbors:
                    if nv < -2 or nv > 2: continue
                    W[idx, r] = nv
                    new_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_new_val = nv
                    W[idx, r] = old_val
                
                if best_new_val != old_val:
                    W[idx, r] = best_new_val
                    current_err = best_new_err
            
            if current_err < 1e-9:
                break
                
        if current_err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def svd_compress(U, V, W, target_rank):
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        r = target_rank
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        for _ in range(50):
            M_U = np.zeros((nm, r))
            for i in range(n):
                for j in range(m):
                    for r_idx in range(r):
                        s = 0
                        for k in range(p):
                            s += V[j*p + k, r_idx] * W[k*n + i, r_idx]
                        M_U[i*m + j, r_idx] = s
            
            M_V = np.zeros((mp, r))
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        s = 0
                        for i in range(n):
                            s += U[i*m + j, r_idx] * W[k*n + i, r_idx]
                        M_V[j*p + k, r_idx] = s
            
            M_W = np.zeros((pn, r))
            for k in range(p):
                for i in range(n):
                    for r_idx in range(r):
                        s = 0
                        for j in range(m):
                            s += U[i*m + j, r_idx] * V[j*p + k, r_idx]
                        M_W[k*n + i, r_idx] = s
            
            # Solve for U, V, W
            # U @ M_U^T = T_unfolded? No, T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
            # This is not a standard linear system because V and W are coupled.
            # We can use a simple gradient step or just keep it simple.
            pass
            
        return None # Placeholder

    best_result = schoolbook()
    best_rank = n * m * p
    
    target_ranks = [32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    for R in target_ranks:
        if time.time() > deadline - 20.0:
            break
            
        success = False
        for attempt in range(5):
            if time.time() > deadline - 20.0:
                break
                
            U = np.random.choice([-1, 0, 1], size=(nm, R))
            V = np.random.choice([-1, 0, 1], size=(mp, R))
            W = np.random.choice([-1, 0, 1], size=(pn, R))
            
            time_limit = time.time() + 5.0
            result = refine_decomposition(U.tolist(), V.tolist(), W.tolist(), R, time_limit)
            if result is not None:
                best_result = result
                best_rank = R
                success = True
                break
                
        if success:
            break
            
    return best_result
