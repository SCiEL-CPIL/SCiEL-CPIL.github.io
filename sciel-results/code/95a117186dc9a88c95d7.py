import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = nm * mp * pn // m # No, rank is n*m*p
    schoolbook_rank = n * m * p
    
    # Schoolbook solution as fallback
    def _schoolbook():
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

    def _validate(U, V, W):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check dimensions
        if U_np.shape != (n * m, rank):
            return False
        if V_np.shape != (m * p, rank):
            return False
        if W_np.shape != (p * n, rank):
            return False
        
        # Check exactness with tolerance
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Try to find a better decomposition
    # Known lower bound: max(n*m, m*p, p*n) = 28
    # Upper bound: 56
    
    # We will try ranks from 55 downwards, but only for a short time each.
    # If we can't find one quickly, we return schoolbook.
    
    # Strategy: Random restarts with gradient descent on the error surface.
    # Error E = sum (sum_c U_ic V_jc W_kc - 1)^2
    
    def _try_rank(r, time_budget=5.0):
        np.random.seed(int(time.time() * 1000) % 10000)
        
        # Initialize with small random integers
        U = np.random.randint(-1, 2, size=(nm, r))
        V = np.random.randint(-1, 2, size=(mp, r))
        W = np.random.randint(-1, 2, size=(pn, r))
        
        # Convert to float for optimization
        U_f = U.astype(float)
        V_f = V.astype(float)
        W_f = W.astype(float)
        
        # Precompute indices for speed
        idx_U = []
        idx_V = []
        idx_W = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_U.append(i * m + j)
                    idx_V.append(j * p + k)
                    idx_W.append(k * n + i)
        
        idx_U = np.array(idx_U)
        idx_V = np.array(idx_V)
        idx_W = np.array(idx_W)
        
        # Local gradient descent
        lr = 0.01
        best_error = float('inf')
        
        start = time.time()
        while time.time() - start < time_budget:
            # Compute error and gradients
            # E = sum_t (sum_c U[ut,c] V[vt,c] W[wt,c] - 1)^2
            # dE/dU[ut,c] = 2 * sum_t (val_t - 1) * V[vt,c] * W[wt,c]
            
            vals = np.zeros(n * m * p)
            for t in range(n * m * p):
                u_idx = idx_U[t]
                v_idx = idx_V[t]
                w_idx = idx_W[t]
                vals[t] = np.dot(U_f[u_idx], V_f[v_idx] * W_f[w_idx])
            
            error = np.sum((vals - 1.0)**2)
            
            if error < best_error:
                best_error = error
                # Snap to integers and check
                U_snap = np.round(U_f).astype(int)
                V_snap = np.round(V_f).astype(int)
                W_snap = np.round(W_f).astype(int)
                
                # Check if snapped version is valid
                if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            if error < 1e-12:
                # Found a real solution, try to snap
                U_snap = np.round(U_f).astype(int)
                V_snap = np.round(V_f).astype(int)
                W_snap = np.round(W_f).astype(int)
                if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                # Try half-integers
                U_snap = np.round(U_f * 2) / 2
                V_snap = np.round(V_f * 2) / 2
                W_snap = np.round(W_f * 2) / 2
                if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
            # Compute gradients
            # grad_U[ut, c] += 2 * (vals[t] - 1) * V[vt, c] * W[wt, c]
            grad_U = np.zeros_like(U_f)
            grad_V = np.zeros_like(V_f)
            grad_W = np.zeros_like(W_f)
            
            for t in range(n * m * p):
                diff = 2 * (vals[t] - 1.0)
                u_idx = idx_U[t]
                v_idx = idx_V[t]
                w_idx = idx_W[t]
                
                # Vectorized update for this t
                prod_VW = V_f[v_idx] * W_f[w_idx]
                prod_UW = U_f[u_idx] * W_f[w_idx]
                prod_UV = U_f[u_idx] * V_f[v_idx]
                
                grad_U[u_idx] += diff * prod_VW
                grad_V[v_idx] += diff * prod_UW
                grad_W[w_idx] += diff * prod_UV
            
            # Update
            U_f -= lr * grad_U
            V_f -= lr * grad_V
            W_f -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.99
            
            # Periodic re-initialization if stuck
            if error > best_error * 1.5 and time.time() - start > 1.0:
                break
        
        return None

    # Try ranks 55, 54, ... down to 30
    # But only if we have time
    for r in range(55, 27, -1):
        if time.time() > deadline:
            break
        
        # Give each rank a small budget
        budget = 2.0
        result = _try_rank(r, budget)
        if result:
            return result
    
    return _schoolbook()
