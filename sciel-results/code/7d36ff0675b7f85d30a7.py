import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 40s margin for safety
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify(U, V, W):
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
            if np.any(np.abs(mat) > 100):
                return False

        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        try:
            T_calc = np.einsum('ijr,jkr,kr i->ijk', U_reshaped, V_reshaped, W_reshaped)
            if not np.allclose(T_calc, 1.0, atol=1e-6):
                return False
        except Exception:
            return False
            
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def loss(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        try:
            T_calc = np.einsum('ijr,jkr,kr i->ijk', U_r, V_r, W_r)
            return np.sum((T_calc - 1.0)**2)
        except:
            return float('inf')

    def local_search_discrete(U, V, W, max_steps=1000):
        current_loss = loss(U, V, W)
        if current_loss < 1e-6:
            return U, V, W, True
        
        for step in range(max_steps):
            if time.time() > deadline:
                return U, V, W, False
            
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                idx = np.random.randint(n*m)
                r = np.random.randint(U.shape[1])
                current_val = U[idx, r]
                # Try nearby half-integers
                candidates = [current_val - 0.5, current_val + 0.5]
                # If value is large, try moving towards 0
                if abs(current_val) > 0.5:
                    candidates.append(current_val - 1.0 if current_val > 0 else current_val + 1.0)
                    candidates.append(0)
                
                best_candidate = None
                best_loss = current_loss
                
                for c in candidates:
                    U[idx, r] = c
                    new_loss = loss(U, V, W)
                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_candidate = c
                
                if best_candidate is not None:
                    U[idx, r] = best_candidate
                    current_loss = best_loss
                    if current_loss < 1e-6:
                        return U, V, W, True
            elif mat_idx == 1:
                idx = np.random.randint(m*p)
                r = np.random.randint(V.shape[1])
                current_val = V[idx, r]
                candidates = [current_val - 0.5, current_val + 0.5]
                if abs(current_val) > 0.5:
                    candidates.append(current_val - 1.0 if current_val > 0 else current_val + 1.0)
                    candidates.append(0)
                
                best_candidate = None
                best_loss = current_loss
                
                for c in candidates:
                    V[idx, r] = c
                    new_loss = loss(U, V, W)
                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_candidate = c
                
                if best_candidate is not None:
                    V[idx, r] = best_candidate
                    current_loss = best_loss
                    if current_loss < 1e-6:
                        return U, V, W, True
            else:
                idx = np.random.randint(p*n)
                r = np.random.randint(W.shape[1])
                current_val = W[idx, r]
                candidates = [current_val - 0.5, current_val + 0.5]
                if abs(current_val) > 0.5:
                    candidates.append(current_val - 1.0 if current_val > 0 else current_val + 1.0)
                    candidates.append(0)
                
                best_candidate = None
                best_loss = current_loss
                
                for c in candidates:
                    W[idx, r] = c
                    new_loss = loss(U, V, W)
                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_candidate = c
                
                if best_candidate is not None:
                    W[idx, r] = best_candidate
                    current_loss = best_loss
                    if current_loss < 1e-6:
                        return U, V, W, True

        return U, V, W, False

    def search_rank(R):
        num_restarts = 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # 1. Real-valued Gradient Descent
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            lr = 0.05
            for step in range(300):
                if time.time() > deadline:
                    return None
                
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                
                T_calc = np.einsum('ijr,jkr,kr i->ijk', U_r, V_r, W_r)
                error = T_calc - 1.0
                
                grad_U = 2.0 * np.einsum('ijk,jkr,kr i->ijr', error, V_r, W_r)
                grad_V = 2.0 * np.einsum('ijk,ijr,kr i->jkr', error, U_r, W_r)
                grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kr i', error, U_r, V_r)
                
                U -= lr * grad_U.reshape(n*m, R)
                V -= lr * grad_V.reshape(m*p, R)
                W -= lr * grad_W.reshape(p*n, R)
                
                if np.max(np.abs(grad_U)) < 1e-5:
                    break
            
            # 2. Round to half-integers
            U_int = round_to_half(U)
            V_int = round_to_half(V)
            W_int = round_to_half(W)
            
            # 3. Verify
            if verify(U_int, V_int, W_int):
                return U_int, V_int, W_int
            
            # 4. Local Discrete Search
            U_rep, V_rep, W_rep, success = local_search_discrete(U_int, V_int, W_int, max_steps=500)
            
            if success:
                if verify(U_rep, V_rep, W_rep):
                    return U_rep, V_rep, W_rep
            
        return None

    # Try ranks 19 to 23
    for R in range(19, 24):
        if time.time() > deadline:
            break
        result = search_rank(R)
        if result:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
