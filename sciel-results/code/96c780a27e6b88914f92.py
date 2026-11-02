import numpy as np
import time
import scipy.optimize

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    
    # Precompute T_TARGET
    T_TARGET = np.zeros((nU, nV, nW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_TARGET[a, b, c] = 1.0

    # Function to check if a set of matrices is a valid decomposition
    def is_valid(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_TARGET, atol=1e-5)

    # Function to round to half-integers and check
    def try_round(U, V, W):
        # Try half-integers
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if is_valid(U_h, V_h, W_h):
            return U_h, V_h, W_h
        
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if is_valid(U_i, V_i, W_i):
            return U_i, V_i, W_i
            
        return None

    best_result = None

    # Try ranks from 41 (known lower bound) up to 95
    # If we find a valid one, return it.
    for rank in range(41, 96):
        if time.time() > deadline - 10:
            break
            
        # Define error function for this rank
        def error_func(params):
            U = params[:nU*rank].reshape(nU, rank)
            V = params[nU*rank:nU*rank + nV*rank].reshape(nV, rank)
            W = params[nU*rank + nV*rank:].reshape(nW, rank)
            T = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_TARGET
            return np.sum(diff * diff)

        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline - 5:
                break
                
            x0 = np.random.randn(nU*rank + nV*rank + nW*rank)
            
            try:
                res = scipy.optimize.minimize(error_func, x0, method='Powell', 
                                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < 1e-4:
                    U = res.x[:nU*rank].reshape(nU, rank)
                    V = res.x[nU*rank:nU*rank + nV*rank].reshape(nV, rank)
                    W = res.x[nU*rank + nV*rank:].reshape(nW, rank)
                    
                    rounded = try_round(U, V, W)
                    if rounded:
                        best_result = (rank, rounded[0], rounded[1], rounded[2])
                        break
            except Exception:
                pass
        
        if best_result:
            break

    if best_result:
        rank, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
