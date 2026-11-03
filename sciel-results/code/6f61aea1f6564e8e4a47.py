import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nmp = n * m * p  # 30
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def tensor_from_decomp(U, V, W):
        r = U.shape[1]
        result = np.zeros((n*m, m*p, p*n), dtype=float)
        for c in range(r):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return result
    
    # Schoolbook
    U0 = np.zeros((n*m, nmp))
    V0 = np.zeros((m*p, nmp))
    W0 = np.zeros((p*n, nmp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1.0
                V0[j*p + k, col] = 1.0
                W0[k*n + i, col] = 1.0
                col += 1
    
    best_rank = nmp
    best_U, best_V, best_W = U0, V0, W0
    
    def solve_U(V, W, T):
        r = V.shape[1]
        M = np.zeros((m*p * p*n, r))
        for c in range(r):
            for j in range(m*p):
                for k in range(p*n):
                    M[j*(p*n) + k, c] = V[j, c] * W[k, c]
        T_flat = T.reshape(n*m, m*p*p*n)
        U_sol = np.linalg.lstsq(M, T_flat.T, rcond=None)[0].T
        return U_sol
    
    def solve_V(U, W, T):
        r = U.shape[1]
        M = np.zeros((n*m * p*n, r))
        for c in range(r):
            for i in range(n*m):
                for k in range(p*n):
                    M[i*(p*n) + k, c] = U[i, c] * W[k, c]
        T_flat = T.reshape(m*p, n*m*p*n)
        V_sol = np.linalg.lstsq(M, T_flat.T, rcond=None)[0].T
        return V_sol
    
    def solve_W(U, V, T):
        r = U.shape[1]
        M = np.zeros((n*m * m*p, r))
        for c in range(r):
            for i in range(n*m):
                for j in range(m*p):
                    M[i*(m*p) + j, c] = U[i, c] * V[j, c]
        T_flat = T.reshape(p*n, n*m*m*p)
        W_sol = np.linalg.lstsq(M, T_flat.T, rcond=None)[0].T
        return W_sol
    
    # Alternating least squares
    for target_rank in range(28, 15, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + target_rank * 100 + trial)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            prev_loss = float('inf')
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                U = solve_U(V, W, T)
                V = solve_V(U, W, T)
                W = solve_W(U, V, T)
                
                T_pred = tensor_from_decomp(U, V, W)
                loss = np.sum((T_pred - T) ** 2)
                
                if loss < 1e-10:
                    found = True
                    break
                
                if loss > prev_loss * 1.01:
                    break
                prev_loss = loss
            
            if found:
                break
        
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    def to_frac_list(mat):
        result = []
        for row in mat:
            frow = []
            for val in row:
                if abs(val) < 1e-10:
                    frow.append(Fraction(0))
                elif abs(val - round(val)) < 1e-9:
                    frow.append(Fraction(int(round(val))))
                else:
                    f = Fraction(val).limit_denominator(1000000)
                    frow.append(f)
            result.append(frow)
        return result
    
    return {"U": to_frac_list(best_U), "V": to_frac_list(best_V), "W": to_frac_list(best_W)}
