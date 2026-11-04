import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    deadline = time.time() + 140
    
    # Try to find a decomposition with rank < 60
    for rank in [45, 50, 55]:
        if time.time() > deadline - 30:
            break
        
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            for it in range(5000):
                if time.time() > deadline - 15:
                    break
                
                Ur = U.reshape(n, m, rank)
                Vr = V.reshape(m, p, rank)
                Wr = W.reshape(p, n, rank)
                
                T_rec = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
                err = np.sum((T_rec - 1.0)**2)
                
                if err < 1e-6:
                    # Try rounding to half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    Ur2 = U_round.reshape(n, m, rank)
                    Vr2 = V_round.reshape(m, p, rank)
                    Wr2 = W_round.reshape(p, n, rank)
                    T_check = np.einsum('ijr,jkr,kir->ijk', Ur2, Vr2, Wr2)
                    
                    if np.max(np.abs(T_check - 1.0)) < 1e-9:
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    break
                
                resid = T_rec - 1.0
                dU = np.einsum('jkr,kir,ijk->ijr', Vr, Wr, resid)
                dV = np.einsum('ijr,kir,ijk->jkr', Ur, Wr, resid)
                dW = np.einsum('ijr,jkr,ijk->kir', Ur, Vr, resid)
                
                lr = 0.001
                U -= lr * dU.reshape(n*m, rank)
                V -= lr * dV.reshape(m*p, rank)
                W -= lr * dW.reshape(p*n, rank)
                
                Un = np.linalg.norm(U)
                Vn = np.linalg.norm(V)
                Wn = np.linalg.norm(W)
                if Un > 1e-10 and Vn > 1e-10 and Wn > 1e-10:
                    scale = (Un * Vn * Wn) ** (1/3)
                    U /= scale
                    V /= scale
                    W /= scale
    
    # Fallback: schoolbook
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
