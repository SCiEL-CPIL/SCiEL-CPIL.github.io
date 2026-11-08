"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 110

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    def make_schoolbook():
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
        return U.tolist(), V.tolist(), W.tolist(), rank
    
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - START_TIME > TIME_LIMIT:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    np.random.seed(42)
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    for rank in range(30, best_rank + 1):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        found = False
        max_attempts = 10 if rank < 50 else 5
        for attempt in range(max_attempts):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Initialize with small random values
            scale = 0.1
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale
            
            # Adam optimizer
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            t = 0
            lr = 0.01
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            for step in range(300):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                
                U_mat = U.reshape(n, m, rank)
                V_mat = V.reshape(m, p, rank)
                W_mat = W.reshape(p, n, rank)
                
                current = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
                residual = current - 1.0
                
                grad_U = np.einsum('jkr,kir,ijk->ijr', V_mat, W_mat, residual)
                grad_V = np.einsum('ijr,kir,ijk->jkr', U_mat, W_mat, residual)
                grad_W = np.einsum('ijr,jkr,ijk->kir', U_mat, V_mat, residual)
                
                t += 1
                mU = beta1*mU + (1-beta1)*grad_U; vU = beta2*vU + (1-beta2)*grad_U**2
                mV = beta1*mV + (1-beta1)*grad_V; vV = beta2*vV + (1-beta2)*grad_V**2
                mW = beta1*mW + (1-beta1)*grad_W; vW = beta2*vW + (1-beta2)*grad_W**2
                mU_hat = mU/(1-beta1**t); vU_hat = vU/(1-beta2**t)
                mV_hat = mV/(1-beta1**t); vV_hat = vV/(1-beta2**t)
                mW_hat = mW/(1-beta1**t); vW_hat = vW/(1-beta2**t)
                
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            U_mat = U.reshape(n, m, rank)
            V_mat = V.reshape(m, p, rank)
            W_mat = W.reshape(p, n, rank)
            current = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
            if np.max(np.abs(current - 1.0)) < 1e-5:
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
