"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # Precompute all valid (u_idx, v_idx, w_idx) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    triples = np.array(triples, dtype=int)
    
    def verify(U, V, W, tol=1e-9):
        U_t = U[triples[:, 0]]
        V_t = V[triples[:, 1]]
        W_t = W[triples[:, 2]]
        result = (U_t * V_t * W_t).sum(axis=1)
        return np.max(np.abs(result - 1.0)) < tol
    
    def schoolbook():
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
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower rank decompositions
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 25 or found:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            lr = 0.02
            for step in range(2000):
                if time.time() > deadline - 20:
                    break
                
                U_t = U[triples[:, 0]]
                V_t = V[triples[:, 1]]
                W_t = W[triples[:, 2]]
                prod = U_t * V_t * W_t
                residuals = 1.0 - prod.sum(axis=1)
                loss = np.sum(residuals ** 2)
                
                if loss < 1e-14:
                    if verify(U, V, W):
                        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), target_rank
                        found = True
                        break
                
                dU = -residuals[:, None] * V_t * W_t
                dV = -residuals[:, None] * U_t * W_t
                dW = -residuals[:, None] * U_t * V_t
                
                gU = np.zeros((n * m, target_rank))
                gV = np.zeros((m * p, target_rank))
                gW = np.zeros((p * n, target_rank))
                np.add.at(gU, triples[:, 0], dU)
                np.add.at(gV, triples[:, 1], dV)
                np.add.at(gW, triples[:, 2], dW)
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                if step % 500 == 499:
                    lr *= 0.5
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
