import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # Schoolbook solution
    rank_sb = n * m * p  # 32
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try to find lower rank decompositions using optimization
    for target_rank in range(20, rank_sb):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial)
            # Initialize with small random values
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3
            
            # Use Adam-like optimizer
            lr = 0.05
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                residual = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = i * m + j
                            vj = j * p + k
                            wk = k * n + i
                            val = np.dot(U[ui, :] * V[vj, :], W[wk, :])
                            r = val - 1.0
                            residual += 0.5 * r * r
                            prod = V[vj, :] * W[wk, :]
                            grad_U[ui, :] += r * prod
                            prod2 = U[ui, :] * W[wk, :]
                            grad_V[vj, :] += r * prod2
                            prod3 = U[ui, :] * V[vj, :]
                            grad_W[wk, :] += r * prod3
                
                # Adam update
                t = step + 1
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if residual < 1e-8:
                    found = True
                    break
            
            if found:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = i * m + j
                            vj = j * p + k
                            wk = k * n + i
                            val = np.dot(U_snap[ui, :] * V_snap[vj, :], W_snap[wk, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    break
        
        if found:
            break
    
    # Convert to lists with proper types
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
