import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor M flattened
    # M[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # else 0
    M = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0

    # Schoolbook solution
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    
    # Try structured algebraic approach using Kronecker products
    # Strategy: Embed 2x3x7 into a larger tensor space using known optimal decompositions
    # Use Strassen's algorithm for 2x2x2 (rank 7) and scalar multiplication for remaining dimensions
    
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Method 1: Try to use Strassen-style decomposition
    # For 2x3x7, we can think of it as:
    # - 2x2 part: use Strassen (rank 7 instead of 8)
    # - 3x7 part: try to optimize
    
    # Let's try a more systematic approach:
    # Decompose the multiplication into blocks and use known results
    
    # The key insight: if we can write 2x3x7 as a sub-problem of a larger 
    # multiplication with known optimal rank, we might get a better bound.
    
    # Try rank values below schoolbook (42)
    for target_rank in [41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() - start_time > deadline:
            break
            
        # Try to find a solution at this rank using random initialization + gradient descent
        found = False
        for restart in range(5):
            if time.time() - start_time > deadline:
                    break
                    
            # Initialize with random values
            np.random.seed(42 + target_rank * 100 + restart)
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Use Adam-like optimizer for better convergence
            lr = 0.05
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 0
            
            for step in range(200):
                if time.time() - start_time > deadline:
                    break
                    
                t += 1
                # Compute current tensor T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # Efficient computation using einsum
                T = np.einsum('at,bt,ct->abc', U, V, W)
                
                # Compute error
                E = T - M
                err_norm = np.linalg.norm(E)
                
                if err_norm < 1e-10:
                    found = True
                    break
                
                # Compute gradients
                # dE/dU[a,t] = sum_{b,c} E[a,b,c] * V[b,t] * W[c,t]
                # dE/dV[b,t] = sum_{a,c} E[a,b,c] * U[a,t] * W[c,t]
                # dE/dW[c,t] = sum_{a,b} E[a,b,c] * U[a,t] * V[b,t]
                
                dU = np.einsum('abc,bt,ct->at', E, V, W)
                dV = np.einsum('abc,at,ct->bt', E, U, W)
                dW = np.einsum('abc,at,bt->ct', E, U, V)
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * dU
                v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
                m_hat_U = m_U / (1 - beta1 ** t)
                v_hat_U = v_U / (1 - beta2 ** t)
                U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * dV
                v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
                m_hat_V = m_V / (1 - beta1 ** t)
                v_hat_V = v_V / (1 - beta2 ** t)
                V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
                m_hat_W = m_W / (1 - beta1 ** t)
                v_hat_W = v_W / (1 - beta2 ** t)
                W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                
                if step % 20 == 0:
                    # Check convergence
                    T_check = np.einsum('at,bt,ct->abc', U, V, W)
                    if np.linalg.norm(T_check - M) < 1e-10:
                        found = True
                        break
            
            if found:
                # Verify exact solution
                T_final = np.einsum('at,bt,ct->abc', U, V, W)
                if np.allclose(T_final, M, atol=1e-10):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
        
        if found:
            break
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
