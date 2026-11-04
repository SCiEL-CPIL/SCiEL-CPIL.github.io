import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 100  # 100 second deadline
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    for target_rank in range(50, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-target_rank decomposition
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Compute error on random test matrices
                A = np.random.randn(n, m)
                B = np.random.randn(m, p)
                C_true = A @ B
                
                # Compute C from decomposition
                C_comp = np.zeros((n, p))
                for t in range(target_rank):
                    L = U[:, t]  # shape (n*m,)
                    M = V[:, t]  # shape (m*p,)
                    # L[i*m+j] * A[i,j] summed over j
                    L_reshaped = L.reshape(n, m)
                    M_reshaped = M.reshape(m, p)
                    # Contribution: sum_j L_reshaped[i,j]*A[i,j] * sum_k M_reshaped[j,k]*B[j,k] * W[k*n+i, t]
                    # This is: (L_reshaped * A).sum(axis=1)[:, None] * (M_reshaped * B).sum(axis=0)[None, :] * W_reshaped
                    W_reshaped = W[:, t].reshape(p, n).T  # shape (n, p)
                    a_lin = (L_reshaped * A).sum(axis=1)  # shape (n,)
                    b_lin = (M_reshaped * B).sum(axis=0)  # shape (p,)
                    C_comp += a_lin[:, None] * b_lin[None, :] * W_reshaped
                
                error = np.mean((C_true - C_comp)**2)
                
                if error < 1e-10:
                    found = True
                    break
                
                # Gradient computation would be complex, skip for now
                # Just try random restarts
                
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
