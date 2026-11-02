import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_ranks = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    # Fallback: Schoolbook rank 24
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Try to find a lower rank decomposition
    # We use a simple random restart + gradient descent approach
    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
    # We want U, V, W such that sum_r U[:,r] V[:,r] W[:,r] approximates T
    # Actually, the decomposition is T_{abc} = sum_r U_{ar} V_{br} W_{cr}
    # Here a is index in A (n*m), b in B (m*p), c in C (p*n)
    
    # Flatten T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = fallback
    best_rank = rank_school
    
    for rank in target_ranks:
        if time.time() - start_time > timeout:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() - start_time > timeout:
                    break
                    
                # Compute current tensor
                # T_curr[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Gradient wrt U[a,r] is sum_{b,c} T[a,b,c] V[b,r] W[c,r] - T_curr[a,b,c] V[b,r] W[c,r]
                # This is expensive to compute fully.
                # Instead, compute error E = T - T_curr
                # grad_U[a,r] = sum_{b,c} E[a,b,c] V[b,r] W[c,r]
                
                # Efficient computation using einsum
                # T_curr = np.einsum('ar,br,cr->abc', U, V, W)
                # E = T - T_curr
                # grad_U = np.einsum('abc,br,cr->ar', E, V, W)
                # grad_V = np.einsum('abc,ar,cr->br', E, U, W)
                # grad_W = np.einsum('abc,ar,br->cr', E, U, V)
                
                T_curr = np.einsum('ar,br,cr->abc', U, V, W)
                E = T - T_curr
                err_norm = np.linalg.norm(E)
                
                if err_norm < 1e-6:
                    # Found a solution
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify snap
                    T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, T, atol=1e-6):
                        # Convert to lists
                        U_list = U_snap.tolist()
                        V_list = V_snap.tolist()
                        W_list = W_snap.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break
                    
                grad_U = np.einsum('abc,br,cr->ar', E, V, W)
                grad_V = np.einsum('abc,ar,cr->br', E, U, W)
                grad_W = np.einsum('abc,ar,br->cr', E, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: snap periodically to help with discrete constraint
                if step % 20 == 0:
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2

    return best_result
