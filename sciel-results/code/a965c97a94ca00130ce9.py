import numpy as np
import time

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill
    
    # Build target tensor T of shape (N*M, M*P, P*N)
    T = np.zeros((N*M, M*P, P*N), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, T, tol=1e-9):
        """Check if sum_t U[a,t]*V[b,t]*W[c,t] == T[a,b,c] for all a,b,c."""
        D = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(D, T, atol=tol)
    
    def schoolbook():
        rank = N * M * P
        U = np.zeros((N*M, rank))
        V = np.zeros((M*P, rank))
        W = np.zeros((P*N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i*M + j, col] = 1.0
                    V[j*P + k, col] = 1.0
                    W[k*N + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try to find a lower-rank decomposition using structured search
    # Key insight: A rank-1 term (u, v, w) contributes u[a]*v[b]*w[c] to T[a,b,c]
    # For the decomposition to be exact, we need sum_t u_t[a]*v_t[b]*w_t[c] = T[a,b,c]
    
    # Strategy: Try ranks from 30 to 35, using a combination of:
    # 1. Random initialization with grid search
    # 2. Local refinement
    
    def try_structured_rank(r, max_restarts=20):
        """Try to find a rank-r decomposition."""
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            rng = np.random.default_rng(restart * 1000 + 42)
            
            # Initialize with random values
            U = rng.normal(0, 1.0, (N*M, r))
            V = rng.normal(0, 1.0, (M*P, r))
            W = rng.normal(0, 1.0, (P*N, r))
            
            # Try to refine using gradient descent on the squared error
            for iteration in range(2000):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                # Compute current decomposition
                D = np.einsum('at,bt,ct->abc', U, V, W)
                E = T - D
                
                # Compute gradients
                # dL/dU[a,t] = -2 * sum_{b,c} E[a,b,c] * V[b,t] * W[c,t]
                # dL/dV[b,t] = -2 * sum_{a,c} E[a,b,c] * U[a,t] * W[c,t]
                # dL/dW[c,t] = -2 * sum_{a,b} E[a,b,c] * U[a,t] * V[b,t]
                
                # Use einsum for efficient computation
                grad_U = -2.0 * np.einsum('abc,bt,ct->at', E, V, W)
                grad_V = -2.0 * np.einsum('abc,at,ct->bt', E, U, W)
                grad_W = -2.0 * np.einsum('abc,at,bt->ct', E, U, V)
                
                # Adaptive learning rate
                loss = np.sum(E**2)
                if loss < 1e-10:
                    break
                    
                lr = 0.01 / (1 + iteration * 0.01)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check if we're close to a solution
                if loss < 1e-8:
                    # Try rounding to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if verify(U_rounded, V_rounded, W_rounded, T):
                        return U_rounded, V_rounded, W_rounded
                    else:
                        # Try different rounding strategies
                        for offset in [-0.25, -0.1, 0.1, 0.25]:
                            U_r = np.round((U + offset) * 2) / 2
                            V_r = np.round((V + offset) * 2) / 2
                            W_r = np.round((W + offset) * 2) / 2
                            if verify(U_r, V_r, W_r, T):
                                return U_r, V_r, W_r
                        # Accept the float solution if very close
                        if loss < 1e-6:
                            return U, V, W
        
        return None
    
    # Try ranks from 30 to 35 (schoolbook is 36)
    for r in range(30, 36):
        if time.time() - start_time > time_limit - 10:
            break
            
        result = try_structured_rank(r, max_restarts=10)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
