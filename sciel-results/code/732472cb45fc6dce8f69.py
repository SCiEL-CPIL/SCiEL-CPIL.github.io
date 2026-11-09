import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Indices
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Target Tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    # We don't need to store the full tensor if we iterate, but for gradient we need fast access.
    # Full tensor size: 9 * 15 * 15 = 2025. Small enough to store.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def compute_residual(U, V, W, R):
        # S[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Efficient computation:
        # S = sum_r outer(U[:, r], outer(V[:, r], W[:, r]))
        # This is a sum of rank-1 tensors.
        # We can compute this by iterating r and adding to S.
        S = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return S - T

    def compute_error(U, V, W, R):
        R_res = compute_residual(U, V, W, R)
        return np.sum(R_res**2)

    def compute_gradients(U, V, W, R):
        S = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        Res = S - T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # grad_U[a, r] = sum_{b,c} Res[a, b, c] V[b, r] W[c, r]
        # grad_V[b, r] = sum_{a,c} Res[a, b, c] U[a, r] W[c, r]
        # grad_W[c, r] = sum_{a,b} Res[a, b, c] U[a, r] V[b, r]
        
        for r in range(R):
            # Use einsum for speed
            # Res shape (n*m, m*p, p*n)
            # V[:, r] shape (m*p,)
            # W[:, r] shape (p*n,)
            
            # grad_U[:, r] = einsum('abc,b,c->a', Res, V[:, r], W[:, r])
            grad_U[:, r] = np.einsum('abc,b,c->a', Res, V[:, r], W[:, r])
            
            # grad_V[:, r] = einsum('abc,a,c->b', Res, U[:, r], W[:, r])
            grad_V[:, r] = np.einsum('abc,a,c->b', Res, U[:, r], W[:, r])
            
            # grad_W[:, r] = einsum('abc,a,b->c', Res, U[:, r], V[:, r])
            grad_W[:, r] = np.einsum('abc,a,b->c', Res, U[:, r], V[:, r])
            
        return grad_U, grad_V, grad_W, np.sum(Res**2)

    def snap_to_half_int(M):
        return np.round(M * 2) / 2

    def verify(U, V, W, R):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        # Check exact decomposition
        S = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=1e-9)

    def to_list(U, V, W, R):
        return {
            "U": [[float(U[i, c]) for c in range(R)] for i in range(n * m)],
            "V": [[float(V[j, c]) for c in range(R)] for j in range(m * p)],
            "W": [[float(W[k, c]) for c in range(R)] for k in range(p * n)]
        }

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return to_list(U, V, W, rank)

    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find better rank
    # We try ranks from 44 down to 35
    for R in range(44, 34, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple random restarts
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
                
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Simple gradient descent with backtracking
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline - 2:
                    break
                    
                grad_U, grad_V, grad_W, err = compute_gradients(U, V, W, R)
                
                if err < 1e-10:
                    break
                    
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if err < 1e-4:
                    lr = 0.001
                elif err < 1e-2:
                    lr = 0.005
                else:
                    lr = 0.01

            # Check if converged
            err = compute_error(U, V, W, R)
            if err < 1e-6:
                # Snap to half integers
                U_s = snap_to_half_int(U)
                V_s = snap_to_half_int(V)
                W_s = snap_to_half_int(W)
                
                if verify(U_s, V_s, W_s, R):
                    best_result = to_list(U_s, V_s, W_s, R)
                    best_rank = R
                    found = True
                    break
        if found:
            break

    return best_result
