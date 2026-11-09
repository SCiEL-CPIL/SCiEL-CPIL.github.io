"""Search for a low-rank exact tensor decomposition of 3x3 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]
N, M, P = 3, 3, 3
RANKS = [23, 24, 25, 26, 27]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
    U has shape (n*m, rank)
    V has shape (m*p, rank)  
    W has shape (p*n, rank)
    and the tensor T[u,v,w] = sum_t U[u,t] * V[v,t] * W[w,t]
    equals 1 if (u,v,w) is a valid multiplication term, else 0.
    
    The output factor follows W[k*n+i, r] convention.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Define the target tensor: T[u,v,w] = 1 iff valid multiplication term
    # u = i*m + j, v = j*p + k, w = k*n + i
    # Conditions: u%3 == v//3, u//3 == w%3, v%3 == w//3
    target = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                target[u, v, w] = 1.0
    
    def objective(x, rank):
        """Compute the sum of squared errors between the tensor from factors
        and the target tensor."""
        if time.time() > deadline:
            return float('inf')
        U = x[0:rank * N * M].reshape(rank, N * M)
        V = x[rank * N * M : 2 * rank * N * M].reshape(rank, M * P)
        W = x[2 * rank * N * M : 3 * rank * N * M].reshape(rank, P * N)
        # Compute tensor T[u,v,w] = sum_t U[u,t] * V[v,t] * W[w,t]
        # Use einsum for efficiency
        T = np.einsum('ut,vt,wt->uvw', U.T, V.T, W.T)
        diff = T - target
        return np.sum(diff * diff)
    
    def gradient(x, rank):
        """Compute gradient of the objective function."""
        if time.time() > deadline:
            return np.zeros_like(x)
        U = x[0:rank * N * M].reshape(rank, N * M)
        V = x[rank * N * M : 2 * rank * N * M].reshape(rank, M * P)
        W = x[2 * rank * N * M : 3 * rank * N * M].reshape(rank, P * N)
        
        # T[u,v,w] = sum_t U[u,t] * V[v,t] * W[w,t]
        # dT/du = sum_t (V[v,t] * W[w,t])
        # d/dU[u,t] = sum_{v,w} (V[v,t]*W[w,t]) * 2*(T[u,v,w]-target[u,v,w])
        
        T = np.einsum('ut,vt,wt->uvw', U.T, V.T, W.T)
        diff = T - target
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Gradient with respect to U
        # dU[u,t] = 2 * sum_{v,w} diff[u,v,w] * V[v,t] * W[w,t]
        for t in range(rank):
            dU[t, :] = 2.0 * np.einsum('uvw,vt,wt->u', diff, V[t, :], W[t, :])
        
        # Gradient with respect to V
        # dV[v,t] = 2 * sum_{u,w} diff[u,v,w] * U[u,t] * W[w,t]
        for t in range(rank):
            dV[t, :] = 2.0 * np.einsum('uvw,ut,wt->v', diff, U[t, :], W[t, :])
        
        # Gradient with respect to W
        # dW[w,t] = 2 * sum_{u,v} diff[u,v,w] * U[u,t] * V[v,t]
        for t in range(rank):
            dW[t, :] = 2.0 * np.einsum('uvw,ut,vt->w', diff, U[t, :], V[t, :])
        
        # Reshape gradients to match x layout
        gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return gx
    
    best_result = None
    
    for rank in RANKS:
        if time.time() > deadline:
            break
            
        max_restarts = 50 if rank >= 25 else 30
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            np.random.seed(42 + rank * 100 + restart)
            x0 = np.random.randn(rank * N * M * 3) * 0.5
            
            try:
                res = minimize(
                    objective, x0, args=(rank,),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'maxls': 5, 'ftol': 1e-10, 'gtol': 1e-8, 'maxcor': 10}
                )
                
                if res.fun < 1e-6:
                    # Verify the solution
                    U = res.x[0:rank * N * M].reshape(rank, N * M)
                    V = res.x[rank * N * M : 2 * rank * N * M].reshape(rank, M * P)
                    W = res.x[2 * rank * N * M : 3 * rank * N * M].reshape(rank, P * N)
                    
                    T = np.einsum('ut,vt,wt->uvw', U.T, V.T, W.T)
                    if np.max(np.abs(T - target)) < 1e-4:
                        best_result = {
                            'U': U.tolist(),
                            'V': V.tolist(),
                            'W': W.tolist()
                        }
                        found = True
                        break
                        
            except Exception as e:
                pass
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition of rank 27
    rank = N * M * P
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1.0
                V[j * P + k][column] = 1.0
                W[k * N + i][column] = 1.0
                column += 1
    return {'U': U, 'V': V, 'W': W}
