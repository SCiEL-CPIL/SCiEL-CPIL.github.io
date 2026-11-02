"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # The multiplication tensor has 24 nonzero entries (all ones)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Try to find a rank-20 decomposition
    # Use a structured approach: initialize with schoolbook and try to merge terms
    
    # Start with schoolbook rank 24
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower rank by optimization
    for target_rank in [20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
        
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            
            # Initialize close to a scaled version of schoolbook
            U = U0[:, :target_rank] * 2.0  # Scale up to compensate for fewer terms
            V = V0[:, :target_rank] * 2.0
            W = W0[:, :target_rank] * 2.0
            
            # Add small random perturbation
            U += np.random.randn(n*m, target_rank) * 0.1
            V += np.random.randn(m*p, target_rank) * 0.1
            W += np.random.randn(p*n, target_rank) * 0.1
            
            # Iterative refinement
            for it in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Compute errors
                errs = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errs[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                
                err = np.sum((errs - 1)**2)
                if err < 1e-12:
                    # Check if we can convert to exact fractions
                    Uf = [[Fraction(U[i,c]).limit_denominator(100000) for c in range(target_rank)] for i in range(n*m)]
                    Vf = [[Fraction(V[i,c]).limit_denominator(100000) for c in range(target_rank)] for i in range(m*p)]
                    Wf = [[Fraction(W[i,c]).limit_denominator(100000) for c in range(target_rank)] for i in range(p*n)]
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(Uf[i*m+j][c]*Vf[j*p+k][c]*Wf[k*n+i][c] for c in range(target_rank))
                                if s != 1:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        return {"U": Uf, "V": Vf, "W": Wf}
                    break
                
                # Gradient step (simplified)
                lr = 0.001
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = errs[i,j,k] - 1
                            dU[i*m+j,:] += e * V[j*p+k,:] * W[k*n+i,:]
                            dV[j*p+k,:] += e * U[i*m+j,:] * W[k*n+i,:]
                            dW[k*n+i,:] += e * U[i*m+j,:] * V[j*p+k,:]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                lr *= 0.999
    
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
