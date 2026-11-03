import numpy as np
import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 3, 3, 4
    nm, mp, pn = n*m, m*p, p*n  # 9, 12, 12
    n_eq = n*m*p  # 36
    
    def residual(U, V, W):
        """Compute sum of squared errors."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        return err
    
    def verify_exact(U, V, W, rank):
        """Check if entries are integer/half-integer and decomposition is exact."""
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    if isinstance(x, Fraction):
                        continue
                    x = float(x)
                    if abs(x * 2 - round(x * 2)) > 1e-9:
                        return False
        # Verify with Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        u = U[i*m+j, c] if isinstance(U[i*m+j, c], Fraction) else Fraction(U[i*m+j, c])
                        v = V[j*p+k, c] if isinstance(V[j*p+k, c], Fraction) else Fraction(V[j*p+k, c])
                        w = W[k*n+i, c] if isinstance(W[k*n+i, c], Fraction) else Fraction(W[k*n+i, c])
                        s += u * v * w
                    if s != 1:
                        return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return Fraction(round(x * 2), 2)
    
    # Try ranks from 21 to 35
    for rank in range(21, 36):
        if time.time() > deadline - 20:
            break
        
        max_restarts = 30 if rank <= 25 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(restart * 7919 + rank * 13)
            
            # Initialize with small random values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Gradient descent with adaptive step size
            best_res = residual(U, V, W)
            lr = 0.01
            stuck = 0
            
            for it in range(500):
                if time.time() > deadline - 3:
                    break
                
                # Compute gradients numerically
                eps = 1e-5
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                # Sample a subset of entries for gradient (for speed)
                idx = np.random.choice(nm * rank, min(20, nm*rank), replace=False)
                for idx_flat in idx:
                    i, c = divmod(idx_flat, rank)
                    orig = U[i, c]
                    U[i, c] = orig + eps
                    gU[i, c] = (residual(U, V, W) - best_res) / eps
                    U[i, c] = orig
                
                idx = np.random.choice(mp * rank, min(20, mp*rank), replace=False)
                for idx_flat in idx:
                    i, c = divmod(idx_flat, rank)
                    orig = V[i, c]
                    V[i, c] = orig + eps
                    gV[i, c] = (residual(U, V, W) - best_res) / eps
                    V[i, c] = orig
                
                idx = np.random.choice(pn * rank, min(20, pn*rank), replace=False)
                for idx_flat in idx:
                    i, c = divmod(idx_flat, rank)
                    orig = W[i, c]
                    W[i, c] = orig + eps
                    gW[i, c] = (residual(U, V, W) - best_res) / eps
                    W[i, c] = orig
                
                # Update
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                new_res = residual(U, V, W)
                if new_res < best_res:
                    best_res = new_res
                    stuck = 0
                    lr = min(lr * 1.05, 0.1)
                else:
                    stuck += 1
                    lr *= 0.9
                    # Revert
                    U += lr * gU
                    V += lr * gV
                    W += lr * gW
                
                if best_res < 1e-10:
                    break
                
                if stuck > 50:
                    # Perturb
                    U += np.random.randn(nm, rank) * 0.1
                    V += np.random.randn(mp, rank) * 0.1
                    W += np.random.randn(pn, rank) * 0.1
                    best_res = residual(U, V, W)
                    stuck = 0
            
            if best_res < 1e-8:
                # Try to rationalize
                U_rat = np.array([[to_half_int(x) for x in row] for row in U], dtype=object)
                V_rat = np.array([[to_half_int(x) for x in row] for row in V], dtype=object)
                W_rat = np.array([[to_half_int(x) for x in row] for row in W], dtype=object)
                
                if verify_exact(U_rat, V_rat, W_rat, rank):
                    # Convert to float for return
                    U_list = [[float(x) for x in row] for row in U_rat]
                    V_list = [[float(x) for x in row] for row in V_rat]
                    W_list = [[float(x) for x in row] for row in W_rat]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we got close but not exact, try local search around the best solution
        # (This is a fallback - skip for now due to time constraints)
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
