"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the target tensor T[i,j,k] = delta_ik
    # T is indexed as (n*m, m*p, n*p)
    # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i
    # T[a,b,c] = 1 if j==j' and i==i' and k==k' else 0
    
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0
    
    def verify(U, V, W):
        """Check if U@V@W reconstructs T."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        recon = np.zeros((n*m, m*p, n*p))
        for r in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.allclose(recon, T, atol=1e-9)
    
    # Start with schoolbook
    rank0 = n*m*p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((n*p, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = rank0
    
    # Try to reduce rank using iterative optimization
    # Strategy: initialize with rank R, optimize to minimize ||recon - T||^2
    # If it reaches ~0, we have a valid rank-R decomposition
    
    for R in range(rank0 - 1, max(10, rank0 - 15), -1):
        if time.time() > deadline:
            break
        
        # Multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(n*p, R) * 0.5
            
            # Optimize
            lr = 0.01
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute reconstruction
                recon = np.zeros((n*m, m*p, n*p))
                for r in range(R):
                    recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                
                # Gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(R):
                    # d/dU[r] ||recon - T||^2
                    # recon[r] = U[:,r] outer V[:,r] outer W[:,r]
                    # grad_U[:,r] = 2 * (recon - T) contracted with V[:,r] and W[:,r]
                    diff = recon - T
                    grad_U[:,r] = 2.0 * np.einsum('ijk,j,k->i', diff, V[:,r], W[:,r])
                    grad_V[:,r] = 2.0 * np.einsum('ijk,i,k->j', diff, U[:,r], W[:,r])
                    grad_W[:,r] = 2.0 * np.einsum('ijk,i,j->k', diff, U[:,r], V[:,r])
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.linalg.norm(recon - T) < 1e-10:
                    break
            
            # Final check
            recon = np.zeros((n*m, m*p, n*p))
            for r in range(R):
                recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
            if np.allclose(recon, T, atol=1e-8):
                found = True
                best_rank = R
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
