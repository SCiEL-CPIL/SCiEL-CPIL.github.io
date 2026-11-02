import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin
    
    n, m, p = 2, 3, 6
    nm, mp, np_ = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook as initial
    U0 = np.zeros((nm, best_rank))
    V0 = np.zeros((mp, best_rank))
    W0 = np.zeros((np_, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T."""
        pred = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(pred - T)) < tol
    
    def refine(U, V, W, max_iter=200, tol=1e-8):
        """Refine using alternating least squares with proper linear solves."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            
            # Update U: for each row a, solve (V ⊙ W)^T (V ⊙ W) u_a = (V ⊙ W)^T T[a,:,:].reshape(-1)
            # Where ⊙ is Khatri-Rao product
            VW = np.einsum('br,cr->bc', V, W)  # (mp, np_)
            # For each a: minimize ||T[a] - u_a @ VW||^2
            # Normal equations: (VW^T VW) u_a = VW^T T[a].reshape(-1)
            A = VW.T @ VW  # (rank, rank)
            for a in range(nm):
                b = VW.T @ T[a].reshape(-1)
                try:
                    U[a] = np.linalg.solve(A + 1e-10*np.eye(rank), b)
                except:
                    U[a] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Update V
            UW = np.einsum('ar,cr->ac', U, W)  # (nm, np_)
            A = UW.T @ UW
            for b in range(mp):
                vec = UW.T @ T[:,b,:].reshape(-1)
                try:
                    V[b] = np.linalg.solve(A + 1e-10*np.eye(rank), vec)
                except:
                    V[b] = np.linalg.lstsq(A, vec, rcond=None)[0]
            
            # Update W
            UV = np.einsum('ar,br->ab', U, V)  # (nm, mp)
            A = UV.T @ UV
            for c in range(np_):
                vec = UV.T @ T[:,:,c].reshape(-1)
                try:
                    W[c] = np.linalg.solve(A + 1e-10*np.eye(rank), vec)
                except:
                    W[c] = np.linalg.lstsq(A, vec, rcond=None)[0]
        
        return U, V, W
    
    # Try lower ranks
    for rank in range(18, best_rank):
        if time.time() > DEADLINE:
            break
        
        for attempt in range(15):
            if time.time() > DEADLINE:
                break
            
            # Random initialization with some structure
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(np_, rank)
            
            # Normalize to avoid scale issues
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
            
            # Refine
            U, V, W = refine(U, V, W, max_iter=100)
            
            # Check
            if verify(U, V, W, tol=1e-6):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank < n*m*p:
            break
    
    return best_result
