import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    start = time.time()
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank 30
    U = np.zeros((n*m, 30))
    V = np.zeros((m*p, 30))
    W = np.zeros((p*n, 30))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try ALS for lower ranks
    for rank in [29, 28, 27, 26, 25, 24]:
        if time.time() - start > 100:
            break
        for trial in range(20):
            if time.time() - start > 100:
                break
            np.random.seed(trial * 10000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            for it in range(500):
                if time.time() - start > 100:
                    break
                # Update U
                # For each a: U[a,:] = argmin ||T[a] - sum_r U[a,r] (V[r,:] ⊗ W[r,:])||
                # Flatten T[a] to (m*p*p*n,)
                # The design matrix for U[a,:] is the Khatri-Rao product of V and W
                # Khatri-Rao: column r is V[:,r] ⊗ W[:,r], shape (m*p*p*n,)
                # This is (m*p*p*n, rank) = (150, rank)
                VW = np.zeros((m*p*p*n, rank))
                for r in range(rank):
                    VW[:,r] = np.kron(V[:,r], W[:,r])
                
                # Solve least squares for each a
                T_flat = T.reshape(n*m, -1)
                # U = T_flat @ VW^+ (pseudo-inverse)
                U, _, _, _ = np.linalg.lstsq(VW, T_flat.T, rcond=None)
                
                # Update V
                UW = np.zeros((n*m*p*n, rank))
                for r in range(rank):
                    UW[:,r] = np.kron(U[:,r], W[:,r])
                T_flat2 = T.transpose(1,0,2).reshape(m*p, -1)
                V, _, _, _ = np.linalg.lstsq(UW, T_flat2.T, rcond=None)
                
                # Update W
                UV = np.zeros((n*m*m*p, rank))
                for r in range(rank):
                    UV[:,r] = np.kron(U[:,r], V[:,r])
                T_flat3 = T.transpose(2,0,1).reshape(p*n, -1)
                W, _, _, _ = np.linalg.lstsq(UV, T_flat3.T, rcond=None)
            
            # Check error
            error = 0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        val = sum(U[a,r]*V[b,r]*W[c,r] for r in range(rank))
                        error += (val - T[a,b,c])**2
            if error < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
