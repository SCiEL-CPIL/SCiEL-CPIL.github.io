import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    # Try rank 21
    r = 21
    
    # Flatten U, V, W
    # U: (nm, r) -> nm*r
    # V: (mp, r) -> mp*r
    # W: (pn, r) -> pn*r
    total_vars = nm * r + mp * r + pn * r
    
    def unpack(x):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = x[nm*r+mp*r:].reshape(pn, r)
        return U, V, W
    
    def residuals(x):
        U, V, W = unpack(x)
        # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, r)
        U3 = U.reshape(n, m, r)
        # Reshape V to (m, p, r)
        V3 = V.reshape(m, p, r)
        # Reshape W to (p, n, r)
        W3 = W.reshape(p, n, r)
        
        # We need sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is a tensor contraction.
        # Result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        
        # We can compute this by iterating over c or using einsum.
        # einsum: 'ijc,jkc,kic->ijk'
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Residuals are result - T
        res = result - T
        return res.flatten()
    
    # Initial guess: Schoolbook-like
    # Set U, V, W such that they are identity-like?
    # Or just random.
    x0 = np.random.randn(total_vars)
    
    try:
        # Use least_squares
        # Method 'lm' is good for small problems, but might be slow.
        # 'trf' is default.
        sol = least_squares(residuals, x0, method='trf', max_nfev=10000)
        
        if sol.cost < 1e-10:
            U, V, W = unpack(sol.x)
            # Round to nearest integer or half-integer
            # The problem says "integer/half-integer entries"
            # Multiply by 2, round, divide by 2
            U2 = np.round(U * 2) / 2
            V2 = np.round(V * 2) / 2
            W2 = np.round(W * 2) / 2
            
            # Verify
            U3 = U2.reshape(n, m, r)
            V3 = V2.reshape(m, p, r)
            W3 = W2.reshape(p, n, r)
            result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            
            if np.allclose(result, T, atol=1e-9):
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
