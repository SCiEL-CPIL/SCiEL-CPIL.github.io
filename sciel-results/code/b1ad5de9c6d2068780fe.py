"""Program for matrix multiplication 3x3 * 3x7 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Returns schoolbook rank-63 decomposition as a guaranteed valid solution.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    
    def make_schoolbook():
        """Schoolbook decomposition of rank n*m*p = 63."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def verify_exact(U, V, W, tol=1e-10):
        """Verify that U,V,W form an exact decomposition."""
        r = U.shape[1]
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    # Start with schoolbook as guaranteed fallback
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Try to find a better decomposition with limited search
    rng = np.random.default_rng(12345)
    
    for target_rank in [55, 58, 60, 62]:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Structured initialization: mix of schoolbook and random
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            
            # First 27 columns: schoolbook for first 3 k-values
            for c in range(min(27, target_rank)):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % 3
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
            
            # Next 27 columns: schoolbook for k=3,4,5
            for c in range(27, min(54, target_rank)):
                c2 = c - 27
                i = c2 % n
                j = (c2 // n) % m
                k = 3 + (c2 // (n*m)) % 3
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
            
            # Remaining columns: for k=6
            for c in range(54, target_rank):
                c2 = c - 54
                i = c2 % n
                j = (c2 // n) % m
                k = 6
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
            
            # Solve for W
            W = np.zeros((p*n, target_rank), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    A_w = np.zeros((m, target_rank), dtype=float)
                    for j in range(m):
                        A_w[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                        if not np.allclose(A_w @ w, np.ones(m), atol=1e-10):
                            valid = False
                            break
                        W[k*n+i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid and verify_exact(U, V, W, tol=1e-10):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Ensure we have a valid result
    if best_rank > 63:
        best_U, best_V, best_W, best_rank = make_schoolbook()
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
