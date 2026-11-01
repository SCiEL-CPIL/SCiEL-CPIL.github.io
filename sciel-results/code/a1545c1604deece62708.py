import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    # Target Tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0

    def residuals(params, r):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        
        rec = np.zeros((n*m, m*p, p*n))
        for c in range(r):
            rec += np.einsum('a,b,d->abd', U[:,c], V[:,c], W[:,c])
            
        return (rec - T).flatten()

    def try_rank(r, max_attempts=10):
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                return None
            np.random.seed(attempt + 100)
            U_init = np.random.randn(n*m, r) * 0.1
            V_init = np.random.randn(m*p, r) * 0.1
            W_init = np.random.randn(p*n, r) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(residuals, params_init, args=(r,), method='trf', max_nfev=2000)
                if res.cost < 1e-10:
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    rec_snap = np.zeros((n*m, m*p, p*n))
                    for c in range(r):
                        rec_snap += np.einsum('a,b,d->abd', U_snap[:,c], V_snap[:,c], W_snap[:,c])
                    
                    if np.allclose(rec_snap, T, atol=1e-9):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                pass
        return None

    # Try rank 10
    sol = try_rank(10, max_attempts=5)
    if sol:
        return sol

    # Fallback to rank 11
    if time.time() - start_time < timeout:
        # Construct rank 11
        # Laderman for 2x2x2 (cols 0,1 of B)
        # Laderman factors for 2x2x2:
        # U (4x7), V (4x7), W (4x7) in 2x2x2 indexing
        # 2x2x2 indices: i,j in 0..1, k in 0..1
        # U_idx = i*2+j
        # V_idx = j*2+k
        # W_idx = k*2+i
        
        # We need to map this to 2x2x3 indices
        # U_idx is same: i*2+j
        # V_idx for k=0,1: j*3+k
        # W_idx for k=0,1: k*2+i
        
        # Laderman decomposition (one version):
        # c0: U[0,0]=1, V[0,0]=1, W[0,0]=1
        # c1: U[0,1]=1, V[1,0]=1, W[1,1]=1  -- Wait, let's use a standard one.
        
        # Standard Laderman:
        # 1: A00 B00 -> C00
        # 2: A00 B11 -> C01
        # 3: A10 B00 -> C10
        # 4: A11 B11 -> C11
        # 5: (A00+A11)(B00+B11) -> C00+C11
        # 6: (A01+A10)(B01+B10) -> C01+C10
        # 7: (A00+A01+A10+A11)(B00+B01+B10+B11) ... no, that's not right.
        
        # Let's just use the schoolbook for the first 2 columns (rank 4) + schoolbook for last column (rank 4) = rank 8?
        # No, schoolbook for 2x2x2 is rank 4.
        # So 4 + 4 = 8?
        # Wait, schoolbook for 2x2x2 is 4 multiplications.
        # A00B00, A01B10, A10B01, A11B11.
        # So rank 4.
        # Then total rank 4 + 4 = 8?
        # But we know rank of 2x2x2 is 7.
        # So using schoolbook for the first part is suboptimal.
        # But rank 8 is better than 12.
        
        # Let's just construct a valid rank 12 solution if 10 fails, to be safe.
        # Or rank 11 using Laderman.
        # Since I don't have Laderman hardcoded, I will stick to 12 if 10 fails.
        # 12 is guaranteed to work.
        
        pass

    # Fallback to rank 12 (Schoolbook)
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
