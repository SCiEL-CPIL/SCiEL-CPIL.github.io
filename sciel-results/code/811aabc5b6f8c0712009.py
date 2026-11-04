import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First, build the known rank-55 decomposition as a fallback
    def build_rank55():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # 5 Strassen blocks on the 2x2 part (j=0,1), each 2 columns
        for block in range(5):
            k0 = 2 * block
            k1 = 2 * block + 1
            
            # M1: (A00 + A11) * (B0k0 + B1k1)
            U[0][col] += 1; U[4][col] += 1
            V[k0][col] += 1; V[10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1; W[k1 * 2 + 1][col] += 1
            col += 1
            
            # M2: (A10 + A11) * B0k0
            U[3][col] += 1; U[4][col] += 1
            V[k0][col] += 1
            W[k0 * 2 + 1][col] += 1; W[k1 * 2 + 1][col] -= 1
            col += 1
            
            # M3: A00 * (B0k1 - B1k1)
            U[0][col] += 1
            V[k1][col] += 1; V[10 + k1][col] -= 1
            W[k1 * 2 + 0][col] += 1; W[k1 * 2 + 1][col] += 1
            col += 1
            
            # M4: A11 * (B1k0 - B0k0)
            U[4][col] += 1
            V[10 + k0][col] += 1; V[k0][col] -= 1
            W[k0 * 2 + 0][col] += 1; W[k0 * 2 + 1][col] += 1
            col += 1
            
            # M5: (A00 + A01) * B1k1
            U[0][col] += 1; U[1][col] += 1
            V[10 + k1][col] += 1
            W[k0 * 2 + 0][col] -= 1; W[k1 * 2 + 0][col] += 1
            col += 1
            
            # M6: (A10 - A00) * (B0k0 + B0k1)
            U[3][col] += 1; U[0][col] -= 1
            V[k0][col] += 1; V[k1][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1
            
            # M7: (A01 - A11) * (B1k0 + B1k1)
            U[1][col] += 1; U[4][col] -= 1
            V[10 + k0][col] += 1; V[10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1
            col += 1
        
        # Schoolbook for the 2x1x10 part (j=2)
        for i in range(2):
            for k in range(10):
                U[i * 3 + 2][col] += 1
                V[2 * 10 + k][col] += 1
                W[k * 2 + i][col] += 1
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    best_result = build_rank55()
    best_rank = 55
    
    # Verify the rank-55 solution
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower-rank solutions using alternating least squares
    np.random.seed(42)
    
    for r in range(30, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple initializations
        for init in range(3):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve least squares
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j, :] appears in: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                        # Let B_k[c] = V[j*p+k,c] * W[k*n+i,c]
                        # Then U[i*m+j,:] @ B.T = ones(p)
                        B = np.zeros((p, r))
                        for k in range(p):
                            B[k, :] = V[j*p+k] * W[k*n+i]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve least squares
                for j in range(m):
                    for k in range(p):
                        # V[j*p+k, :] appears in: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let C_i[c] = U[i*m+j,c] * W[k*n+i,c]
                        # Then V[j*p+k,:] @ C.T = ones(n)
                        C = np.zeros((n, r))
                        for i in range(n):
                            C[i, :] = U[i*m+j] * W[k*n+i]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(C.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve least squares
                for k in range(p):
                    for i in range(n):
                        # W[k*n+i, :] appears in: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                        # Let D_j[c] = U[i*m+j,c] * V[j*p+k,c]
                        # Then W[k*n+i,:] @ D.T = ones(m)
                        D = np.zeros((m, r))
                        for j in range(m):
                            D[j, :] = U[i*m+j] * V[j*p+k]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(D.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_res = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_res = max(max_res, abs(val - 1.0))
                if max_res < 1e-8:
                    found = True
                    break
            
            if found:
                break
        
        if found:
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # Continue to try even lower ranks
            continue
    
    return best_result
