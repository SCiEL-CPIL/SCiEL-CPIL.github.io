import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = 115.0
    start_time = time.time()
    
    # Build rank-56 Strassen solution
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    t = 0
    for a in range(2):
        for b in range(4):
            for s in range(7):
                if s == 0:
                    U[0*m+2*a, t] += 1
                    U[1*m+2*a+1, t] += 1
                elif s == 1:
                    U[1*m+2*a, t] += 1
                    U[1*m+2*a+1, t] += 1
                elif s == 2:
                    U[0*m+2*a, t] += 1
                elif s == 3:
                    U[1*m+2*a+1, t] += 1
                elif s == 4:
                    U[0*m+2*a, t] += 1
                    U[0*m+2*a+1, t] += 1
                elif s == 5:
                    U[1*m+2*a, t] += 1
                    U[0*m+2*a, t] += -1
                elif s == 6:
                    U[0*m+2*a+1, t] += 1
                    U[1*m+2*a+1, t] += -1
                
                if s == 0:
                    V[2*a*p+2*b, t] += 1
                    V[(2*a+1)*p+2*b+1, t] += 1
                elif s == 1:
                    V[2*a*p+2*b, t] += 1
                elif s == 2:
                    V[2*a*p+2*b+1, t] += 1
                    V[(2*a+1)*p+2*b+1, t] += -1
                elif s == 3:
                    V[(2*a+1)*p+2*b, t] += 1
                    V[2*a*p+2*b, t] += -1
                elif s == 4:
                    V[(2*a+1)*p+2*b+1, t] += 1
                elif s == 5:
                    V[2*a*p+2*b, t] += 1
                    V[2*a*p+2*b+1, t] += 1
                elif s == 6:
                    V[(2*a+1)*p+2*b, t] += 1
                    V[(2*a+1)*p+2*b+1, t] += 1
                
                if s == 0:
                    W[2*b*n+0, t] += 1
                    W[(2*b+1)*n+1, t] += 1
                elif s == 1:
                    W[2*b*n+1, t] += 1
                    W[(2*b+1)*n+1, t] += -1
                elif s == 2:
                    W[(2*b+1)*n+0, t] += 1
                    W[(2*b+1)*n+1, t] += 1
                elif s == 3:
                    W[2*b*n+0, t] += 1
                    W[2*b*n+1, t] += 1
                elif s == 4:
                    W[2*b*n+0, t] += -1
                    W[(2*b+1)*n+0, t] += 1
                elif s == 5:
                    W[(2*b+1)*n+1, t] += 1
                elif s == 6:
                    W[2*b*n+0, t] += 1
                
                t += 1
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs = sum(U[i*m+j, tt] * V[j*p+k, tt] * W[k*n+i, tt] for tt in range(rank))
                    if abs(1.0 - rhs) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback to rank 64
        U = np.zeros((n*m, 64), dtype=float)
        V = np.zeros((m*p, 64), dtype=float)
        W = np.zeros((p*n, 64), dtype=float)
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, t] = 1
                    V[j*p+k, t] = 1
                    W[k*n+i, t] = 1
                    t += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
