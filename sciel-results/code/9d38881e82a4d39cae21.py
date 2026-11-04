import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def generate_strassen(n, m, p):
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
            V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
            W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
            
            U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
            V[0*2+0, 1] = 1
            W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
            
            U[0*2+0, 2] = 1
            V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
            W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
            
            U[1*2+1, 3] = 1
            V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
            W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
            
            U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
            V[1*2+1, 4] = 1
            W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
            
            U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
            V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
            W[1*2+1, 5] = 1
            
            U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
            V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
            W[0*2+0, 6] = 1
            
            return U, V, W

        if m > 2:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = generate_strassen(n, m1, p)
            U2, V2, W2 = generate_strassen(n, m2, p)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            
            for i in range(n):
                for j in range(m1):
                    U[i*m + j, 0:r1] = U1[i*m1 + j, :]
                for j in range(m2):
                    U[i*m + m1 + j, r1:r1+r2] = U2[i*m2 + j, :]
                    
            for j in range(m1):
                for k in range(p):
                    V[j*p + k, 0:r1] = V1[j*p + k, :]
            for j in range(m2):
                for k in range(p):
                    V[(m1+j)*p + k, r1:r1+r2] = V2[j*p + k, :]
                    
            for k in range(p):
                for i in range(n):
                    W[k*n + i, 0:r1] = W1[k*n + i, :]
                    W[k*n + i, r1:r1+r2] = W2[k*n + i, :]
                    
            return U, V, W
            
        elif p > 2:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = generate_strassen(n, m, p1)
            U2, V2, W2 = generate_strassen(n, m, p2)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            
            for i in range(n):
                for j in range(m):
                    U[i*m + j, 0:r1] = U1[i*m + j, :]
                    U[i*m + j, r1:r1+r2] = U2[i*m + j, :]
                    
            for j in range(m):
                for k in range(p1):
                    V[j*p + k, 0:r1] = V1[j*p1 + k, :]
                for k in range(p2):
                    V[j*p + p1 + k, r1:r1+r2] = V2[j*p2 + k, :]
                    
            for k in range(p1):
                for i in range(n):
                    W[k*n + i, 0:r1] = W1[k*n + i, :]
            for k in range(p2):
                for i in range(n):
                    W[(p1+k)*n + i, r1:r1+r2] = W2[k*n + i, :]
                    
            return U, V, W
            
        elif n > 2:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = generate_strassen(n1, m, p)
            U2, V2, W2 = generate_strassen(n2, m, p)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            
            for i in range(n1):
                for j in range(m):
                    U[i*m + j, 0:r1] = U1[i*m + j, :]
            for i in range(n2):
                for j in range(m):
                    U[(n1+i)*m + j, r1:r1+r2] = U2[i*m + j, :]
                    
            for j in range(m):
                for k in range(p):
                    V[j*p + k, 0:r1] = V1[j*p + k, :]
                    V[j*p + k, r1:r1+r2] = V2[j*p + k, :]
                    
            for k in range(p):
                for i in range(n1):
                    W[k*n + i, 0:r1] = W1[k*n + i, :]
                for i in range(n2):
                    W[k*n + (n1+i), r1:r1+r2] = W2[k*n + i, :]
                    
            return U, V, W
        else:
            raise ValueError("Base case not handled")

    U56, V56, W56 = generate_strassen(n, m, p)
    rank56 = U56.shape[1]
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Return the rank-56 solution
    if verify_decomposition(U56, V56, W56, n, m, p, tol=1e-9):
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    
    # Fallback
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
