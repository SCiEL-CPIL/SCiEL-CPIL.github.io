"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125
    
    nz_a, nz_b, nz_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(i*m+j)
                nz_b.append(j*p+k)
                nz_c.append(k*n+i)
    nz_a = np.array(nz_a, dtype=np.int32)
    nz_b = np.array(nz_b, dtype=np.int32)
    nz_c = np.array(nz_c, dtype=np.int32)
    
    for rank in range(1, n*m*p + 1):
        if time.time() > deadline:
            break
        result = search(nz_a, nz_b, nz_c, n, m, p, rank, deadline)
        if result:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def search(nz_a, nz_b, nz_c, n, m, p, rank, deadline):
    du, dv, dw = n*m, m*p, p*n
    
    for seed in range(40):
        if time.time() > deadline:
            return None
        np.random.seed(seed * 113 + 53)
        U = np.random.randn(du, rank) * 0.2
        V = np.random.randn(dv, rank) * 0.2
        W = np.random.randn(dw, rank) * 0.2
        
        lr = 0.1
        prev_loss = np.inf
        for step in range(400):
            if time.time() > deadline:
                return None
            
            U_nz, V_nz, W_nz = U[nz_a], V[nz_b], W[nz_c]
            rec = np.einsum('ir,ir,ir->i', U_nz, V_nz, W_nz)
            err = rec - 1.0
            loss = float(np.dot(err, err))
            
            if loss < 1e-14:
                if verify(U, V, W, n, m, p):
                    return (U.tolist(), V.tolist(), W.tolist())
            
            if loss > prev_loss * 1.01 and step > 50:
                break
            prev_loss = loss
            
            gU = 2.0 * err[:, None] * V_nz * W_nz
            gV = 2.0 * err[:, None] * U_nz * W_nz
            gW = 2.0 * err[:, None] * U_nz * V_nz
            
            dU = np.zeros((du, rank))
            dV = np.zeros((dv, rank))
            dW = np.zeros((dw, rank))
            np.add.at(dU, nz_a, gU)
            np.add.at(dV, nz_b, gV)
            np.add.at(dW, nz_c, gW)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if step % 200 == 199:
                lr *= 0.5
        
        if verify(U, V, W, n, m, p):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def verify(U, V, W, n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    Trec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return bool(np.allclose(Trec, T, atol=1e-10))

# EVOLVE-BLOCK-END
