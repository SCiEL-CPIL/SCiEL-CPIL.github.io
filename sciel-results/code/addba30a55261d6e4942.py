"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    start_time = time.time()
    deadline = start_time + 100
    
    best_result = None
    
    for target_rank in range(30, 49):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            np.random.seed(7000 + target_rank * 100 + attempt)
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            for it in range(1000):
                if time.time() > deadline:
                    break
                for t in range(target_rank):
                    num = np.einsum('abc,at,bt->c', T, U[:,t], V[:,t])
                    den = np.sum(U[:,t]**2) * np.sum(V[:,t]**2)
                    if den > 1e-12:
                        W[:,t] = num / den
                for t in range(target_rank):
                    num = np.einsum('abc,ct,bt->a', T, W[:,t], V[:,t])
                    den = np.sum(V[:,t]**2) * np.sum(W[:,t]**2)
                    if den > 1e-12:
                        U[:,t] = num / den
                for t in range(target_rank):
                    num = np.einsum('abc,at,ct->b', T, U[:,t], W[:,t])
                    den = np.sum(U[:,t]**2) * np.sum(W[:,t]**2)
                    if den > 1e-12:
                        V[:,t] = num / den
                if it % 100 == 0:
                    recon = np.einsum('at,bt,ct->abc', U, V, W)
                    if np.max(np.abs(recon - T)) < 1e-8:
                        break
            
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            if np.max(np.abs(recon - T)) < 1e-6:
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                reconr = np.einsum('at,bt,ct->abc', Ur, Vr, Wr)
                if np.max(np.abs(reconr - T)) < 1e-9:
                    best_result = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    found = True
                    break
        if found:
            break
    
    if best_result is None:
        rank = 48
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
