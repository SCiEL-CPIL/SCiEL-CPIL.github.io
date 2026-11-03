import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 96)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        
        # U index: i*m + j
        U[i * m + j, c] = 1.0
        # V index: j*p + k
        V[j * p + k, c] = 1.0
        # W index: k*n + i (matches contract W[k*n+i, r])
        W[k * n + i, c] = 1.0
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {res['U'].shape[1]}")
    print(res["U"].shape, res["V"].shape, res["W"].shape)
