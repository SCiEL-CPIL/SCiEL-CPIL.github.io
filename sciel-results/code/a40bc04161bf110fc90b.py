import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    # Verification
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Reconstruct tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check against expected
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                Expected[a_idx, b_idx, c_idx] = 1
                
    if np.allclose(T, Expected):
        print("Valid decomposition")
    else:
        print("Invalid decomposition")
        print("Diff:", np.sum(T - Expected))
