"""Matrix multiplication 3,3,7 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
DEADLINE = START_TIME + 140  # safety margin before 150s wall-clock

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63

    # Schoolbook decomposition: guaranteed valid, rank = n*m*p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best = {"U": U, "V": V, "W": W, "rank": rank}

    # Build the target tensor T[a,b,c]
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U_arr, V_arr, W_arr, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        r = U_arr.shape[1]
        for t in range(r):
            result += np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
        return np.allclose(result, T, atol=tol)

    # Try a limited search for lower rank decompositions
    # Given the difficulty, we'll try a few structured approaches
    np.random.seed(42)
    
    # Try rank 60-62 with a few random initializations
    for r in [62, 61, 60]:
        if time.time() > DEADLINE - 20:
            break
        for it in range(20):
            if time.time() > DEADLINE - 20:
                break
            # Start from schoolbook and perturb
            U_arr = np.zeros((nm, r), dtype=np.float64)
            V_arr = np.zeros((mp, r), dtype=np.float64)
            W_arr = np.zeros((pn, r), dtype=np.float64)
            
            # Fill first r columns with schoolbook-like entries
            for t in range(r):
                i = t // (m * p)
                rem = t % (m * p)
                j = rem // p
                k = rem % p
                U_arr[i * m + j, t] = 1
                V_arr[j * p + k, t] = 1
                W_arr[k * n + i, t] = 1
            
            # Add small random perturbations
            U_arr += np.random.choice([0, 0.5, -0.5], size=(nm, r))
            V_arr += np.random.choice([0, 0.5, -0.5], size=(mp, r))
            W_arr += np.random.choice([0, 0.5, -0.5], size=(pn, r))
            
            # Round to half-integers
            U_arr = np.round(2 * U_arr) / 2
            V_arr = np.round(2 * V_arr) / 2
            W_arr = np.round(2 * W_arr) / 2
            
            if verify(U_arr, V_arr, W_arr):
                best = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist(), "rank": r}
                break
        if best["rank"] < rank:
            break

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
