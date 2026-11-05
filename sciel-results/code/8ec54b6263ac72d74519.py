import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_target = 23
    
    # Target tensor T
    # T[u, v, w] = 1 if u=3i+j, v=3j+k, w=3k+i
    # u in 0..8, v in 0..8, w in 0..8
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def compute_R(U, V, W):
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        return np.einsum('uc,vc,wc->uvw', U, V, W)

    def compute_grads(U, V, W):
        R = compute_R(U, V, W)
        E = R - T
        gU = 2.0 * np.einsum('uvw,vc,wc->uc', E, V, W)
        gV = 2.0 * np.einsum('uvw,uc,wc->vc', E, U, W)
        gW = 2.0 * np.einsum('uvw,uc,vc->wc', E, U, V)
        return E, gU, gV, gW

    def verify(U, V, W):
        R = compute_R(U, V, W)
        # Check if R matches T exactly (within floating point error, but we want integer/half-integer)
        # The problem asks for integer/half-integer entries.
        # So we check if rounded entries produce exact T.
        U_r = np.round(U * 2.0) / 2.0
        V_r = np.round(V * 2.0) / 2.0
        W_r = np.round(W * 2.0) / 2.0
        
        R_r = compute_R(U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True, U_r, V_r, W_r
        return False, None, None, None

    def run_adam(U, V, W, steps=2000, lr=0.01):
        # Simple Adam
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        b1 = 0.9; b2 = 0.999; eps = 1e-8
        
        for t in range(1, steps + 1):
            E, gU, gV, gW = compute_grads(U, V, W)
            loss = np.sum(E * E)
            
            mU = b1 * mU + (1 - b1) * gU
            vU = b2 * vU + (1 - b2) * (gU ** 2)
            mU_hat = mU / (1 - b1 ** t)
            vU_hat = vU / (1 - b2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = b1 * mV + (1 - b1) * gV
            vV = b2 * vV + (1 - b2) * (gV ** 2)
            mV_hat = mV / (1 - b1 ** t)
            vV_hat = vV / (1 - b2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * (gW ** 2)
            mW_hat = mW / (1 - b1 ** t)
            vW_hat = vW / (1 - b2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if t % 100 == 0 and loss < 1e-6:
                break
                
        return U, V, W, loss

    start_time = time.time()
    deadline = 120.0
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find rank 23
    attempts = 0
    while time.time() - start_time < deadline:
        attempts += 1
        # Random init
        U = np.random.randn(n*m, rank_target) * 0.1
        V = np.random.randn(m*p, rank_target) * 0.1
        W = np.random.randn(p*n, rank_target) * 0.1
        
        # Run optimization
        U, V, W, loss = run_adam(U, V, W, steps=1000, lr=0.01)
        
        if loss < 1e-8:
            is_valid, U_r, V_r, W_r = verify(U, V, W)
            if is_valid:
                # Convert to lists
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        # If time is running out, break
        if time.time() - start_time > deadline - 10:
            break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
