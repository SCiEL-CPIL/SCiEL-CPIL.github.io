import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > tol:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _round_to_half_ints(X):
    return np.round(X * 2) / 2

def _alternating_minimization(rank, max_iter=100, seed=42):
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    T[i*m+j, jp*p+k, k*n+i] = 1.0 if j == jp else 0.0
    
    U = rng.randn(n*m, rank) * 0.3
    V = rng.randn(m*p, rank) * 0.3
    W = rng.randn(p*n, rank) * 0.3
    
    for it in range(max_iter):
        # Fix U,V solve W
        F = np.einsum('ar,br->abr', U, V).reshape(-1, rank)
        FTF = F.T @ F + 1e-12 * np.eye(rank)
        FTF_inv = np.linalg.pinv(FTF)
        for c in range(p*n):
            W[c, :] = FTF_inv @ (F.T @ T[:, :, c].flatten())
        
        # Fix U,W solve V
        G = np.einsum('ar,cr->acr', U, W).reshape(-1, rank)
        GTF = G.T @ G + 1e-12 * np.eye(rank)
        GTF_inv = np.linalg.pinv(GTF)
        for b in range(m*p):
            V[b, :] = GTF_inv @ (G.T @ T[:, b, :].flatten())
        
        # Fix V,W solve U
        H = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
        HTF = H.T @ H + 1e-12 * np.eye(rank)
        HTF_inv = np.linalg.pinv(HTF)
        for a in range(n*m):
            U[a, :] = HTF_inv @ (H.T @ T[a, :, :].flatten())
    
    err = np.sum(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T))
    if err < 1e-6:
        return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    for rank in [20, 21, 22, 23, 24, 25, 26]:
        if time.time() > deadline - 15:
            break
        for seed in range(30):
            if time.time() > deadline - 15:
                break
            result = _alternating_minimization(rank, max_iter=80, seed=seed)
            if result is not None:
                U, V, W = result
                Ur = _round_to_half_ints(U)
                Vr = _round_to_half_ints(V)
                Wr = _round_to_half_ints(W)
                if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), tol=1e-8):
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    
    return _schoolbook()
# EVOLVE-BLOCK-END
