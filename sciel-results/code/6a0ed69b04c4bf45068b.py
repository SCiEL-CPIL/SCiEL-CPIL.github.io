import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 9
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_full(U, V, W, tol=1e-8):
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(result - T)) < tol
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try random search for lower rank
    np.random.seed(42)
    
    for rank in range(36, min(70, rank_sb)):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            
            # Local search
            for it in range(300):
                if time.time() > deadline - 10:
                    break
                
                result = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(result - T))
                
                if err < 1e-10:
                    found = True
                    break
                
                # Perturb
                scale = max(0.005, err * 0.05)
                f = np.random.randint(0, 3)
                if f == 0:
                    U[np.random.randint(0, n*m), np.random.randint(0, rank)] += np.random.randn() * scale
                elif f == 1:
                    V[np.random.randint(0, m*p), np.random.randint(0, rank)] += np.random.randn() * scale
                else:
                    W[np.random.randint(0, p*n), np.random.randint(0, rank)] += np.random.randn() * scale
            
            if found:
                break
        
        if found and verify_full(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
