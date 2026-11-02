import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    nm = n * m
    mp = m * p
    pn = p * n
    rank_sb = n * m * p  # 54
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((nm, rank_sb), dtype=float)
    V_sb = np.zeros((mp, rank_sb), dtype=float)
    W_sb = np.zeros((pn, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Try to find better decompositions using ALS
    for rank in range(50, rank_sb):  # Try ranks 50 to 53
        if time.time() > deadline - 15:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + rank * 10 + attempt)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Alternating least squares
            for iteration in range(40):
                if time.time() > deadline - 15:
                    break
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        C = np.zeros((p, rank))
                        for k in range(p):
                            C[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C.T, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        C = np.zeros((n, rank))
                        for i in range(n):
                            C[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C.T, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except:
                            pass
                
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except:
                            pass
                
                # Check for convergence
                if check_decomposition(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break
        
        if best_rank <= rank:
            break
    
    # Convert to list of lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
