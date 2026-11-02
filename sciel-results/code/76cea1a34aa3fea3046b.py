import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    
    # Try rank 23
    try:
        U, V, W = _construct_rank_23()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback to rank 26
    U, V, W = _construct_rank_26()
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We need T[i,j,k] = 1 for all i,j,k
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    # Sum over R
    # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # This is a contraction.
    # We can do it by iterating over i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(U.shape[1]):
                    val += U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank_23():
    # ... implementation ...
    pass

def _construct_rank_26():
    # ... implementation ...
    pass
