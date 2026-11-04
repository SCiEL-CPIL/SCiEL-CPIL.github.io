import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Target tensor indices (non-zero entries)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = a * (mp * pn) + b * pn + c
                target_indices.append(idx)
    target_indices = np.array(target_indices)
    
    # Zero indices
    all_indices = np.arange(nm * mp * pn)
    zero_mask = np.ones(len(all_indices), dtype=bool)
    zero_mask[target_indices] = False
    zero_indices = all_indices[zero_mask]

    def compute_tensor(U, V, W, rank):
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # T = sum_r U[:,r] V[:,r] W[:,r]
        # Using einsum is efficient for small tensors
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return T.flatten()

    def error_func(x, rank):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        T_flat = compute_tensor(U, V, W, rank)
        
        # Residuals: 
        # 1. T[target] - 1
        # 2. T[zero]
        res_nonzero = T_flat[target_indices] - 1.0
        res_zero = T_flat[zero_indices]
        
        return np.concatenate([res_nonzero, res_zero])

    def check_valid(U, V, W, rank, tol=1e-9):
        T_flat = compute_tensor(U, V, W, rank)
        if np.max(np.abs(T_flat[target_indices] - 1.0)) > tol:
            return False
        if np.max(np.abs(T_flat[zero_indices])) > tol:
            return False
        return True

    def snap_to_half_integer(M):
        return np.round(M * 2.0) / 2.0

    best_rank = 24
    best_U = None
    best_V = None
    best_W = None

    # Trivial solution as fallback
    U_trivial = np.zeros((nm, 24))
    V_trivial = np.zeros((mp, 24))
    W_trivial = np.zeros((pn, 24))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    # Search for lower ranks
    # Try ranks from 12 upwards
    for rank in range(12, 24):
        if time.time() > deadline - 10.0:
            break
            
        found = False
        # Number of restarts depends on time remaining
        # We have ~140s total. Let's allocate time per rank.
        # Rank 12 is hardest, give it more time.
        max_restarts = 20 if rank <= 15 else 10
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            np.random.seed(123 + attempt * 7 + rank)
            
            # Initialize with small random values
            scale = 1.0
            U_init = np.random.randn(nm, rank) * scale
            V_init = np.random.randn(mp, rank) * scale
            W_init = np.random.randn(pn, rank) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use least_squares for robustness
                res = least_squares(
                    lambda x: error_func(x, rank), 
                    x0, 
                    method='trf', 
                    max_nfev=10000,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12
                )
                
                if res.cost < 1e-10:
                    U_sol = res.x[:nm*rank].reshape(nm, rank)
                    V_sol = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Snap to half-integers
                    U_s = snap_to_half_integer(U_sol)
                    V_s = snap_to_half_integer(V_sol)
                    W_s = snap_to_half_integer(W_sol)
                    
                    if check_valid(U_s, V_s, W_s, rank):
                        best_rank = rank
                        best_U = U_s
                        best_V = V_s
                        best_W = W_s
                        found = True
                        break # Found valid for this rank, try lower rank? No, we are going up. 
                        # Wait, I am iterating 12 to 23. So if I find 12, I break the loop.
                        
            except Exception as e:
                pass

        if found:
            break

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
