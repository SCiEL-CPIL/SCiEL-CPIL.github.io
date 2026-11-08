import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Target tensor: T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start with schoolbook
    rank0 = n * m * p  # 42
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if decomposition is exact. Returns True if valid."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def rank_of(U, V, W):
        return U.shape[1]
    
    # Try to reduce rank by removing terms and adjusting others
    # Use local search
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try greedy removal: for each term, see if we can remove it and adjust
    # This is complex. Let's try a different approach: randomized search for low-rank solutions
    
    # Approach: try ranks from low to high, use randomized initialization + local search
    
    # Known lower bound: at least max(n*m, m*p, n*p) = 21? No, that's not right.
    # The rank must be at least the max of the dimensions in some sense.
    
    # Let's try ranks from 21 upwards (since mp=21 is the largest)
    # Actually, the rank must be at least the "tensor rank" which is at least max(nm, mp, pn) for this type?
    # Not necessarily. Let's just search.
    
    # Try a few ranks with randomized initialization
    random.seed(42)
    np.random.seed(42)
    
    def try_rank(r, max_iters=5000):
        """Try to find a rank-r decomposition using local search."""
        nonlocal best_rank, best_U, best_V, best_W
        
        if r >= best_rank:
            return
        
        # Random initialization with small values
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Compute error
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for rr in range(r):
                            s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                        err += (s - 1.0) ** 2
            return err
        
        err = compute_error(U, V, W)
        best_local_err = err
        
        # Local search: perturb random entries
        for it in range(max_iters):
            if time.time() > deadline:
                return
            
            # Random perturbation
            idx = random.randint(0, 2)
            if idx == 0:
                row = random.randint(0, nm - 1)
                col = random.randint(0, r - 1)
                delta = random.gauss(0, 0.1)
                U[row, col] += delta
            elif idx == 1:
                row = random.randint(0, mp - 1)
                col = random.randint(0, r - 1)
                delta = random.gauss(0, 0.1)
                V[row, col] += delta
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, r - 1)
                delta = random.gauss(0, 0.1)
                W[row, col] += delta
            
            new_err = compute_error(U, V, W)
            if new_err < best_local_err:
                best_local_err = new_err
                if new_err < 1e-9:
                    # Found a solution!
                    # Check if entries are rational-ish
                    if verify(U, V, W):
                        best_rank = r
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        return
            else:
                # Revert
                if idx == 0:
                    U[row, col] -= delta
                elif idx == 1:
                    V[row, col] -= delta
                else:
                    W[row, col] -= delta
        
        return
    
    # Try ranks from 21 to 41
    for r in range(21, 42):
        if time.time() > deadline:
            break
        try_rank(r, max_iters=2000)
        if best_rank <= r:
            break
    
    # If we found a better solution, convert to list format
    if best_rank < rank0:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    else:
        U_list = U0.tolist()
        V_list = V0.tolist()
        W_list = W0.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
